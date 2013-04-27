/** backprop.c

Implements a simple backpropagation neural network.
See backprop.h for more information on using the backprop network.


Author: Joshua Petitt
Available at: https://github.com/jpmec/ann


Copyright (c) 2012-2013 Joshua Petitt
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/


#include "backprop.h"

#include <math.h>
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


#include <assert.h>
#define BACKPROP_ASSERT(_arg_)    assert(_arg_)


// The small part of the golden ratio
#define BACKPROP_MIN_GOLD    (0.3819660113)


#define USE_BACKPROP_TRACE

#ifdef USE_BACKPROP_TRACE
#include <stdio.h>
#define BACKPROP_TRACE()    printf("%s:%d\t%s\n", __FILE__, __LINE__, __FUNCTION__)
#define BACKPROP_TRACE_END()
#else
#define BACKPROP_TRACE()
#define BACKPROP_TRACE_END()
#endif



#define USE_BACKPROP_DEBUG_PRINTF

#ifdef USE_BACKPROP_DEBUG_PRINTF
#include <stdio.h>
#define BACKPROP_DEBUG_PRINTF(_fmt_, ...)    printf(_fmt_, ...)
#else
#define BACKPROP_DEBUG_PRINTF(_fmt_, ...)
#endif




/***************************************************************
 * Backprop
 **************************************************************/

#pragma mark Backprop


/** Module context structure.
 */
typedef struct Backprop
{
  void* (*onMalloc) (size_t);
  void (*onFree) (void*);

  void (*onMallocFail) (size_t);

  size_t malloc_total;
  size_t free_total;

} Backprop_t;


static Backprop_t Backprop;




void Backprop_SetMalloc(void* (*f) (size_t))
{
  BACKPROP_TRACE();

  Backprop.onMalloc = f;
}




void Backprop_SetFree(void (*f) (void*))
{
  BACKPROP_TRACE();

  Backprop.onFree = f;
}




BACKPROP_FLOAT_T Backprop_Sigmoid(BACKPROP_FLOAT_T x)
{
  BACKPROP_TRACE();

  return 1.0 / (1.0 + ((BACKPROP_FLOAT_T) exp(-x)));  // the sigmoid function
}




void Backprop_RandomSeed(unsigned long seed)
{
  BACKPROP_TRACE();

  srand((unsigned int) seed);
}




void Backprop_RandomSeedTime(void)
{
  BACKPROP_TRACE();

  Backprop_RandomSeed(time(NULL));
}




int Backprop_UniformRandomInt(void)
{
  BACKPROP_TRACE();

  return rand();
}




BACKPROP_FLOAT_T Backprop_UniformRandomFloat(void)
{
  BACKPROP_TRACE();

  const double x = rand();
  return (BACKPROP_FLOAT_T)(x / RAND_MAX);
}




static void* Backprop_Malloc(size_t size)
{
  BACKPROP_TRACE();

  void* ptr = NULL;

  if (Backprop.onMalloc)
  {
    ptr = Backprop.onMalloc(size);
  }

  else
  {
    ptr = malloc(size);    // malloc the memory
  }



  if (!ptr)
  {
    if (Backprop.onMallocFail)
    {
      Backprop.onMallocFail(size);
    }
  }
  else
  {
    Backprop.malloc_total += size;

    printf("malloc/free = %ld/%ld, ptr = %p\n", Backprop.malloc_total, Backprop.free_total, ptr);
    memset(ptr, 0, size);  // set memory to zero, not necessary but helps track down bugs...
  }

  return ptr;
}




static void Backprop_Free(void* ptr, size_t size)
{
  BACKPROP_TRACE();

  Backprop.free_total += size;

  printf("malloc/free = %ld/%ld, ptr = %p\n", Backprop.malloc_total, Backprop.free_total, ptr);

  if (Backprop.onFree)
  {
    Backprop.onFree(ptr);
    return;
  }

  free(ptr);
}




void Backprop_SetOnMallocFail(void (*f) (size_t))
{
  BACKPROP_TRACE();

  Backprop.onMallocFail = f;
}




size_t Backprop_GetMallocTotal(void)
{
  BACKPROP_TRACE();

  return Backprop.malloc_total;
}




size_t Backprop_GetFreeTotal(void)
{
  BACKPROP_TRACE();

  return Backprop.free_total;
}




size_t Backprop_GetMallocInUse(void)
{
  BACKPROP_TRACE();

  return Backprop.malloc_total - Backprop.free_total;
}




void Backprop_ResetTotals(void)
{
  BACKPROP_TRACE();

  Backprop.malloc_total = 0;
  Backprop.free_total = 0;
}




static BackpropByteArray_t BackpropByteArray_Malloc(size_t size)
{
  BACKPROP_TRACE();

  BackpropByteArray_t array;
  array.size = size;

  array.data = Backprop_Malloc(size * sizeof(BACKPROP_BYTE_T));

  return array;
}




static void BackpropByteArray_Free(BackpropByteArray_t array)
{
  BACKPROP_TRACE();

  Backprop_Free(array.data, array.size);
}




/*-------------------------------------------------------------------*
 *
 * BackpropLayer
 *
 *-------------------------------------------------------------------*/

#pragma mark BackpropLayer


/**
 * Backprop Layer structure.
 *
 * y = sig(W * x)
 *
 */
struct BackpropLayer
{
	BACKPROP_SIZE_T x_count; ///< Number of inputs to each neuron (M).
	BACKPROP_SIZE_T y_count; ///< Number of neurons in the layer (N).

	BACKPROP_FLOAT_T* W; ///< Pointer to weight matrix  [NxM].
	BACKPROP_FLOAT_T* g; ///< Pointer to layer gradient [Nx1].

	BACKPROP_FLOAT_T* x; ///< Pointer to layer input    [Mx1].
	BACKPROP_FLOAT_T* y; ///< Pointer to layer output   [Nx1].

};




static size_t BackpropLayer_x_MallocSize(BACKPROP_SIZE_T x_count, BACKPROP_SIZE_T y_count)
{
  BACKPROP_TRACE();

  return x_count * sizeof(BACKPROP_FLOAT_T);
}




static size_t BackpropLayer_W_MallocSize(BACKPROP_SIZE_T x_count, BACKPROP_SIZE_T y_count)
{
  BACKPROP_TRACE();

  return x_count * y_count * sizeof(BACKPROP_FLOAT_T);
}




static size_t BackpropLayer_y_MallocSize(BACKPROP_SIZE_T x_count, BACKPROP_SIZE_T y_count)
{
  BACKPROP_TRACE();

  return y_count * sizeof(BACKPROP_FLOAT_T);
}




static size_t BackpropLayer_g_MallocSize(BACKPROP_SIZE_T x_count, BACKPROP_SIZE_T y_count)
{
  BACKPROP_TRACE();

  // g is same length as output y
  return y_count * sizeof(BACKPROP_FLOAT_T);
}




static size_t BackpropLayer_MallocInternalSize(BACKPROP_SIZE_T x_count, BACKPROP_SIZE_T y_count)
{
  BACKPROP_TRACE();

  return   BackpropLayer_x_MallocSize(x_count, y_count)
  + BackpropLayer_W_MallocSize(x_count, y_count)
  + BackpropLayer_y_MallocSize(x_count, y_count)
  + BackpropLayer_g_MallocSize(x_count, y_count);
}




static size_t BackpropLayer_MallocSize(BACKPROP_SIZE_T x_count, BACKPROP_SIZE_T y_count)
{
  BACKPROP_TRACE();

  return sizeof(struct BackpropLayer) + BackpropLayer_MallocInternalSize(x_count, y_count);
}




static void BackpropLayer_MallocInternal(BackpropLayer_t* self, BACKPROP_SIZE_T x_count, BACKPROP_SIZE_T y_count)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  BACKPROP_ASSERT(x_count);
  BACKPROP_ASSERT(y_count);

  const BACKPROP_SIZE_T x_size = BackpropLayer_x_MallocSize(x_count, y_count);
  self->x = Backprop_Malloc(x_size);
  self->x_count = x_count;

  const BACKPROP_SIZE_T W_size = BackpropLayer_W_MallocSize(x_count, y_count);
  self->W = Backprop_Malloc(W_size);

  const BACKPROP_SIZE_T y_size = BackpropLayer_y_MallocSize(x_count, y_count);
  self->y = Backprop_Malloc(y_size);
  self->y_count = y_count;

  const BACKPROP_SIZE_T g_size = BackpropLayer_g_MallocSize(x_count, y_count);
  self->g = Backprop_Malloc(g_size);

}




static void BackpropLayer_FreeInternal(BackpropLayer_t* layer)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(layer);

  if (!layer)
  {
    printf("no layer\n");
    return;
  }

  printf("calculating\n");

  const size_t x_count = layer->x_count;
  const size_t y_count = layer->y_count;

  printf("%ld, %ld\n", x_count, y_count);

  const size_t x_size = BackpropLayer_x_MallocSize(x_count, y_count);
  const size_t y_size = BackpropLayer_y_MallocSize(x_count, y_count);
  const size_t W_size = BackpropLayer_W_MallocSize(x_count, y_count);
  const size_t g_size = BackpropLayer_g_MallocSize(x_count, y_count);

  printf("%ld, %ld, %ld, %ld\n", x_size, y_size, W_size, g_size);

  Backprop_Free(layer->x, x_size);
  Backprop_Free(layer->y, y_size);

  Backprop_Free(layer->W, W_size);
  Backprop_Free(layer->g, g_size);
}




struct BackpropLayer* BackpropLayer_Malloc(BACKPROP_SIZE_T x_size, BACKPROP_SIZE_T y_size)
{
  BACKPROP_TRACE();

  struct BackpropLayer* ptr = Backprop_Malloc(sizeof(struct BackpropLayer));

  printf("malloc layer = %p\n", ptr);
  printf("malloc layer x_count = %ld\n", ptr->x_count);
  printf("malloc layer y_count = %ld\n", ptr->y_count);

  BackpropLayer_MallocInternal(ptr, x_size, y_size);

  printf("malloc layer x_count = %ld\n", ptr->x_count);
  printf("malloc layer y_count = %ld\n", ptr->y_count);

  return ptr;
}




void BackpropLayer_Free(struct BackpropLayer* layer)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(layer);

  printf("layer = %p\n", layer);

  BackpropLayer_FreeInternal(layer);
  Backprop_Free(layer, sizeof(struct BackpropLayer));
}




BACKPROP_SIZE_T BackpropLayer_GetXCount(const struct BackpropLayer* self)
{
  BACKPROP_TRACE();
  BACKPROP_ASSERT(self);
  return self->x_count;
}




BACKPROP_FLOAT_T BackpropLayer_GetX(const struct BackpropLayer* self, size_t i)
{
  BACKPROP_TRACE();
  BACKPROP_ASSERT(self);
  return self->x[i];
}




void BackpropLayer_SetAtX(struct BackpropLayer* self, BACKPROP_SIZE_T i, BACKPROP_FLOAT_T value)
{
  BACKPROP_TRACE();
  BACKPROP_ASSERT(self);
  self->x[i] = value;
}




BACKPROP_SIZE_T BackpropLayer_GetYCount(const struct BackpropLayer* self)
{
  BACKPROP_TRACE();
  BACKPROP_ASSERT(self);
  return self->y_count;
}




BACKPROP_FLOAT_T BackpropLayer_GetY(const struct BackpropLayer* self, size_t i)
{
  BACKPROP_TRACE();
  BACKPROP_ASSERT(self);
  return self->y[i];
}




void BackpropLayer_SetAtY(struct BackpropLayer* self, size_t i, BACKPROP_FLOAT_T value)
{
  BACKPROP_TRACE();
  BACKPROP_ASSERT(self);
  self->y[i] = value;
}




BACKPROP_FLOAT_T BackpropLayer_GetAtG(const struct BackpropLayer* self, size_t i)
{
  BACKPROP_TRACE();
  BACKPROP_ASSERT(self);
  return self->g[i];
}




void BackpropLayer_SetAtG(const struct BackpropLayer* self, size_t i, BACKPROP_FLOAT_T value)
{
  BACKPROP_TRACE();
  BACKPROP_ASSERT(self);
  self->g[i] = value;
}




BACKPROP_FLOAT_T BackpropLayer_GetAtW(const struct BackpropLayer* self, BACKPROP_SIZE_T i)
{
  BACKPROP_TRACE();
  BACKPROP_ASSERT(self);
  return self->W[i];
}


void BackpropLayer_SetAtW(struct BackpropLayer* self, size_t i, BACKPROP_FLOAT_T value)
{
  BACKPROP_TRACE();
  BACKPROP_ASSERT(self);
  self->W[i] = value;
}



BACKPROP_FLOAT_T* BackpropLayer_GetW(struct BackpropLayer* self)
{
  BACKPROP_TRACE();
  BACKPROP_ASSERT(self);
  return self->W;
}



const BACKPROP_FLOAT_T* BackpropLayer_GetConstW(const struct BackpropLayer* self)
{
  BACKPROP_TRACE();
  BACKPROP_ASSERT(self);
  return self->W;
}




BACKPROP_SIZE_T BackpropLayer_WeightCount(const BackpropLayer_t* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  return self->x_count * self->y_count;
}




/** Generate a uniform pseudo-random number in the range of (-1, 1).
 */
BACKPROP_FLOAT_T BackpropLayer_RandomWeight(void)
{
  BACKPROP_TRACE();

  return 2.0 * Backprop_UniformRandomFloat() - 1.0;
}




BACKPROP_SIZE_T Backprop_RandomArrayIndex(size_t lower, size_t upper)
{
  BACKPROP_TRACE();

  if (lower >= upper)
  {
    return lower;
  }

  else
  {
    BACKPROP_SIZE_T value = (BACKPROP_SIZE_T) (Backprop_UniformRandomFloat() * (upper - lower) + lower);

    if (value >= upper)
    {
      value = upper - 1;
    }

    return value;
  }
}




static int BackpropLayer_IsSimilar(const BackpropLayer_t* self, const BackpropLayer_t* other)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  BACKPROP_ASSERT(other);

  if (self == other)
  {
    return 0; // same is not similar
  }

  return ((self->x_count == other->x_count) && (self->y_count == other->y_count));
}




static void BackpropLayer_DeepCopy(const BackpropLayer_t* self, BackpropLayer_t* dest)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  BACKPROP_ASSERT(dest);

  memcpy(dest->x, self->x, self->x_count * sizeof(BACKPROP_FLOAT_T));
  memcpy(dest->y, self->y, self->y_count * sizeof(BACKPROP_FLOAT_T));
  memcpy(dest->W, self->W, self->x_count * self->y_count * sizeof(BACKPROP_FLOAT_T));
  memcpy(dest->g, self->g, self->y_count * sizeof(BACKPROP_FLOAT_T));
}




void BackpropLayer_Randomize(BackpropLayer_t* self, BACKPROP_FLOAT_T gain)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  {
    size_t count = BackpropLayer_WeightCount(self);

    BACKPROP_FLOAT_T* W = self->W;

    do
    {
        *W += gain * BackpropLayer_RandomWeight();
        ++W;

    } while (--count);
  }
}




void BackpropLayer_Identity(BackpropLayer_t* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  {
    BACKPROP_FLOAT_T* W = self->W;

    BACKPROP_SIZE_T y = self->y_count;
    do
    {
      BACKPROP_SIZE_T x = self->x_count;
      do
      {
        *W = (x == y);
        ++W;
      } while (--x);

    } while (--y);
  }
}




void BackpropLayer_Prune(BackpropLayer_t* self, BACKPROP_FLOAT_T threshold)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  {
    size_t count = BackpropLayer_WeightCount(self);

    BACKPROP_FLOAT_T* W = self->W;
    do
    {
      if (threshold > fabs(*W))
      {
        *W = 0.0;
      }

      ++W;

    } while (--count);
  }
}




void BackpropLayer_Round(BackpropLayer_t* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  {
    size_t count = BackpropLayer_WeightCount(self);

    BACKPROP_FLOAT_T* W = self->W;
    do
    {
      *W = round(*W);
      ++W;

    } while (--count);
  }
}




void BackpropLayer_Reset(BackpropLayer_t* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  {
    memset(self->x, 0, self->x_count * sizeof(BACKPROP_FLOAT_T));
    memset(self->y, 0, self->y_count * sizeof(BACKPROP_FLOAT_T));
    memset(self->g, 0, self->y_count * sizeof(BACKPROP_FLOAT_T));
  }
}




static void BackpropLayer_Input(BackpropLayer_t* self, const BACKPROP_FLOAT_T* values, BACKPROP_SIZE_T values_size)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  BACKPROP_ASSERT(values);
  BACKPROP_ASSERT(values_size);
  BACKPROP_ASSERT(values_size <= self->x_count);
  {
    BACKPROP_FLOAT_T* x = self->x;

    for(size_t i = 0; i < values_size; ++i)
    {
        x[i] = values[i];
    }
  }
}




void BackpropLayer_Activate(BackpropLayer_t* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  {
    size_t y_count = self->y_count;

    const BACKPROP_FLOAT_T* W = self->W;
    BACKPROP_FLOAT_T* y = self->y;

    BACKPROP_ASSERT(y_count);
    BACKPROP_ASSERT(W);
    BACKPROP_ASSERT(y);

    // for each neuron in layer
    do
    {
      const BACKPROP_FLOAT_T* x = self->x;
      BACKPROP_ASSERT(x);
      {
        // calculate weighted input
        BACKPROP_FLOAT_T sum = 0;

        size_t x_count = self->x_count;
        do
        {
          sum += (*W) * (*x);

          ++W;
          ++x;

        } while (--x_count);

        // compute activation function and save output of layer
        *y = (BACKPROP_FLOAT_T) Backprop_Sigmoid(sum);

        ++y;
      }
    } while (--y_count);
  }
}




static void BackpropLayer_WeightedGradient(const BackpropLayer_t* l, BACKPROP_FLOAT_T* Wg)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(l);
  BACKPROP_ASSERT(Wg);

  for(size_t i=0; i < l->x_count; ++i)
  {
    Wg[i]=0;

    for(size_t j=0; j < l->y_count; ++j)
    {
      Wg[i] += (*(l->W + j*l->x_count + i)) * (l->g[j]);
    }
  }
}




BACKPROP_SIZE_T BackpropLayer_GetWeightsCount(const BackpropLayer_t* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return self->x_count * self->y_count;
}




static BACKPROP_SIZE_T BackpropLayer_GetWeightsSize(const BackpropLayer_t* self)
{
  BACKPROP_TRACE();

  return BackpropLayer_GetWeightsCount(self) * sizeof(BACKPROP_FLOAT_T);
}




BACKPROP_FLOAT_T BackpropLayer_GetWeightsSum(const BackpropLayer_t* self)
{
  BACKPROP_TRACE();
  {
    BACKPROP_FLOAT_T sum = 0.0;
    BACKPROP_SIZE_T count = BackpropLayer_GetWeightsCount(self);
    BACKPROP_FLOAT_T* w = self->W;

    do
    {
      sum += *w;
      ++w;
    } while (--count);

    return sum;
  }
}




BACKPROP_FLOAT_T BackpropLayer_GetWeightsMean(const BackpropLayer_t* self)
{
  BACKPROP_TRACE();

  const BACKPROP_FLOAT_T sum = BackpropLayer_GetWeightsSum(self);
  const BACKPROP_SIZE_T count = BackpropLayer_GetWeightsCount(self);

  return sum / count;
}




BACKPROP_FLOAT_T BackpropLayer_GetWeightsStdDev(const BackpropLayer_t* self)
{
  BACKPROP_TRACE();

  const BACKPROP_SIZE_T count = BackpropLayer_GetWeightsCount(self);

  if (0 == count)
  {
    return 0.0;
  }
  else
  {
    const BACKPROP_FLOAT_T mean = BackpropLayer_GetWeightsMean(self);

    BACKPROP_FLOAT_T ddsum = 0.0;
    BACKPROP_FLOAT_T* w = self->W;

    BACKPROP_SIZE_T i = count;
    do
    {
      const BACKPROP_FLOAT_T d = *w - mean;

      ddsum += d*d;

      ++w;
    } while (--i);

    return sqrt(ddsum / count);
  }
}




BACKPROP_SIZE_T BackpropLayer_GetSize(const BackpropLayer_t* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return BackpropLayer_MallocSize(self->x_count, self->y_count);
}




/*-------------------------------------------------------------------*
 *
 * BackpropLayersArray
 *
 *-------------------------------------------------------------------*/


size_t BackpropLayersArray_GetCount(const struct BackpropLayersArray* self)
{
  BACKPROP_TRACE();
  BACKPROP_ASSERT(self);

  return self->count;
}




struct BackpropLayer* BackpropLayersArray_GetLayer(struct BackpropLayersArray* self, size_t i)
{
  BACKPROP_TRACE();
  BACKPROP_ASSERT(self);

  return self->data + i;
}




const struct BackpropLayer* BackpropLayersArray_GetConstLayer(const struct BackpropLayersArray* self, size_t i)
{
  return BackpropLayersArray_GetLayer((struct BackpropLayersArray*) self, i);
}




/*-------------------------------------------------------------------*
 *
 * BackpropNetwork
 *
 *-------------------------------------------------------------------*/

#pragma mark BackpropNetwork



/**
 * Backprop Network structure.
 */
struct BackpropNetwork
{
  BackpropByteArray_t x;         ///< Byte array input, each bit represents neuron input.
  BackpropLayersArray_t layers;
  BackpropByteArray_t y;         ///< Byte output array, each bit represents neuron output.

  BACKPROP_FLOAT_T jitter;       ///< Amount of jitter associated with input.
};




int BackpropNetwork_IsValid(const struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  return    (0 != self)
         && (0 < BackpropNetwork_GetXSize(self))
         && (0 < BackpropNetwork_GetYSize(self));

}




const BackpropByteArray_t* BackpropNetwork_GetX(const struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return &self->x;
}




BACKPROP_SIZE_T BackpropNetwork_GetXSize(const struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return self->x.size;
}




const BackpropByteArray_t* BackpropNetwork_GetY(const struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return &self->y;
}




BACKPROP_SIZE_T BackpropNetwork_GetYSize(const struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return self->y.size;
}




const BackpropLayersArray_t* BackpropNetwork_GetLayers(const struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return &self->layers;
}



BACKPROP_SIZE_T BackpropNetwork_GetLayersCount(const struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return self->layers.count;
}





BACKPROP_FLOAT_T BackpropNetwork_GetJitter(const struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return self->jitter;
}




void BackpropNetwork_SetJitter(struct BackpropNetwork* self, BACKPROP_FLOAT_T jitter)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  self->jitter = jitter;
}




size_t BackpropNetwork_MallocSize(BACKPROP_SIZE_T x_size, BACKPROP_SIZE_T y_size, BACKPROP_SIZE_T layers_count)
{
  BACKPROP_TRACE();

  size_t total_size = sizeof(struct BackpropNetwork)
                      + x_size * sizeof(BACKPROP_BYTE_T)
                      + y_size * sizeof(BACKPROP_BYTE_T)
                      + layers_count * sizeof(BackpropLayer_t);

  if (1 == layers_count)
  {
    BackpropLayer_MallocInternalSize(x_size * CHAR_BIT, y_size * CHAR_BIT);
  }
  else
  {
    // compute size of hidden layers
    BACKPROP_SIZE_T hid_count = CHAR_BIT * ((x_size > y_size) ? x_size : y_size);

    total_size += (layers_count - 1) * BackpropLayer_MallocInternalSize(x_size * CHAR_BIT, hid_count);

    total_size += BackpropLayer_MallocInternalSize(hid_count, y_size * CHAR_BIT);
  }

  return total_size;
}




struct BackpropNetwork* BackpropNetwork_Malloc(BACKPROP_SIZE_T x_size, BACKPROP_SIZE_T y_size, BACKPROP_SIZE_T layers_count, bool chain_layers)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(x_size);
  BACKPROP_ASSERT(y_size);
  BACKPROP_ASSERT(layers_count);
  {
    struct BackpropNetwork* ptr = Backprop_Malloc(sizeof(struct BackpropNetwork));

    ptr->x = BackpropByteArray_Malloc(x_size);
    ptr->y = BackpropByteArray_Malloc(y_size);

    ptr->layers.count = layers_count;
    ptr->layers.data = Backprop_Malloc(layers_count * sizeof(BackpropLayer_t));

    if (layers_count == 1)
    {
      BackpropLayer_MallocInternal(&ptr->layers.data[0], x_size * CHAR_BIT, y_size * CHAR_BIT);
    }
    else
    {
      // compute size of hidden layers
      BACKPROP_SIZE_T hid_count = CHAR_BIT * ((x_size > y_size) ? x_size : y_size);

      // malloc input layer
      BackpropLayer_MallocInternal(&ptr->layers.data[0], x_size * CHAR_BIT, hid_count);

      // malloc hidden layers
      {
        size_t i = 1;
        for (; i < layers_count - 1; ++i)
        {
          BackpropLayer_MallocInternal(&ptr->layers.data[i], hid_count, hid_count);
        }

        // malloc output layer
        BackpropLayer_MallocInternal(&ptr->layers.data[i], hid_count, y_size * CHAR_BIT);
      }
    }

    return ptr;
  }
}




void BackpropNetwork_Free(struct BackpropNetwork* network)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(network);

  for (size_t i = 0; i < network->layers.count; ++i)
  {
    BackpropLayer_FreeInternal(&network->layers.data[i]);
  }

  Backprop_Free(network->x.data, network->x.size * sizeof(BACKPROP_BYTE_T));
  Backprop_Free(network->y.data, network->y.size * sizeof(BACKPROP_BYTE_T));
  Backprop_Free(network->layers.data, network->layers.count * sizeof(BackpropLayer_t));

  Backprop_Free(network, sizeof(struct BackpropNetwork));
}




/** Copy input bits to the network first layer inputs.
 */
static void BackpropNetwork_InputToLayer0(struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  {
    // Input each bit value
    BackpropLayer_t* layer0 = BackpropNetwork_GetFirstLayer(self);

    BACKPROP_FLOAT_T* x = layer0->x;

    BACKPROP_ASSERT((self->x.size * CHAR_BIT) == layer0->x_count);

    for (size_t i = 0; i < self->x.size; ++i)
    {
      // convert bits to float
      BACKPROP_BYTE_T bits = self->x.data[i];

      size_t b = CHAR_BIT;
      do
      {
        // set value to either 0.0 or 1.0 +/- jitter
        *x = (bits & 1) + 2.0 * self->jitter * Backprop_UniformRandomFloat() - 1.0;
        bits >>= 1;
        ++x;

      } while (--b);
    }
  }
}




static int BackpropNetwork_IsSimilar(const struct BackpropNetwork* self, const struct BackpropNetwork* other)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  BACKPROP_ASSERT(other);

  if (self == other)
  {
    return 0; // same is not similar
  }

  if ((self->x.size != other->x.size) || (self->y.size != other->y.size) || (self->layers.count != other->layers.count))
  {
    return 0;
  }

  for (size_t i = 0; i < self->layers.count; ++i)
  {
    if (!BackpropLayer_IsSimilar(&self->layers.data[i], &other->layers.data[i]))
    {
      return 0;
    }
  }

  return 1;
}




static void BackpropNetwork_DeepCopy(const struct BackpropNetwork* self, struct BackpropNetwork* dest)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  BACKPROP_ASSERT(dest);

  if (!BackpropNetwork_IsSimilar(self, dest))
  {
    return;
  }

  for (size_t i = 0; i < self->layers.count; ++i)
  {
    BackpropLayer_DeepCopy(&self->layers.data[i], &dest->layers.data[i]);
  }
}




void BackpropNetwork_Input(struct BackpropNetwork* self, const BACKPROP_BYTE_T* values, BACKPROP_SIZE_T values_size)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  BACKPROP_ASSERT(values);
  BACKPROP_ASSERT(values_size);

  BACKPROP_ASSERT(self->x.size == values_size);

  // copy data into input array
  for (size_t i = 0; i < values_size; ++i)
  {
    self->x.data[i] = values[i];
  }
}




void BackpropNetwork_InputCStr(struct BackpropNetwork* self, const char* values)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  BACKPROP_ASSERT(values);
  {
    const BACKPROP_SIZE_T values_size = strlen(values);

    if (!values_size)
    {
      return;
    }

    BackpropNetwork_Input(self, (const BACKPROP_BYTE_T*) values, values_size);
  }
}




BACKPROP_SIZE_T BackpropNetwork_GetOutput(const struct BackpropNetwork* self, BACKPROP_BYTE_T* values, BACKPROP_SIZE_T values_size)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  BACKPROP_ASSERT(values);
  BACKPROP_ASSERT(values_size);
  {
    const BACKPROP_BYTE_T* data = self->y.data;
    const BACKPROP_SIZE_T y_size = self->y.size;

    BACKPROP_SIZE_T i = (y_size < values_size) ? y_size : values_size;
    do
    {
      *values = *data;
      ++values;
      ++data;

    } while (--i);

    return y_size;
  }
}




size_t BackpropNetwork_GetOutputCStr(const struct BackpropNetwork* self, char* str, size_t str_size)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  BACKPROP_ASSERT(str);

  if (!str_size)
  {
    return 0;
  }

  return BackpropNetwork_GetOutput(self, (BACKPROP_BYTE_T*) str, str_size);
}




static void BackpropNetwork_ActivateLayers(struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  BACKPROP_ASSERT(self->layers.count > 0);
  {
    // activate all layers except last
    const size_t layers_count = self->layers.count - 1;

    for(size_t i=0; i < layers_count; ++i)
    {
      BackpropLayer_t* layer = &self->layers.data[i];

      BackpropLayer_Activate(layer);
      BackpropLayer_Input(&self->layers.data[i + 1], layer->y, layer->y_count);
    }

    // activate the last layer
    BackpropLayer_Activate(&self->layers.data[layers_count]);
  }
}




BackpropLayer_t* BackpropNetwork_GetFirstLayer(struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  BACKPROP_ASSERT(self->layers.data);
  BACKPROP_ASSERT(self->layers.count);

  return &self->layers.data[0];
}




BackpropLayer_t* BackpropNetwork_GetLastLayer(struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  BACKPROP_ASSERT(self->layers.data);
  BACKPROP_ASSERT(self->layers.count);

  if (self->layers.count)
  {
    return &self->layers.data[self->layers.count - 1];
  }
  else
  {
    return &self->layers.data[0];
  }
}




const BackpropLayer_t* BackpropNetwork_GetConstLastLayer(const struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  return BackpropNetwork_GetLastLayer((struct BackpropNetwork*) self);
}




static void BackpropNetwork_LastLayerToOutput(struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  {
    const BackpropLayer_t* last_layer = BackpropNetwork_GetConstLastLayer(self);
    BACKPROP_ASSERT(last_layer);
    {
      // Input each bit value
      BACKPROP_FLOAT_T* y = last_layer->y;
      BACKPROP_ASSERT((self->y.size * CHAR_BIT) == last_layer->y_count);

      for (size_t i = 0; i < self->y.size; ++i)
      {
        // convert bits to float
        BACKPROP_BYTE_T bits = 0;
        BACKPROP_SIZE_T bit_shift = 0;

        size_t b = CHAR_BIT;
        do
        {
          // set value to 1 if greater than 0.5, otherwise set to 0
          const BACKPROP_BYTE_T bit = (*y) > 0.5;
          bits |= (bit << bit_shift);

          ++bit_shift;
          ++y;

        } while (--b);

        self->y.data[i] = bits;
      }
    }
  }
}




void BackpropNetwork_Activate(struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  // convert byte input in to network layer input
  BackpropNetwork_InputToLayer0(self);

  // activate the network layers
  BackpropNetwork_ActivateLayers(self);

  // copy to output bits
  BackpropNetwork_LastLayerToOutput(self);
}




void BackpropNetwork_Randomize(struct BackpropNetwork* self, BACKPROP_FLOAT_T gain, unsigned int seed)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  Backprop_RandomSeed(seed);

  for(size_t i = 0; i < self->layers.count; ++i)
  {
    BackpropLayer_Randomize(self->layers.data + i, gain);
  }
}




void BackpropNetwork_Round(struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  for(size_t i = 0; i < self->layers.count; ++i)
  {
    BackpropLayer_Round(self->layers.data + i);
  }
}




void BackpropNetwork_Identity(struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  for(size_t i = 0; i < self->layers.count; ++i)
  {
    BackpropLayer_Identity(self->layers.data + i);
  }
}




void BackpropNetwork_Reset(struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  memset(self->x.data, 0, self->x.size);
  memset(self->y.data, 0, self->y.size);

  for (size_t i = 0; i < self->layers.count; ++i)
  {
    BackpropLayer_Reset(&self->layers.data[i]);
  }
}




void BackpropNetwork_Prune(struct BackpropNetwork* self, BACKPROP_FLOAT_T threshold)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  for(size_t i = 0; i < self->layers.count; ++i)
  {
    BackpropLayer_Prune(&self->layers.data[i], threshold);
  }
}




BACKPROP_SIZE_T BackpropNetwork_GetWeightsCount(const struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  BACKPROP_SIZE_T total_count = 0;

  for(size_t i = 0; i < self->layers.count; ++i)
  {
    total_count += BackpropLayer_WeightCount(&self->layers.data[i]);
  }

  return total_count;
}




BACKPROP_SIZE_T BackpropNetwork_GetWeightsSize(const struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return BackpropNetwork_GetWeightsCount(self) * sizeof(BACKPROP_FLOAT_T);
}




BACKPROP_FLOAT_T BackpropNetwork_GetWeightsSum(const struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  BACKPROP_FLOAT_T result = 0;

  for(size_t i = 0; i < self->layers.count; ++i)
  {
    result += BackpropLayer_GetWeightsSum(self->layers.data + i);
  }

  return result;
}




BACKPROP_FLOAT_T BackpropNetwork_GetWeightsMean(const struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  const BACKPROP_FLOAT_T sum = BackpropNetwork_GetWeightsSum(self);
  const BACKPROP_SIZE_T count = BackpropNetwork_GetWeightsCount(self);

  return sum / count;
}




BACKPROP_FLOAT_T BackpropNetwork_GetWeightsStdDev(const struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

//  const BACKPROP_FLOAT_T mean = BackpropNetwork_GetWeightsMean(self);

//  BACKPROP_FLOAT_T result = 0;

//  BACKPROP_FLOAT_T dev_sum = 0;

//  BACKPROP_FLOAT_T* W = self->layers.data[0];

//  for(size_t i = 0; i < self->layers.count; ++i)
//  {

//  }

//  return result;
  return 0.0;
}




static BACKPROP_SIZE_T BackpropNetwork_GetLayersSize(const struct BackpropNetwork* self)
{
  BACKPROP_TRACE();

  const struct BackpropLayersArray* layers = BackpropNetwork_GetLayers(self);

  BACKPROP_SIZE_T total_size = 0;

  const BACKPROP_SIZE_T count = layers->count;
  for (BACKPROP_SIZE_T i = 0; i < count; ++i)
  {
    total_size += BackpropLayer_GetSize(layers->data + i);
  }

  return total_size;
}




void BackpropNetwork_GetStats(const struct BackpropNetwork* self, BackpropNetworkStats_t* stats)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  BACKPROP_ASSERT(stats);

  stats->x_size = self->x.size;
  stats->y_size = self->y.size;
  stats->layers_count = self->layers.count;

  stats->layers_W_count = BackpropNetwork_GetWeightsCount(self);
  stats->layers_W_size = BackpropNetwork_GetWeightsSize(self);
  stats->layers_W_avg = BackpropNetwork_GetWeightsMean(self);
  stats->layers_W_stddev = BackpropNetwork_GetWeightsStdDev(self);

  stats->layers_size = BackpropNetwork_GetLayersSize(self);
}







/*-------------------------------------------------------------------*
 *
 * BackpropTrainingSet
 *
 *-------------------------------------------------------------------*/

#pragma mark BackpropTrainingSet


BackpropTrainingSet_t* BackpropTrainingSet_Malloc(size_t count, size_t x_size, size_t y_size)
{
  BACKPROP_TRACE();

  BackpropTrainingSet_t* self = Backprop_Malloc(sizeof(BackpropTrainingSet_t));

  self->dims.count = count;

  self->dims.x_size = x_size;
  self->dims.y_size = y_size;

  if (count && x_size)
  {
    self->x = Backprop_Malloc(count * x_size);
  }

  if (count && y_size)
  {
    self->y = Backprop_Malloc(count * y_size);
  }

  return self;
}




void BackpropTrainingSet_Free(BackpropTrainingSet_t* self)
{
  BACKPROP_TRACE();

  size_t count = self->dims.count;
  size_t x_size = self->dims.x_size;
  size_t y_size = self->dims.y_size;

  Backprop_Free(self->x, count * x_size);
  Backprop_Free(self->y, count * y_size);
  Backprop_Free(self, sizeof(BackpropTrainingSet_t));
}




size_t BackpropTrainingSet_GetXSize(BackpropTrainingSet_t* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return self->dims.x_size;
}




size_t BackpropTrainingSet_GetYSize(BackpropTrainingSet_t* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return self->dims.y_size;
}




void BackpropTrainingSet_GetPair(BackpropTrainingSet_t* self, BACKPROP_SIZE_T index, BACKPROP_BYTE_T* x, BACKPROP_BYTE_T* y)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  BACKPROP_ASSERT(x);
  BACKPROP_ASSERT(y);

  memcpy(x, self->x + index * self->dims.x_size, self->dims.x_size);
}


//void BackpropTrainingSet_SetPair(BackpropTrainingSet_t* self, BACKPROP_SIZE_T index, const BACKPROP_BYTE_T* x, const BACKPROP_BYTE_T* y)
//{
//  BACKPROP_ASSERT(self);
//  BACKPROP_ASSERT(x);
//  BACKPROP_ASSERT(y);
//}




/*-------------------------------------------------------------------*
 *
 * BackpropLearningAccelerator
 *
 *-------------------------------------------------------------------*/

#pragma mark BackpropLearningAccelerator


void BackpropLearningAccelerator_SetToDefault(BackpropLearningAccelerator_t* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  memset(self, 0, sizeof(BackpropLearningAccelerator_t));

  self->min_learning_rate = 0.1;
  self->max_learning_rate = 0.9;
  self->acceleration = 0.1;
}




BACKPROP_FLOAT_T BackpropLearningAccelerator_Accelerate(BackpropLearningAccelerator_t* self, BACKPROP_FLOAT_T learning_rate, BACKPROP_FLOAT_T error_now, BACKPROP_FLOAT_T error_prev)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);
  BACKPROP_ASSERT(self->min_learning_rate <= self->max_learning_rate);
  {
    const BACKPROP_FLOAT_T max_learning_rate = self->max_learning_rate;
    BACKPROP_FLOAT_T min_learning_rate = self->min_learning_rate;
    BACKPROP_FLOAT_T acceleration = self->acceleration;
    const BACKPROP_FLOAT_T error_diff = error_now - error_prev;

    if (min_learning_rate > max_learning_rate)
    {
      min_learning_rate = max_learning_rate;
    }

    if (error_diff > 0) // more error is bad
    {
      learning_rate = min_learning_rate;
      acceleration = 0;
    }
    else // less error is good
    {
      if (learning_rate < min_learning_rate)
      {
        learning_rate = min_learning_rate;
        acceleration = 0;
      }
      else if (learning_rate > max_learning_rate)
      {
        learning_rate = max_learning_rate;
        acceleration = 0;
      }
    }

    return learning_rate + acceleration;
  }
}








/*-------------------------------------------------------------------*
 *
 * BackpropTrainer
 *
 *-------------------------------------------------------------------*/

#pragma mark BackpropTrainer


/** Backprop Trainer structure.
 *  Holds parameters that affect network training.
 */
struct BackpropTrainer
{
  BACKPROP_FLOAT_T error_tolerance;                   ///< Minimum allowable error for network to be considered trained.
  BACKPROP_FLOAT_T learning_rate;                     ///< Weight adjustment factor used when training networks.
  BACKPROP_FLOAT_T mutation_rate;                     ///< Amount of mutation applied to network weight matrices.
  BACKPROP_FLOAT_T momentum_rate;

  BackpropLearningAccelerator_t learning_accelerator;

  BACKPROP_SIZE_T max_reps;                           ///< Maximum number of training repetitions given to a single neuron.
  BACKPROP_SIZE_T max_batch_sets;                     ///< Maximum number of training sets ran per batch.
  BACKPROP_SIZE_T max_batches;                        ///< Maximum number of batches per training session.

  BACKPROP_FLOAT_T stagnate_tolerance;
  BACKPROP_SIZE_T max_stagnate_sets;
  BACKPROP_SIZE_T max_stagnate_batches;

  BACKPROP_FLOAT_T min_set_weight_correction_limit;   ///< Minimum weight correction that must be reached by each training set or else trainer gives up training.
  BACKPROP_FLOAT_T min_batch_weight_correction_limit; ///< Minimum weight correction that must be reached by each batch or else trainer gives up training.

  BACKPROP_FLOAT_T batch_prune_threshold;             ///< Weight threshold for network pruning.
  BACKPROP_FLOAT_T batch_prune_rate;                  ///< Amount that pruning threshold is increased for each batch.

  BACKPROP_FLOAT_T training_ratio;                    ///< Ratio of training set pairs to be used as training input.  1.0 means all training pairs, 0.5 means on average only half are used.

  BACKPROP_FLOAT_T** W_prev;                          ///< Array of pointers to layer weight matrices that store the previous training weights.

  struct BackpropTrainerEvents events;                ///< Structure of event callback function pointers.

};




static BACKPROP_FLOAT_T BackpropTrainer_ComputeByteError(BACKPROP_BYTE_T byte1, BACKPROP_BYTE_T byte2)
{
  BACKPROP_TRACE();

  BACKPROP_FLOAT_T error = 0;

  size_t b = CHAR_BIT;
  do
  {
    error += ((byte1 & 1) != (byte2 & 1));

    byte1 >>= 1;
    byte2 >>= 1;

  } while (--b);

  return error;
}




static BACKPROP_FLOAT_T BackpropTrainer_ComputeError(const struct BackpropNetwork* network, const BACKPROP_BYTE_T* yd, BACKPROP_SIZE_T yd_size)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(network);
  BACKPROP_ASSERT(yd);
  BACKPROP_ASSERT(yd_size);
  BACKPROP_ASSERT(yd_size == network->y.size);
  {
    BACKPROP_BYTE_T* y = network->y.data;

    BACKPROP_FLOAT_T error = 0;
    do
    {
      error += BackpropTrainer_ComputeByteError(*y, *yd);

      ++y;
      ++yd;

    } while (--yd_size);

    return error;
  }
}




size_t BackpropTrainer_MallocSize(const struct BackpropNetwork* network)
{
  BACKPROP_TRACE();

  return sizeof(BackpropTrainer_t);
}




BackpropTrainer_t* BackpropTrainer_Malloc(struct BackpropNetwork* network)
{
  BACKPROP_TRACE();

  return Backprop_Malloc(BackpropTrainer_MallocSize(network));
}




void BackpropTrainer_Free(BackpropTrainer_t* trainer)
{
  BACKPROP_TRACE();

  Backprop_Free(trainer, sizeof(BackpropTrainer_t));
}



struct BackpropTrainerEvents* BackpropTrainer_GetEvents(BackpropTrainer_t* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return &self->events;
}




void BackpropTrainer_SetToDefault(BackpropTrainer_t* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  memset(self, 0, sizeof(BackpropTrainer_t));

  self->error_tolerance = 0;
  self->max_reps = 0xFF;
  self->max_batch_sets = 0xFF;
  self->learning_rate = BACKPROP_MIN_GOLD;
  self->mutation_rate = 0.001;
  self->momentum_rate = 0.01;
  self->stagnate_tolerance = 1;
  self->max_stagnate_sets = 0x0F;
  self->max_stagnate_batches = 0x0F;
  self->min_batch_weight_correction_limit = 0.1;
  self->min_set_weight_correction_limit = 0.1;

  self->batch_prune_rate = 0.1;
  self->batch_prune_threshold = 0.5;
  self->max_batches = 0xFF;

  self->training_ratio = 0.5;

  BackpropLearningAccelerator_SetToDefault(&self->learning_accelerator);
}




BACKPROP_FLOAT_T BackpropTrainer_GetErrorTolerance(const struct BackpropTrainer* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return self->error_tolerance;
}




void BackpropTrainer_SetErrorTolerance(struct BackpropTrainer* self, BACKPROP_FLOAT_T value)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  self->error_tolerance = value;
}




BACKPROP_FLOAT_T BackpropTrainer_GetLearningRate(const struct BackpropTrainer* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return self->learning_rate;
}




void BackpropTrainer_SetLearningRate(struct BackpropTrainer* self, BACKPROP_FLOAT_T value)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  self->learning_rate = value;
}




BACKPROP_FLOAT_T BackpropTrainer_GetMutationRate(const struct BackpropTrainer* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return self->mutation_rate;
}




void BackpropTrainer_SetMutationRate(struct BackpropTrainer* self, BACKPROP_FLOAT_T value)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  self->mutation_rate = value;
}




BACKPROP_FLOAT_T BackpropTrainer_GetMomentumRate(const struct BackpropTrainer* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return self->momentum_rate;
}




void BackpropTrainer_SetMomentumRate(struct BackpropTrainer* self, BACKPROP_FLOAT_T value)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  self->momentum_rate = value;
}




BACKPROP_SIZE_T BackpropTrainer_GetMaxReps(const struct BackpropTrainer* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return self->max_reps;
}




void BackpropTrainer_SetMaxReps(struct BackpropTrainer* self, BACKPROP_SIZE_T value)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  self->max_reps = value;
}




BACKPROP_SIZE_T BackpropTrainer_GetMaxBatchSets(const struct BackpropTrainer* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return self->max_batch_sets;
}




void BackpropTrainer_SetMaxBatchSets(struct BackpropTrainer* self, BACKPROP_SIZE_T value)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  self->max_batch_sets = value;
}




BACKPROP_SIZE_T BackpropTrainer_GetMaxBatches(const struct BackpropTrainer* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return self->max_batches;
}




void BackpropTrainer_SetMaxBatches(struct BackpropTrainer* self, BACKPROP_SIZE_T value)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  self->max_batches = value;
}




BACKPROP_SIZE_T BackpropTrainer_GetMaxStagnateSets(const struct BackpropTrainer* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return self->max_stagnate_sets;
}




void BackpropTrainer_SetMaxStagnateSets(struct BackpropTrainer* self, BACKPROP_SIZE_T value)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  self->max_stagnate_sets = value;
}




BACKPROP_SIZE_T BackpropTrainer_GetMaxStagnateBatches(const struct BackpropTrainer* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return self->max_stagnate_batches;
}




void BackpropTrainer_SetMaxStagnateBatches(struct BackpropTrainer* self, BACKPROP_SIZE_T value)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  self->max_stagnate_batches = value;
}




BACKPROP_FLOAT_T BackpropTrainer_GetMinSetWeightCorrectionLimit(const struct BackpropTrainer* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return self->min_set_weight_correction_limit;
}




void BackpropTrainer_SetMinSetWeightCorrectionLimit(struct BackpropTrainer* self, BACKPROP_FLOAT_T value)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  self->min_set_weight_correction_limit = value;
}




BACKPROP_FLOAT_T BackpropTrainer_GetMinBatchWeightCorrectionLimit(const struct BackpropTrainer* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return self->min_batch_weight_correction_limit;
}




void BackpropTrainer_SetMinBatchWeightCorrectionLimit(struct BackpropTrainer* self, BACKPROP_FLOAT_T value)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  self->min_batch_weight_correction_limit = value;
}




BACKPROP_FLOAT_T BackpropTrainer_GetTrainingRatio(const struct BackpropTrainer* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return self->training_ratio;
}




void BackpropTrainer_SetTrainingRatio(struct BackpropTrainer* self, BACKPROP_FLOAT_T value)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  self->training_ratio = value;
}




BACKPROP_FLOAT_T BackpropTrainer_GetBatchPruneThreshold(const struct BackpropTrainer* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return self->batch_prune_threshold;
}




void BackpropTrainer_SetBatchPruneThreshold(struct BackpropTrainer* self, BACKPROP_FLOAT_T value)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  self->batch_prune_threshold = value;
}




BACKPROP_FLOAT_T BackpropTrainer_GetStagnateTolerance(const struct BackpropTrainer* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return self->stagnate_tolerance;
}




void BackpropTrainer_SetStagnateTolerance(struct BackpropTrainer* self, BACKPROP_FLOAT_T value)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  self->stagnate_tolerance = value;
}




BACKPROP_FLOAT_T BackpropTrainer_GetBatchPruneRate(const struct BackpropTrainer* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  return self->batch_prune_rate;
}




void BackpropTrainer_SetBatchPruneRate(struct BackpropTrainer* self, BACKPROP_FLOAT_T value)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  self->batch_prune_rate = value;
}




BACKPROP_FLOAT_T BackpropTrainer_ExerciseConst(BackpropTrainer_t* trainer, BackpropExerciseStats_t* stats, struct BackpropNetwork* network, const BackpropConstTrainingSet_t* training_set)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(trainer);
  BACKPROP_ASSERT(stats);
  BACKPROP_ASSERT(network);
  BACKPROP_ASSERT(training_set);
  {
    const long int clock_start = clock();

    BACKPROP_FLOAT_T error = 0;

    const BACKPROP_BYTE_T* x = training_set->x;
    const BACKPROP_BYTE_T* y = training_set->y;

    const size_t count = training_set->dims.count;

    memset(stats, 0, sizeof(BackpropExerciseStats_t));

    for(size_t i = 0; i < count; ++i)
    {
      BackpropNetwork_Input(network, x, training_set->dims.x_size);

      if (trainer->events.AfterInput)
      {
        trainer->events.AfterInput(network);
      }

      BackpropNetwork_Activate(network);

      if (trainer->events.AfterActivate)
      {
        trainer->events.AfterActivate(network);
      }

      error += BackpropTrainer_ComputeError(network, y, training_set->dims.y_size);

      x += training_set->dims.x_size;
      y += training_set->dims.y_size;

      ++(stats->activate_count);
    }

    {
      const long int clock_stop = clock();
      stats->exercise_clock_ticks += (clock_stop - clock_start);
    }

    stats->error += error;
    return error;
  }
}




BACKPROP_FLOAT_T BackpropTrainer_Exercise(BackpropTrainer_t* trainer, BackpropExerciseStats_t* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(trainer);
  BACKPROP_ASSERT(stats);
  BACKPROP_ASSERT(network);
  BACKPROP_ASSERT(training_set);
  {
    BackpropConstTrainingSet_t const_training_set;
    const_training_set.dims = training_set->dims;

    const_training_set.x = training_set->x;
    const_training_set.y = training_set->y;

    return BackpropTrainer_ExerciseConst(trainer, stats, network, &const_training_set);
  }
}




BACKPROP_FLOAT_T BackpropTrainer_TeachPair( BackpropTrainer_t* trainer, BackpropTrainingStats_t* stats
                                          , struct BackpropNetwork* network
                                          , const BACKPROP_BYTE_T* x, BACKPROP_SIZE_T x_size
                                          , const BACKPROP_BYTE_T* y_desired, BACKPROP_SIZE_T y_desired_size)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(trainer);
  BACKPROP_ASSERT(network);
  BACKPROP_ASSERT(x);
  BACKPROP_ASSERT(x_size);
  BACKPROP_ASSERT(y_desired);
  BACKPROP_ASSERT(y_desired_size);
  BACKPROP_ASSERT(network->layers.count > 1);
  {
    BACKPROP_FLOAT_T error = 0;
    BACKPROP_FLOAT_T weight_correction_total = 0;

    if (trainer->events.BeforeTeachPair)
    {
      trainer->events.BeforeTeachPair(trainer, stats, network, x, x_size, y_desired, y_desired_size);
    }

    BackpropNetwork_Input(network, x, x_size);

    if (trainer->events.AfterInput)
    {
      trainer->events.AfterInput(network);
    }

    BackpropNetwork_Activate(network);

    if (trainer->events.AfterActivate)
    {
      trainer->events.AfterActivate(network);
    }

    error = BackpropTrainer_ComputeError(network, y_desired, y_desired_size);

    if (error < trainer->error_tolerance)
    {
      return error;
    }

   {
      // update the output layer
      BackpropLayer_t* layer = BackpropNetwork_GetLastLayer(network);

      BACKPROP_FLOAT_T* W = layer->W;
      BACKPROP_FLOAT_T* g = layer->g;
      BACKPROP_FLOAT_T* y = layer->y;
      const BACKPROP_BYTE_T* yd = y_desired;

    //  BACKPROP_FLOAT_T* W_prev = trainer->W_prev[network->layers.count - 1];

      size_t size = y_desired_size;
      do
      {
        size_t yd_bit = 1;

        size_t b = CHAR_BIT;
        do
        {
          const BACKPROP_FLOAT_T yd_bit_value = 0 < ((*yd) & yd_bit);
          const BACKPROP_FLOAT_T output_error = (yd_bit_value - (*y));
          const BACKPROP_FLOAT_T local_gradient = (*y) * (1 - (*y));
          const BACKPROP_FLOAT_T local_gradient_output_error = local_gradient * output_error;
          const BACKPROP_FLOAT_T correction_strength = (trainer->learning_rate) * local_gradient_output_error;

          *g = local_gradient_output_error;

          // update the layer weights
          for(size_t j = 0; j < layer->x_count; ++j)
          {
            BACKPROP_FLOAT_T momentum = 0.0;
            BACKPROP_FLOAT_T mutation = 0.0;

            if (trainer->momentum_rate)
            {
              // TODO add momentum
              //momentum = trainer->momentum_rate * (*W - W_prev);
            }

            if (trainer->mutation_rate)
            {
              mutation = trainer->mutation_rate * BackpropLayer_RandomWeight();
            }

            {
              const BACKPROP_FLOAT_T correction = correction_strength * (layer->x[j]);
              weight_correction_total += fabs(correction);

              (*W) += correction + momentum + mutation;
            }

    //        *W_prev = *W;  // update for next round

            ++W;
    //        ++W_prev;
          }

          yd_bit <<= 1;
          ++g;
          ++y;

        } while (--b);

        ++yd;

      } while(--size);


      // compute error and update weights
      for(size_t k = network->layers.count - 1; k > 0; --k)  // for each layer in the network
      {
        BackpropLayer_t* pl_next = &network->layers.data[k];
        layer = &network->layers.data[k-1];

        // calculate weighted gradient of next layer
        BackpropLayer_WeightedGradient(pl_next, layer->g);

        // calculate the error
        W = layer->W;
    //    W_prev = trainer->W_prev[k-1];

        for(size_t i = 0; i < layer->y_count; ++i)
        {
          layer->g[i] *= layer->y[i] * (1 - layer->y[i]);  // local gradient

          for(size_t j=0; j < layer->x_count; ++j)
          {
            BACKPROP_FLOAT_T momentum = 0.0;
            BACKPROP_FLOAT_T mutation = 0.0;

            if (trainer->momentum_rate)
            {
              // TODO add momentum
              //momentum = trainer->momentum_rate * (*W - W_prev);
            }

            if (trainer->mutation_rate)
            {
              mutation = trainer->mutation_rate * BackpropLayer_RandomWeight();
            }

            //      learning rate *   gradient    *   signal
            {
              const BACKPROP_FLOAT_T correction = (trainer->learning_rate) * (layer->g[i]) * (layer->x[j]);
              weight_correction_total += fabs(correction);
              (*W) += correction + momentum + mutation;
            }

    //        *W_prev = *W; // update for next round

            ++W;
    //        ++W_prev;
          }
        }
      }
    }
    // re-activate the network and compute the new error
    BackpropNetwork_Activate(network);

    if (trainer->events.AfterActivate)
    {
      trainer->events.AfterActivate(network);
    }

    error = BackpropTrainer_ComputeError(network, y_desired, y_desired_size);

    if (trainer->events.AfterTeachPair)
    {
      trainer->events.AfterTeachPair(trainer, stats, network, x, x_size, y_desired, y_desired_size, network->y.data, network->y.size, error, weight_correction_total);
    }

    stats->batch_weight_correction_total += weight_correction_total;
    stats->set_weight_correction_total += weight_correction_total;

    ++stats->teach_total;

    return error;
  }
}




BACKPROP_FLOAT_T BackpropTrainer_TrainPair( BackpropTrainer_t* trainer
                                          , BackpropTrainingStats_t* stats
                                          , struct BackpropNetwork* network
                                          , const BACKPROP_BYTE_T* x, BACKPROP_SIZE_T x_size
                                          , const BACKPROP_BYTE_T* y_desired, BACKPROP_SIZE_T y_desired_size)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(trainer);
  BACKPROP_ASSERT(stats);
  BACKPROP_ASSERT(network);
  BACKPROP_ASSERT(x);
  BACKPROP_ASSERT(x_size);
  BACKPROP_ASSERT(y_desired);
  BACKPROP_ASSERT(y_desired_size);
  {
    const BACKPROP_FLOAT_T tolerance = trainer->error_tolerance;
    size_t reps = trainer->max_reps;
    BACKPROP_FLOAT_T error = 0;

    if (trainer->events.BeforeTrainPair)
    {
      trainer->events.BeforeTrainPair(trainer, stats, network, x, x_size, y_desired, y_desired_size);
    }

    do
    {
      error = BackpropTrainer_TeachPair(trainer, stats, network, x, x_size, y_desired, y_desired_size);

    } while (--reps && (error > tolerance));


    if (trainer->events.AfterTrainPair)
    {
      trainer->events.AfterTrainPair(trainer, stats, network, x, x_size, y_desired, y_desired_size, network->y.data, network->y.size, error);
    }

    ++stats->pair_total;

    return error;
  }
}




BACKPROP_FLOAT_T BackpropTrainer_TrainSet( BackpropTrainer_t* trainer
                                         , BackpropTrainingStats_t* stats
                                         , struct BackpropNetwork* network
                                         , const BackpropTrainingSet_t* training_set)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(trainer);
  BACKPROP_ASSERT(network);
  BACKPROP_ASSERT(training_set);

  if (0 == training_set->dims.count)
  {
    return 0;
  }

  else
  {
    BACKPROP_FLOAT_T error = 0;

    // determine how many training sets to present
    BACKPROP_SIZE_T training_set_count = (BACKPROP_SIZE_T) (trainer->training_ratio * training_set->dims.count);

    if (training_set_count > training_set->dims.count)
    {
      training_set_count = training_set->dims.count - 1;
    }

    if (training_set_count < 1)
    {
      training_set_count = 1;
    }

    if (trainer->events.BeforeTrainSet)
    {
      trainer->events.BeforeTrainSet(trainer, stats, network, training_set);
    }

    for(size_t i = 0; i < training_set_count; ++i)
    {
      // preset a random training set
      size_t j = Backprop_RandomArrayIndex(0, training_set->dims.count);

      const BACKPROP_BYTE_T* x = training_set->x + j * training_set->dims.x_size;
      const BACKPROP_BYTE_T* y = training_set->y + j * training_set->dims.y_size;

      const BACKPROP_FLOAT_T pair_error = BackpropTrainer_TrainPair(trainer, stats, network, x, training_set->dims.x_size, y, training_set->dims.y_size);

      error += pair_error;
    }

    if (trainer->events.AfterTrainSet)
    {
      trainer->events.AfterTrainSet(trainer, stats, network, training_set, error);
    }

    // update stats
    ++stats->set_total;

    return error;
  }
}




BACKPROP_FLOAT_T BackpropTrainer_TrainBatch( BackpropTrainer_t* trainer
                                           , BackpropTrainingStats_t* stats
                                           , BackpropExerciseStats_t* exercise_stats
                                           , struct BackpropNetwork* network
                                           , const BackpropTrainingSet_t* training_set)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(trainer);
  BACKPROP_ASSERT(network);
  BACKPROP_ASSERT(training_set);
  {
    const BACKPROP_FLOAT_T tolerance = trainer->error_tolerance;
    const BACKPROP_FLOAT_T stagnate_tolerance = trainer->stagnate_tolerance;
    const BACKPROP_FLOAT_T max_stagnate_sets = trainer->max_stagnate_sets;
    const BACKPROP_FLOAT_T max_batch_sets = trainer->max_batch_sets;

    BACKPROP_SIZE_T stagnate_sets = 0;
    BACKPROP_SIZE_T batch_sets = 0;

    BACKPROP_FLOAT_T error = BackpropTrainer_Exercise(trainer, exercise_stats, network, training_set);
    BACKPROP_FLOAT_T last_error = error;

    if (trainer->events.BeforeTrainBatch)
    {
      trainer->events.BeforeTrainBatch(trainer, stats, network, training_set);
    }

    do
    {
      stats->set_weight_correction_total = 0;

      error = BackpropTrainer_TrainSet(trainer, stats, network, training_set);

      trainer->learning_rate = BackpropLearningAccelerator_Accelerate(&trainer->learning_accelerator, trainer->learning_rate, error, last_error);

      if (error <= tolerance)
      {
        error = BackpropTrainer_Exercise(trainer, exercise_stats, network, training_set);
      }

      if (trainer->min_set_weight_correction_limit > stats->set_weight_correction_total)
      {
        if (trainer->events.AfterStubbornSet)
        {
          trainer->events.AfterStubbornSet(trainer, stats, network, training_set, error);
        }

        if ((last_error <= error) || ((last_error - error) < stagnate_tolerance))
        {
          ++stagnate_sets;
          if (trainer->events.AfterStagnateSet)
          {
            trainer->events.AfterStagnateSet(trainer, stats, network, training_set, batch_sets, stagnate_sets, error);
          }
        }
        else
        {
          stagnate_sets = 0;
        }
      }

      if (trainer->events.AfterTrainSet)
      {
        trainer->events.AfterTrainSet(trainer, stats, network, training_set, error);
      }

      last_error = error;
      ++batch_sets;


      if (error <= tolerance)
      {
        break;
      }

      if ((max_batch_sets) && (max_batch_sets <= batch_sets))
      {
        break;
      }

      if (max_stagnate_sets <= stagnate_sets)
      {
        break;
      }

      if (stats->set_weight_correction_total <= trainer->min_set_weight_correction_limit)
      {
        break;
      }

    } while (1);


    if ((stagnate_sets >= max_stagnate_sets) && trainer->events.AfterMaxStagnateSets)
    {
      trainer->events.AfterMaxStagnateSets(trainer, stats, network, training_set, batch_sets, stagnate_sets, error);
    }

    if (trainer->events.AfterTrainBatch)
    {
      trainer->events.AfterTrainBatch(trainer, stats, network, training_set, batch_sets, error);
    }

    ++stats->batches_total;

    return error;
  }
}




BACKPROP_FLOAT_T BackpropTrainer_Train( BackpropTrainer_t* trainer
                                      , BackpropTrainingStats_t* stats
                                      , BackpropExerciseStats_t* exercise_stats
                                      , struct BackpropNetwork* network
                                      , const BackpropTrainingSet_t* training_set)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(trainer);
  BACKPROP_ASSERT(stats);
  BACKPROP_ASSERT(network);
  BACKPROP_ASSERT(training_set);
  {
    const BACKPROP_FLOAT_T stagnate_tolerance = trainer->stagnate_tolerance;
    const BACKPROP_FLOAT_T max_stagnate_batches = trainer->max_stagnate_batches;
    BACKPROP_FLOAT_T stagnate_batches = 0;

    const BACKPROP_SIZE_T max_batch_count = trainer->max_batches;
    BACKPROP_SIZE_T batch_count = 0;

    BACKPROP_FLOAT_T batch_prune_threshold = trainer->batch_prune_rate;

    const BACKPROP_FLOAT_T tolerance = trainer->error_tolerance;
    BACKPROP_FLOAT_T error = BackpropTrainer_Exercise(trainer, exercise_stats, network, training_set);
    BACKPROP_FLOAT_T last_error = error;

    long int clock_start = clock();

    if (error < trainer->error_tolerance)
    {
      return error;
    }

    if (trainer->events.BeforeTrain)
    {
      trainer->events.BeforeTrain(trainer, stats, network, training_set);
    }

    do
    {
      stats->batch_weight_correction_total = 0;

      error = BackpropTrainer_TrainBatch(trainer, stats, exercise_stats, network, training_set);

      if (error > tolerance)
      {
        if (trainer->batch_prune_threshold && trainer->batch_prune_rate)
        {
          batch_prune_threshold = trainer->batch_prune_rate;
        }
      }
      else
      {
        if (trainer->batch_prune_threshold)
        {
          BackpropTrainer_Prune(trainer, network, trainer->batch_prune_threshold);
          batch_prune_threshold += trainer->batch_prune_rate;

          if (batch_prune_threshold > trainer->batch_prune_threshold)
          {
            batch_prune_threshold = trainer->batch_prune_threshold;
          }
        }
      }

      error = BackpropTrainer_Exercise(trainer, exercise_stats, network, training_set);

      if (error > tolerance)
      {
        if (trainer->min_batch_weight_correction_limit > stats->batch_weight_correction_total)
        {
          ++stats->stubborn_batches_total;
          if (trainer->events.AfterStubbornBatch)
          {
            trainer->events.AfterStubbornBatch(trainer, stats, network, training_set, error);
          }

          if ((last_error <= error) || ((last_error - error) < stagnate_tolerance))
          {
            if (trainer->events.AfterStagnateBatch)
            {
              trainer->events.AfterStagnateBatch(trainer, stats, network, training_set, stagnate_batches, error);
            }
            ++stagnate_batches;
            ++stats->stagnate_batches_total;
          }
        }
      }

      last_error = error;

      ++batch_count;

      if (error <= tolerance)
      {
        break;
      }

      if ((max_batch_count) && (max_batch_count <= batch_count))
      {
        break;
      }

      if (trainer->batch_prune_threshold <= batch_prune_threshold)
      {
        break;
      }

      if ((max_stagnate_batches) && (max_stagnate_batches <= stagnate_batches))
      {
        break;
      }

    } while (1);

    if ((stagnate_batches >= max_stagnate_batches) && trainer->events.AfterMaxStagnateBatches)
    {
      trainer->events.AfterMaxStagnateBatches(trainer, stats, network, training_set, stagnate_batches, error);
    }

    if (error > tolerance)
    {
      if (trainer->events.AfterTrainFailure)
      {
        trainer->events.AfterTrainFailure(trainer, stats, network, training_set, error);
      }
    }
    else
    {
      if (trainer->events.AfterTrainSuccess)
      {
        trainer->events.AfterTrainSuccess(trainer, stats, network, training_set, error);
      }
    }

    if (trainer->events.AfterTrain)
    {
      trainer->events.AfterTrain(trainer, stats, network, training_set, error);
    }

    {
      long int clock_stop = clock();
      stats->train_clock = clock_stop - clock_start;
    }

    return error;
  }
}




void BackpropTrainer_Prune(BackpropTrainer_t* trainer, struct BackpropNetwork* network, BACKPROP_FLOAT_T threshold)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(trainer);
  BACKPROP_ASSERT(network);

  BackpropNetwork_Prune(network, trainer->batch_prune_threshold);
}




static struct BackpropNetwork** BackpropNetwork_MallocPool(BACKPROP_SIZE_T x_size, BACKPROP_SIZE_T y_size, BACKPROP_SIZE_T layers_count, BACKPROP_SIZE_T pool_count, bool chain_layers)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(x_size);
  BACKPROP_ASSERT(y_size);
  BACKPROP_ASSERT(layers_count);
  BACKPROP_ASSERT(pool_count);
  {
    struct BackpropNetwork** network_pool = Backprop_Malloc(pool_count * sizeof(struct BackpropNetwork*));

    for (size_t i = 0; i < pool_count; ++i)
    {
      struct BackpropNetwork* new_network = BackpropNetwork_Malloc(x_size, y_size, layers_count, chain_layers);

      if (!new_network)
      {
        break;
      }

      network_pool[i] = new_network;
    }

    return network_pool;
  }
}




static void BackpropNetwork_FreePool(struct BackpropNetwork** network_pool, BACKPROP_SIZE_T pool_count)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(network_pool);
  BACKPROP_ASSERT(pool_count);

  for (size_t i = 0; i < pool_count; ++i)
  {
    BackpropNetwork_Free(network_pool[i]);
  }

  Backprop_Free(network_pool, pool_count * sizeof(struct BackpropNetwork*));
}








/*-------------------------------------------------------------------*
 *
 * BackpropEvolver
 *
 *-------------------------------------------------------------------*/

#pragma mark BackpropEvolver


static void BackpropEvolver_MateLayers(BackpropEvolver_t* evolver, BackpropLayer_t* beta, const BackpropLayer_t* alpha)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(beta);
  BACKPROP_ASSERT(alpha);
  {
    BACKPROP_FLOAT_T* W_b = (beta->W);
    BACKPROP_FLOAT_T* W_a = (alpha->W);

    size_t count = beta->x_count * beta->y_count;

    const BACKPROP_FLOAT_T mate_rate = evolver->mate_rate;
    const BACKPROP_FLOAT_T one_minus_mate_rate = 1.0 - evolver->mate_rate;
    do
    {
      const BACKPROP_FLOAT_T rand_a = BackpropLayer_RandomWeight() * mate_rate;
      const BACKPROP_FLOAT_T rand_b = BackpropLayer_RandomWeight() * one_minus_mate_rate;

      const BACKPROP_FLOAT_T W_new =  (((*W_a) + rand_a) + ((*W_b) + rand_b)) / 2;

      (*W_b) = W_new;

      ++W_b;
      ++W_a;

    } while (--count);
  }
}




static void BackpropEvolver_MateNetworks(BackpropEvolver_t* evolver,  BackpropEvolutionStats_t* evolution_stats, struct BackpropNetwork* beta, const struct BackpropNetwork* alpha)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(beta);
  BACKPROP_ASSERT(alpha);

  ++evolution_stats->mate_networks_count;

  for (size_t i = 0; i < beta->layers.count; ++i)
  {
    if (evolver->BeforeMateLayers)
    {
      evolver->BeforeMateLayers(evolver, evolution_stats, beta, alpha);
    }

    BackpropEvolver_MateLayers(evolver, &beta->layers.data[i], &alpha->layers.data[i]);

    if (evolver->AfterMateLayers)
    {
      evolver->AfterMateLayers(evolver, evolution_stats, beta, alpha);
    }
  }
}




void BackpropEvolver_SetToDefault(BackpropEvolver_t* self)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(self);

  memset(self, 0, sizeof(BackpropEvolver_t));

  self->pool_count = 4;                // 1 alpha + 2 beta + 1 charlie
  self->mate_rate = BACKPROP_MIN_GOLD;
  self->max_generations = 4; //32;
  self->mutation_limit = 1.0;
  self->seed = 0;
  self->random_gain = 4.0;
}




/** Use an evolutionary algorithm to evolve a network trained for the given training set.
 */
BACKPROP_FLOAT_T BackpropEvolver_Evolve(BackpropEvolver_t* evolver, BackpropEvolutionStats_t* evolution_stats, BackpropTrainer_t* trainer, BackpropTrainingStats_t* training_stats, BackpropExerciseStats_t* exercise_stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set)
{
  BACKPROP_TRACE();

  BACKPROP_ASSERT(evolver);
  BACKPROP_ASSERT(evolution_stats);
  BACKPROP_ASSERT(trainer);
  BACKPROP_ASSERT(training_stats);
  BACKPROP_ASSERT(exercise_stats);
  BACKPROP_ASSERT(network);
  BACKPROP_ASSERT(training_set);
  {
    const bool chain_layers = true;

    long int clock_start = clock();

    // allocate network pool
    struct BackpropNetwork** network_pool = BackpropNetwork_MallocPool( network->x.size
                                                                      , network->y.size
                                                                      , network->layers.count
                                                                      , evolver->pool_count
                                                                      , chain_layers);

    // copy existing network data into pool
    BackpropNetwork_DeepCopy(network, network_pool[0]);

    // randomize rest of pool
    {
      unsigned int seed = evolver->seed;
      for (size_t i = 1; i < evolver->pool_count; ++i)
      {
        BackpropNetwork_Randomize(network_pool[i], evolver->random_gain, seed);
        ++seed;
      }
    }

    {
      // get error benchmarks
      BACKPROP_FLOAT_T error = BackpropTrainer_Exercise(trainer, exercise_stats, network_pool[0], training_set);

      BACKPROP_FLOAT_T best_error = error;
      struct BackpropNetwork* best = network_pool[0];

      BACKPROP_FLOAT_T worst_error = error;
      struct BackpropNetwork* worst = network_pool[0];

      // clear out stats
      *evolution_stats = (BackpropEvolutionStats_t) {0};

      // batch train the network pool
      {
        BACKPROP_SIZE_T generation_count = 0;
        while ((error > trainer->error_tolerance) && (generation_count < evolver->max_generations))
        {
          if (evolver->BeforeGeneration)
          {
            evolver->BeforeGeneration(evolver, evolution_stats, generation_count);
          }

          // train all pool members
          for (size_t i = 1; i < evolver->pool_count; ++i)
          {
            error = BackpropTrainer_TrainBatch(trainer, training_stats, exercise_stats, network_pool[i], training_set);

            if (error < best_error)
            {
              best_error = error;
              best = network_pool[i];
            }

            if (error < trainer->error_tolerance)
            {
              break;
            }

            if (error > worst_error)
            {
              worst_error = error;
              worst = network_pool[i];
            }
          }

          if (error < trainer->error_tolerance)
          {
            break;
          }

          // evolve pool members
          for (size_t i = 0; i < evolver->pool_count; ++i)
          {
            struct BackpropNetwork* network = network_pool[i];

            // do not mate best with self or worst member of pool
            if ((network == best) || (network == worst))
            {
              continue;
            }

            if (evolver->BeforeMateNetworks)
            {
              evolver->BeforeMateNetworks(evolver, evolution_stats, network);
            }

            BackpropEvolver_MateNetworks(evolver, evolution_stats, network, best);

            if (evolver->AfterMateNetworks)
            {
              evolver->AfterMateNetworks(evolver, evolution_stats, network, best);
            }
          }

          if (evolver->AfterGeneration)
          {
            evolver->AfterGeneration(evolver, evolution_stats, generation_count);
          }

          ++generation_count;
          ++evolution_stats->generation_count;
        }
      }

      // copy out best network data
      BackpropNetwork_DeepCopy(best, network);

      // all done
      BackpropNetwork_FreePool(network_pool, evolver->pool_count);

      {
        long int clock_stop = clock();
        evolution_stats->evolve_clock = clock_stop - clock_start;
      }

      return best_error;
    }
  }
}
