/** backprop_io.c


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




#include "backprop_io.h"
#include "backprop.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <ctype.h>




#ifndef NDEBUG
  #include <assert.h>
  #define BACKPROP_IO_ASSERT(_arg_)    assert(_arg_)

#else
  #define BACKPROP_IO_ASSERT(_arg_)

#endif




/*-------------------------------------------------------------------*
 *
 * FILE I/O FUNCTIONS
 *
 *-------------------------------------------------------------------*/

#pragma mark file_io




static int fstrcmp(FILE* file, const char* str)
{
  BACKPROP_IO_ASSERT(file);
  BACKPROP_IO_ASSERT(str);
  {
    size_t c_count = 0;
    char c = 0;
    do
    {
      if (!str)
      {
        break;
      }

      c = fgetc(file);
      ++c_count;

      if (c > *str)
      {
        fseek(file, -c_count, SEEK_CUR); // put back contents
        return 1;
      }

      else if ((c < *str))
      {
        fseek(file, -c_count, SEEK_CUR); // put back contents
        return -1;
      }

      ++str;

    } while ((*str != 0) && (c != 0) && (c != EOF));

    return 0;
  }
}




static size_t fskipcomma(FILE* file)
{
  BACKPROP_IO_ASSERT(file);
  {
    char c;
    size_t c_count = 0;

    c = fgetc(file);
    ++c_count;

    if (c != ',')
    {
      // rewind 1 for non-space character
      fseek(file, -1, SEEK_CUR);
      --c_count;
    }

    return c_count;
  }
}




static size_t fskipspace(FILE* file)
{
  BACKPROP_IO_ASSERT(file);
  {
    char c;
    size_t c_count = 0;

    do
    {
      c = fgetc(file);
      ++c_count;

    } while (isspace(c));

    // rewind 1 for non-space character
    fseek(file, -1, SEEK_CUR);
    --c_count;

    return c_count;
  }
}



//static size_t fskipifspace(FILE* file)
//{
//  BACKPROP_IO_ASSERT(file);
//
//  fskipspace(file);
//
//  return 1;
//}



static size_t fskipstr(FILE* file, const char* str)
{
  BACKPROP_IO_ASSERT(file);
  BACKPROP_IO_ASSERT(str);
  {
    size_t c_count = 0;

    while (*str != '\0')
    {
      if (!str)
      {
        return c_count;
      }
      else
      {
        const char c = fgetc(file);
        ++c_count;

        if (c != *str)
        {
          fseek(file, -c_count, SEEK_CUR);
          return 0;
        }

        ++str;
      }
    };

    return c_count;
  }
}








/*-------------------------------------------------------------------*
 *
 * JSON FUNCTIONS
 *
 *-------------------------------------------------------------------*/

#pragma mark json

static size_t json_fscanpair_str_size(FILE* file, const char* str, BACKPROP_SIZE_T* size)
{
  BACKPROP_IO_ASSERT(file);
  BACKPROP_IO_ASSERT(str);
  BACKPROP_IO_ASSERT(size);
  {
    size_t c_count = 0;

    c_count += fskipspace(file);

    if (fstrcmp(file, str) != 0)
    {
      return 0;
    }

    c_count += fskipspace(file);

    {
      char c = fgetc(file);
      ++c_count;

      if (c != ':')
      {
        fseek(file, -c_count, SEEK_CUR);
        return 0;
      }
    }

    c_count += fskipspace(file);

    {
      int fscanf_result = fscanf(file, "%lu", size);

      if (!fscanf_result)
      {
        fseek(file, -c_count, SEEK_CUR);
      }

      c_count += fscanf_result;
    }

    c_count += fskipspace(file);

    return c_count;
  }
}




static size_t json_fskipcomma(FILE* file)
{
  size_t c_count = fskipspace(file);
  c_count += fskipcomma(file);

  if (c_count)
  {
    c_count += fskipspace(file);
  }

  return c_count;
}




static size_t json_fskipstr(FILE* file, const char* str)
{
  size_t c_count = fskipspace(file);
  c_count += fskipstr(file, str);

  if (c_count)
  {
    c_count += fskipspace(file);
  }

  return c_count;
}




static size_t json_fscanarray_byte(FILE* file, BACKPROP_BYTE_T* dest, size_t dest_count)
{
  BACKPROP_IO_ASSERT(file);
  BACKPROP_IO_ASSERT(dest);
  BACKPROP_IO_ASSERT(dest_count);
  {
    size_t file_count = 0;
    do
    {
      file_count += fskipspace(file);

      {
        int i;
        int fscanf_result = fscanf(file, "%X", &i);

        if (!fscanf_result)
        {
          break;
        }

        file_count += fscanf_result;
        *dest = i & 0xFF;
      }

      file_count += json_fskipcomma(file);

      ++dest;
    } while (--dest_count);

    return file_count;
  }
}




static size_t json_fscanarray_float(FILE* file, BACKPROP_FLOAT_T* dest, size_t dest_count)
{
  BACKPROP_IO_ASSERT(file);
  BACKPROP_IO_ASSERT(dest);
  BACKPROP_IO_ASSERT(dest_count);
  {
    size_t c_count = 0;
    do
    {
      c_count += fskipspace(file);

      {
        double d = 0;
        int fscanf_result = fscanf(file, "%lf", &d);
        *dest = d;

        if (!fscanf_result)
        {
          printf("bad scan\n");
          break;
        }

        c_count += fscanf_result;
      }
      c_count += json_fskipcomma(file);

      ++dest;
    } while (--dest_count);

    return c_count;
  }
}




static size_t json_fprintarray_byte(FILE* file, const BACKPROP_BYTE_T* array, size_t size)
{
  BACKPROP_IO_ASSERT(file);
  {
    if (!array || !size)
    {
      return 0;
    }

    {
      size_t file_count = 0;

      file_count += fprintf(file, "[");

      fprintf(file, "0x%02X", *array);
      ++array;
      --size;

      if (size)
      {
        do
        {
          fprintf(file, ", 0x%02X", *array);
          ++array;
        } while (--size);
      }

      file_count += fprintf(file, "]");

      return file_count;
    }
  }
}




//static size_t BackpropConstByteArray_Fprintf(BackpropConstByteArray_t* self, FILE* file)
//{
//  BACKPROP_IO_ASSERT(self);
//  BACKPROP_IO_ASSERT(file);
//  {
//    size_t file_count = 0;
//
//    file_count += fprintf(file, "{size: %lu, data:\n", self->size);
//    file_count += json_fprintarray_byte(file, self->data, self->size);
//    file_count += fprintf(file, "}");
//
//    return file_count;
//  }
//}





/*-------------------------------------------------------------------*
 *
 * Backprop
 *
 *-------------------------------------------------------------------*/

#pragma mark Backprop



void Backprop_PutsOnMallocFail(size_t size)
{
  printf("{ backprop_malloc_fail: %lu }\n", size);
}








/*-------------------------------------------------------------------*
 *
 * BackpropLayer
 *
 *-------------------------------------------------------------------*/

#pragma mark BackpropLayer


void BackpropLayer_PrintfInput(const struct BackpropLayer* self)
{
  BACKPROP_IO_ASSERT(self);

  printf("[");

  printf("%f", BackpropLayer_GetX(self, 0));

  const size_t x_count = BackpropLayer_GetXCount(self);
  for (size_t i = 1; i < x_count; ++i)
  {
    printf(", %f", BackpropLayer_GetX(self, i));
  }

  puts("]");
}




void BackpropLayer_PrintfOutput(const struct BackpropLayer* self)
{
  BACKPROP_IO_ASSERT(self);

  printf("[");

  printf("%f", BackpropLayer_GetY(self, 0));

  const size_t y_count = BackpropLayer_GetYCount(self);
  for (size_t i = 1; i < y_count; ++i)
  {
    printf(", %f", BackpropLayer_GetY(self, i));
  }

  puts("]");
}




size_t BackpropLayer_FprintfWeights(FILE* file, const struct BackpropLayer* self)
{
  BACKPROP_IO_ASSERT(file);
  BACKPROP_IO_ASSERT(self);
  {
    size_t fprintf_size = fprintf(file, "[ ");

    const size_t x_count = BackpropLayer_GetXCount(self);
    size_t newline_count = CHAR_BIT;
    size_t W_row_count = x_count * CHAR_BIT;
    size_t count = x_count * BackpropLayer_GetYCount(self);

    const BACKPROP_FLOAT_T* W = BackpropLayer_GetConstW(self);

    if (*W >= 0)
    {
      fprintf_size += fprintf(file, " %f", *W);
    }
    else
    {
      fprintf_size += fprintf(file, "%f", *W);
    }

    ++W;
    --count;
    --newline_count;
    --W_row_count;

    do
    {
      if (*W >= 0)
      {
        fprintf_size += fprintf(file, ",  %f", *W);
      }
      else
      {
        fprintf_size += fprintf(file, ", %f", *W);
      }
      ++W;

      --newline_count;
      if(!newline_count)
      {
        newline_count = 8;
        fprintf_size += fprintf(file, "\n");
      }

      --W_row_count;
      if (!W_row_count)
      {
        W_row_count = x_count * CHAR_BIT;
        fprintf_size += fprintf(file, "\n");
      }

    } while (--count);

    fprintf_size += fprintf(file, "]");

    return fprintf_size;
  }
}




void BackpropLayer_PrintfWeights(const struct BackpropLayer* self)
{
  BACKPROP_IO_ASSERT(self);

  BackpropLayer_FprintfWeights(stdout, self);
}




void BackpropLayer_PutsWeights(const struct BackpropLayer* self)
{
  BACKPROP_IO_ASSERT(self);

  BackpropLayer_PrintfWeights(self);
  printf("\n");
}




//static size_t BackpropLayer_SaveWeights(BackpropLayer_t* self, FILE *fp)
//{
//  BACKPROP_IO_ASSERT(self);
//  BACKPROP_IO_ASSERT(fp);
//  {
//    const size_t count = (self->x_count)*(self->y_count);
//
//    return fwrite(self->W, sizeof(float) * count , 1, fp);
//  }
//}




//
//static void SaveLayerWeightsHDT(BackpropLayer_t* l, FILE *fp)
//{
//  BACKPROP_FLOAT_T* W = l->W;
//  fprintf(fp, "{\n");
//
//  for(size_t i = 0; i < l->x_count; ++i)
//  {
//    fprintf(fp, "{");
//
//    for(size_t j = 0; j < l->y_count; ++j)
//    {
//      fprintf(fp, "%f", *W);
//      ++W;
//
//      if(j < l->x_count - 1)
//      {
//        fprintf(fp,", ");
//      }
//    }
//
//    if(i < l->y_count - 1)
//    {
//      fprintf(fp,"},\n");
//    }
//    else
//    {
//      fprintf(fp,"}\n");
//    }
//  }
//  fprintf(fp,"}");
//}




static size_t BackpropLayer_LoadWeights(struct BackpropLayer* self, FILE *file)
{
  BACKPROP_IO_ASSERT(self);
  BACKPROP_IO_ASSERT(file);
  {
    BACKPROP_FLOAT_T* W = BackpropLayer_GetW(self);
    BACKPROP_IO_ASSERT(W);
    {
      size_t c_count = fskipspace(file);
      c_count += fskipstr(file, "[");
      c_count += fskipspace(file);

      if (c_count)
      {
        const BACKPROP_SIZE_T x_count = BackpropLayer_GetXCount(self);
        const BACKPROP_SIZE_T y_count = BackpropLayer_GetYCount(self);

        c_count += json_fscanarray_float(file, W, x_count * y_count);
        c_count += fskipstr(file, "]");
        c_count += fskipspace(file);
      }

      return c_count;
    }
  }
}








/*-------------------------------------------------------------------*
 *
 * BackpropNetwork
 *
 *-------------------------------------------------------------------*/

#pragma mark BackpropNetwork




void BackpropNetwork_Printf(const struct BackpropNetwork* self)
{
  BACKPROP_IO_ASSERT(self);

  BackpropNetwork_PrintfInput(self);
  BackpropNetwork_PrintfOutput(self);
}




void BackpropNetwork_PrintfInput(const struct BackpropNetwork* self)
{
  BACKPROP_IO_ASSERT(self);

  printf("[");

  {
    size_t i = 0;

    printf("0x%02X", BackpropNetwork_GetX(self)->data[i]);

    while (++i < BackpropNetwork_GetX(self)->size)
    {
      printf(", 0x%02X", BackpropNetwork_GetX(self)->data[i]);
    }
  }

  printf("]");
}




void BackpropNetwork_PutsInput(const struct BackpropNetwork* self)
{
  BACKPROP_IO_ASSERT(self);

  BackpropNetwork_PrintfInput(self);
  printf("\n");
}




void BackpropNetwork_PrintfLayersInput(const struct BackpropNetwork* self)
{
  BACKPROP_IO_ASSERT(self);

  printf("[");

  {
    size_t i = 0;

    const struct BackpropLayersArray* layers = BackpropNetwork_GetLayers(self);

    const size_t x_size = BackpropNetwork_GetX(self)->size;

    const struct BackpropLayer* layer = BackpropLayersArray_GetConstLayer(layers, 0);
    printf("%f", BackpropLayer_GetAtX(layer, i));

    while (++i < x_size)
    {
      printf(", %f", BackpropLayer_GetAtX(layer, i));
    }
  }

  printf("]");
}




void BackpropNetwork_PutsLayersInput(const struct BackpropNetwork* self)
{
  BACKPROP_IO_ASSERT(self);

  BackpropNetwork_PrintfLayersInput(self);
  printf("\n");
}




void BackpropNetwork_PrintfLayersOutput(const struct BackpropNetwork* self)
{
  BACKPROP_IO_ASSERT(self);
  {
    const struct BackpropLayersArray* layers = BackpropNetwork_GetLayers(self);
    const struct BackpropLayer* last_layer = BackpropNetwork_GetConstLastLayer(self);

    size_t last_index = BackpropLayersArray_GetCount(layers);

    if (last_index)
    {
      --last_index;
    }

    printf("[");

    {
      BACKPROP_SIZE_T y_size = BackpropNetwork_GetYSize(self);
      const size_t size = y_size * CHAR_BIT;
      size_t i = 0;

      printf("%f", BackpropLayer_GetAtY(last_layer, i));

      while (++i < size)
      {
        printf(", %f", BackpropLayer_GetAtY(last_layer, i));
      }
    }

    printf("]");
  }
}




void BackpropNetwork_PutsLayersOutput(const struct BackpropNetwork* self)
{
  BACKPROP_IO_ASSERT(self);

  BackpropNetwork_PrintfLayersOutput(self);
  printf("\n");
}




void BackpropNetwork_PrintfOutput(const struct BackpropNetwork* self)
{
  BACKPROP_IO_ASSERT(self);

  printf("[");

  {
    const BackpropByteArray_t* y = BackpropNetwork_GetY(self);
    BACKPROP_SIZE_T y_size = BackpropNetwork_GetYSize(self);
    if (y_size)
    {
      printf("%02X", y->data[0]);
      ++y;

      for(size_t i = 1; i < y_size; ++i)
      {
        printf(", %02X", y->data[i]);
        ++y;
      }
    }
  }

  printf("]");
}




void BackpropNetwork_PutsOutput(const struct BackpropNetwork* self)
{
  BACKPROP_IO_ASSERT(self);

  BackpropNetwork_PrintfOutput(self);
  printf("\n");
}




void BackpropNetwork_PrintfInputOutput(const struct BackpropNetwork* self)
{
  BACKPROP_IO_ASSERT(self);

  printf("{ x: ");
  BackpropNetwork_PrintfInput(self);
  printf(", y: ");
  BackpropNetwork_PrintfOutput(self);
  printf(" }");
}




void BackpropNetwork_PutsInputOutput(const struct BackpropNetwork* self)
{
  BACKPROP_IO_ASSERT(self);

  BackpropNetwork_PrintfInputOutput(self);
  printf("\n");
}




size_t BackpropNetwork_FprintfWeights(FILE* file, const struct BackpropNetwork* self)
{
  BACKPROP_IO_ASSERT(self);
  {
    BACKPROP_SIZE_T x_size = BackpropNetwork_GetXSize(self);
    BACKPROP_SIZE_T y_size = BackpropNetwork_GetYSize(self);
    const struct BackpropLayersArray* layers = BackpropNetwork_GetLayers(self);

    size_t fprintf_size = fprintf(file, "{network_weights: {x_size: %ld, y_size: %ld, layers_count: %ld, layers: ", x_size, y_size, layers->count);

    fprintf_size += fprintf(file, "[\n");

    fprintf_size += BackpropLayer_FprintfWeights(file, BackpropLayersArray_GetConstLayer(layers, 0));

    for(size_t i = 1; i < layers->count; ++i)
    {
      fprintf_size += fprintf(file, ",\n");
      fprintf_size += BackpropLayer_FprintfWeights(file, BackpropLayersArray_GetConstLayer(layers, i));
    }
    fprintf_size += fprintf(file, "]\n");

    fprintf_size += fprintf(file, "}}");

    return fprintf_size;
  }
}




void BackpropNetwork_PrintfWeights(const struct BackpropNetwork* self)
{
  BackpropNetwork_FprintfWeights(stdout, self);
}




void BackpropNetwork_PutsWeights(const struct BackpropNetwork* self)
{
  BackpropNetwork_PrintfWeights(self);
  puts("\n");
}




size_t BackpropNetwork_SaveWeights(const struct BackpropNetwork* self, const char* filename)
{
  BACKPROP_IO_ASSERT(self);
  BACKPROP_IO_ASSERT(filename);
  {
    FILE *file = fopen(filename, "w");

    if(NULL == file)
    {
      printf("Cannot open %s\n",filename);
      return 0;
    }
    else
    {
      const size_t write_size = BackpropNetwork_FprintfWeights(file, self);
      fclose(file);
      return write_size;
    }
  }
}





size_t BackpropNetwork_LoadWeights(struct BackpropNetwork* self, const char* filename)
{
  BACKPROP_IO_ASSERT(self);
  {
    size_t c_count = 0;

    BACKPROP_SIZE_T x_size = 0;
    BACKPROP_SIZE_T y_size = 0;
    BACKPROP_SIZE_T layers_count = 0;

    BACKPROP_SIZE_T network_x_size = BackpropNetwork_GetXSize(self);
    BACKPROP_SIZE_T network_y_size = BackpropNetwork_GetYSize(self);
    const struct BackpropLayersArray* network_layers = BackpropNetwork_GetLayers(self);

    FILE *file = NULL;

    if((file = fopen(filename,"rb")) == NULL)
    {
      printf("Cannot open %s\n",filename);
      return 0;
    }

    c_count += fskipstr(file, "{network_weights: {");

    if (!c_count)
    {
      return c_count;
    }

    c_count += json_fscanpair_str_size(file, "x_size", &x_size);
    c_count += json_fskipcomma(file);
    c_count += json_fscanpair_str_size(file, "y_size", &y_size);
    c_count += json_fskipcomma(file);
    c_count += json_fscanpair_str_size(file, "layers_count", &layers_count);
    c_count += json_fskipcomma(file);
    c_count += fskipstr(file, "layers:");
    c_count += fskipspace(file);

    if (c_count)
    {
      if (  (network_x_size != x_size)
          ||(network_y_size != y_size)
          ||(network_layers->count != layers_count))
      {
        return c_count;
      }

      c_count += fskipstr(file, "[");

      for(size_t i = 0; i < layers_count; ++i)
      {
        c_count += BackpropLayer_LoadWeights(BackpropLayersArray_GetConstLayer(network_layers, i), file);
        c_count += json_fskipcomma(file);
      }

      c_count += fskipstr(file, "]");
    }
    c_count += fskipstr(file, "}}");

    fclose(file);

    return c_count;
  }
}








/*-------------------------------------------------------------------*
 *
 * BackpropNetworkStats
 *
 *-------------------------------------------------------------------*/

#pragma mark BackpropNetworkStats




size_t BackpropNetworkStats_Fprintf(const BackpropNetworkStats_t* self, FILE* file)
{
  BACKPROP_IO_ASSERT(self);

  return fprintf( file
                , "network_stats: "
                  "{ x_size: %lu"
                  ", y_size: %lu"
                  ", layers_count: %lu"
                  ", layers_size: %lu"
                  ", layers_W_count: %lu"
                  ", layers_W_size: %lu"
                  ", layers_W_avg: %f"
                  ", layers_W_stddef: %f"
                  " }"
                , self->x_size
                , self->y_size
                , self->layers_count
                , self->layers_size
                , self->layers_W_count
                , self->layers_W_size
                , self->layers_W_avg
                , self->layers_W_stddev);
}




size_t BackpropNetworkStats_Printf(const BackpropNetworkStats_t* self)
{
  BACKPROP_IO_ASSERT(self);

  return BackpropNetworkStats_Fprintf(self, stdout);
}




size_t BackpropNetworkStats_Puts(const BackpropNetworkStats_t* self)
{
  BACKPROP_IO_ASSERT(self);
  {
    size_t result = BackpropNetworkStats_Printf(self);
    result += printf("\n");

    return result;
  }
}




/*-------------------------------------------------------------------*
 *
 * NETWORK TRAINING SET
 *
 *-------------------------------------------------------------------*/

#pragma mark BackpropTrainingSetDimensions


static size_t BackpropTrainingSetDimensions_Fprintf(const BackpropTrainingSetDimensions_t* self, FILE* file)
{
  BACKPROP_IO_ASSERT(self);
  BACKPROP_IO_ASSERT(file);
  {
    size_t file_count = 0;

    file_count += fprintf( file
                         , "dimensions: {count: %lu, x_size: %lu, y_size: %lu}"
                         , self->count, self->x_size, self->y_size);

    return file_count;
  }
}




static size_t BackpropTrainingSetDimensions_Fparsef(BackpropTrainingSetDimensions_t* self, FILE* file)
{
  BACKPROP_IO_ASSERT(self);
  BACKPROP_IO_ASSERT(file);
  {
    size_t file_count = 0;
    size_t count = 0;
    size_t x_size = 0;
    size_t y_size = 0;

    file_count += json_fskipstr(file, "dimensions:");
    file_count += json_fskipstr(file, "{");

    file_count += json_fscanpair_str_size(file, "count", &count);
    file_count += json_fskipcomma(file);
    file_count += json_fscanpair_str_size(file, "x_size", &x_size);
    file_count += json_fskipcomma(file);
    file_count += json_fscanpair_str_size(file, "y_size", &y_size);

    file_count += json_fskipstr(file, "}");

    self->count = count;
    self->x_size = x_size;
    self->y_size = y_size;

    return file_count;
  }
}




/*-------------------------------------------------------------------*
 *
 * NETWORK TRAINING SET
 *
 *-------------------------------------------------------------------*/

#pragma mark BackpropTrainingSet


size_t BackpropTrainingSet_Fprintf(const BackpropTrainingSet_t* self, FILE* file)
{
  BACKPROP_IO_ASSERT(self);
  BACKPROP_IO_ASSERT(file);
  {
    size_t file_count = 0;

    const size_t pair_count = self->dims.count;
    const size_t x_size = self->dims.x_size;
    const size_t y_size = self->dims.y_size;
    const BACKPROP_BYTE_T* x = self->x;
    const BACKPROP_BYTE_T* y = self->y;

    file_count += fprintf(file, "training_set: {\n");

    file_count += BackpropTrainingSetDimensions_Fprintf(&self->dims, file);
    file_count += fprintf(file, ", ");
    file_count += fprintf(file, "\n");

    file_count += fprintf(file, "x:\n[ ");
    file_count += json_fprintarray_byte(file, x, x_size);
    file_count += fprintf(file, "\n");
    x += x_size;

    for (size_t i = 1; i < pair_count; ++i)
    {
      file_count += fprintf(file, ", ");
      file_count += json_fprintarray_byte(file, x, x_size);
      file_count += fprintf(file, "\n");
      x += x_size;
    }
    file_count += fprintf(file, "],\n");

    file_count += fprintf(file, "y:\n[ ");

    file_count += json_fprintarray_byte(file, y, y_size);
    file_count += fprintf(file, "\n");
    y += y_size;

    for (size_t i = 1; i < pair_count; ++i)
    {
      file_count += fprintf(file, ", ");
      file_count += json_fprintarray_byte(file, y, y_size);
      file_count += fprintf(file, "\n");
      y += y_size;
    }
    file_count += fprintf(file, "]");
    file_count += fprintf(file, "\n");

    file_count += fprintf(file, "}");
    return file_count;
  }
}




size_t BackpropTrainingSet_Printf(const BackpropTrainingSet_t* self)
{
  BACKPROP_IO_ASSERT(self);

  return BackpropTrainingSet_Fprintf(self, stdout);
}



size_t BackpropTrainingSet_Puts(const BackpropTrainingSet_t* self)
{
  BACKPROP_IO_ASSERT(self);
  {
    size_t size = BackpropTrainingSet_Printf(self);
    size += printf("\n");

    return size;
  }
}




size_t BackpropTrainingSet_Fparsef(BackpropTrainingSet_t* self, FILE* file)
{
  BACKPROP_IO_ASSERT(self);
  BACKPROP_IO_ASSERT(file);
  {
    size_t file_count = 0;
    BACKPROP_BYTE_T* x = self->x;

    BackpropTrainingSetDimensions_t dims = {0};

    file_count += fskipstr(file, "training_set: {");
    file_count += BackpropTrainingSetDimensions_Fparsef(&dims, file);

    if (   (dims.count != self->dims.count)
        || (dims.x_size != self->dims.x_size)
        || (dims.y_size != self->dims.y_size))
    {
      fseek(file, -file_count, SEEK_CUR);
      return 0;
    }

    file_count += json_fskipcomma(file);
    file_count += json_fskipstr(file, "x:");
    file_count += json_fskipstr(file, "[");

    for (size_t i = 0; i < dims.count; ++i)
    {
      file_count += json_fskipstr(file, "[");
      file_count += json_fscanarray_byte(file, x, dims.x_size);
      file_count += json_fskipstr(file, "]");
      file_count += json_fskipcomma(file);

      x += dims.x_size;
    }

    file_count += json_fskipstr(file, "]");

    {
      BACKPROP_BYTE_T* y = self->y;

      file_count += json_fskipcomma(file);
      file_count += json_fskipstr(file, "y:");
      file_count += json_fskipstr(file, "[");

      for (size_t i = 0; i < dims.count; ++i)
      {
        file_count += json_fskipstr(file, "[");
        file_count += json_fscanarray_byte(file, y, dims.y_size);
        file_count += json_fskipstr(file, "]");
        file_count += json_fskipcomma(file);

        y += dims.y_size;
      }
    }

    file_count += json_fskipstr(file, "]");

    return file_count;
  }
}




size_t BackpropTrainingSet_LoadDimensions(BackpropTrainingSetDimensions_t* dims, const char* filename)
{
  BACKPROP_IO_ASSERT(dims);
  BACKPROP_IO_ASSERT(filename);
  {
    size_t file_count = 0;

    FILE* file = fopen(filename,"r+");

    if(file == NULL)
    {
      return 0;
    }
    else
    {
      file_count += fskipstr(file, "training_set: {");
      file_count += BackpropTrainingSetDimensions_Fparsef(dims, file);

      fclose(file);
    }

    return file_count;
  }
}




size_t BackpropTrainingSet_Load(BackpropTrainingSet_t* self, const char* filename)
{
  BACKPROP_IO_ASSERT(self);
  {
    size_t count = 0;

    FILE* file = fopen(filename,"r+");

    if(file == NULL)
    {
      return 0;
    }
    else
    {
      count += BackpropTrainingSet_Fparsef(self, file);

      fclose(file);
    }

    return count;
  }
}




size_t BackpropTrainingSet_Save(const BackpropTrainingSet_t* self, const char* filename)
{
  BACKPROP_IO_ASSERT(self);
  {
    size_t count = 0;

    FILE* file = fopen(filename,"w");

    if(file == NULL)
    {
      return 0;
    }
    else
    {
      count += BackpropTrainingSet_Fprintf(self, file);

      fclose(file);
    }

    return count;
  }
}








/*-------------------------------------------------------------------*
 *
 * BackpropTrainer
 *
 *-------------------------------------------------------------------*/

#pragma mark BackpropTrainer

void BackpropTrainer_PrintfAfterInput( const struct BackpropTrainer* trainer
                                     , const struct BackpropNetwork* network
                                     , const BACKPROP_BYTE_T* x
                                     , BACKPROP_SIZE_T x_size)
{
 // TODO
}


void BackpropTrainer_PrintfAfterTeachPair( const struct BackpropTrainer* trainer
                                         , const struct BackpropTrainingStats* stats
                                         , const struct BackpropNetwork* network
                                         , const BACKPROP_BYTE_T* x, const BACKPROP_SIZE_T x_size
                                         , const BACKPROP_BYTE_T* yd, const BACKPROP_SIZE_T yd_size
                                         , const BACKPROP_BYTE_T* y, const BACKPROP_SIZE_T y_size
                                         , BACKPROP_FLOAT_T error, BACKPROP_FLOAT_T weight_correction)
{
  BACKPROP_IO_ASSERT(trainer);
  BACKPROP_IO_ASSERT(stats);
  BACKPROP_IO_ASSERT(network);

  printf("{ taught_pair: { error: %f, weight_correction: %f }}", error, weight_correction);

}




void BackpropTrainer_PutsAfterTeachPair( const struct BackpropTrainer* trainer
                                       , const struct BackpropTrainingStats* stats
                                       , const struct BackpropNetwork* network
                                       , const BACKPROP_BYTE_T* x, const BACKPROP_SIZE_T x_size
                                       , const BACKPROP_BYTE_T* yd, const BACKPROP_SIZE_T yd_size
                                       , const BACKPROP_BYTE_T* y, const BACKPROP_SIZE_T y_size
                                       , BACKPROP_FLOAT_T error, BACKPROP_FLOAT_T weight_correction)
{
  BackpropTrainer_PrintfAfterTeachPair(trainer, stats, network, x, x_size, yd, yd_size, y, y_size, error, weight_correction);
  printf("\n");

}


void BackpropTrainer_FprintfAfterTrainSet(  FILE* file
                                          , struct BackpropTrainer* trainer
                                          , const struct BackpropTrainingStats* stats
                                          , struct BackpropNetwork* network
                                          , const BackpropTrainingSet_t* training_set
                                          , BACKPROP_FLOAT_T error)
{
  fprintf(file, "trained_set: { error: %f, weight_correction: %f }", error, stats->set_weight_correction_total);
}


void BackpropTrainer_PrintfAfterTrainSet(  struct BackpropTrainer* trainer
                                         , const struct BackpropTrainingStats* stats
                                         , struct BackpropNetwork* network
                                         , const BackpropTrainingSet_t* training_set
                                         , BACKPROP_FLOAT_T error)
{
  BackpropTrainer_FprintfAfterTrainSet(stdout, trainer, stats, network, training_set, error);
}


void BackpropTrainer_PutsAfterTrainSet(  struct BackpropTrainer* trainer
                                       , const struct BackpropTrainingStats* stats
                                       , struct BackpropNetwork* network
                                       , const BackpropTrainingSet_t* training_set
                                       , BACKPROP_FLOAT_T error)
{
  BackpropTrainer_PrintfAfterTrainSet(trainer, stats, network, training_set, error);
  printf("\n");
}



void BackpropTrainer_AfterTrainBatch(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_SIZE_T batches, BACKPROP_FLOAT_T error)
{
  printf("trained_batch: {batch: %ld, error: %f }", batches, error);
}




void BackpropTrainer_PrintfAfterStagnateSet( struct BackpropTrainer* trainer
                                           , const struct BackpropTrainingStats* stats
                                           , struct BackpropNetwork* network
                                           , const BackpropTrainingSet_t* training_set
                                           , BACKPROP_SIZE_T batches
                                           , BACKPROP_SIZE_T stagnate_sets
                                           , BACKPROP_FLOAT_T error)
{
  printf("stagnate_set: {set: %ld, batch: %ld, error: %f }", stagnate_sets, batches, error);
}




void BackpropTrainer_PutsAfterStagnateSet(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_SIZE_T batches, BACKPROP_SIZE_T stagnate_sets, BACKPROP_FLOAT_T error)
{
  BackpropTrainer_PrintfAfterStagnateSet(trainer, stats, network, training_set, batches, stagnate_sets, error);
  printf("\n");

}




void BackpropTrainer_PrintfAfterStagnateBatch( struct BackpropTrainer* trainer
                                             , const struct BackpropTrainingStats* stats
                                             , struct BackpropNetwork* network
                                             , const BackpropTrainingSet_t* training_set
                                             , BACKPROP_SIZE_T batches
                                             , BACKPROP_FLOAT_T error)
{
  printf("stagnate_batch: { batch: %ld, error: %f }", batches, error);
}




void BackpropTrainer_PutsAfterStagnateBatch(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_SIZE_T batches, BACKPROP_FLOAT_T error)
{
  BackpropTrainer_PutsAfterStagnateBatch(trainer, stats, network, training_set, batches, error);
  printf("\n");
}




void BackpropTrainer_PrintfAfterTrainSuccess(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_FLOAT_T error)
{
  printf("train: { success: true, error: %f}", error);
}




void BackpropTrainer_PutsAfterTrainSuccess(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_FLOAT_T error)
{
  BackpropTrainer_PrintfAfterTrainSuccess(trainer, stats, network, training_set, error);
  printf("\n");
}




void BackpropTrainer_PrintfAfterTrainFailure( struct BackpropTrainer* trainer
                                            , const struct BackpropTrainingStats* stats
                                            , struct BackpropNetwork* network
                                            , const BackpropTrainingSet_t* training_set
                                            , BACKPROP_FLOAT_T error)
{
  printf("train: { success: false, error: %f }", error);
}




void BackpropTrainer_PutsAfterTrainFailure(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_FLOAT_T error)
{
  BackpropTrainer_PrintfAfterTrainFailure(trainer, stats, network, training_set, error);
  printf("\n");
}




void BackpropTrainer_SetToDefaultIO(struct BackpropTrainer* trainer)
{
  BACKPROP_IO_ASSERT(trainer);
  {
    struct BackpropTrainerEvents* events = BackpropTrainer_GetEvents(trainer);

    events->AfterInput = NULL;
    events->AfterActivate = NULL;
    events->AfterExercisePair = NULL;
    events->AfterExercise = NULL;

    events->BeforeTrain = NULL;
    events->AfterTrainSuccess = NULL;
    events->AfterTrainFailure = NULL;
    events->AfterTrain = NULL;

    events->BeforeTrainBatch = NULL;
    events->AfterTrainBatch = NULL;

    events->AfterStagnateSet = NULL;
    events->AfterMaxStagnateSets = NULL;
    events->AfterStubbornSet = NULL;

    events->AfterStagnateBatch = NULL;
    events->AfterMaxStagnateBatches = NULL;
    events->AfterStubbornBatch = NULL;

    events->BeforeTrainSet = NULL;
    events->AfterTrainSet = NULL;

    events->BeforeTrainPair = NULL;
    events->AfterTrainPair = NULL;

    events->BeforeTeachPair = NULL;
    events->AfterTeachPair = NULL;
  }
}




void BackpropTrainer_SetToVerboseIO(struct BackpropTrainer* trainer)
{
  BACKPROP_IO_ASSERT(trainer);
  {
    BackpropTrainer_SetToDefaultIO(trainer);

    struct BackpropTrainerEvents* events = BackpropTrainer_GetEvents(trainer);

    events->AfterInput = BackpropTrainer_PrintfAfterInput;
    //events->AfterActivate = BackpropTainer_PrintfAfterActivate;
    //events->AfterExercisePair = BackpropTrainer_PrintfAfterExercisePair;
    //events->AfterExercise = BackpropTrainer_PrintfAfterExercise;
    //
    //events->BeforeTrain = BackpropTrainer_PrintfBeforeTrain;
    //events->AfterTrainSuccess = BackpropTrainer_PrinfAfterTrainSuccess;
    //events->AfterTrainFailure = BackpropTrainer_PrintfAfterTrainFailure;
    //events->AfterTrain = BackpropTrainer_PrintfAfterTrain;
    //
    //events->BeforeTrainBatch = BackpropTrainer_PrintfBeforeTrainBatch;
    //events->AfterTrainBatch = BackpropTrainer_PrintfAfterTrainBatch;
    //
    //events->AfterStagnateSet = BackpropTrainer_PrintfAfterStagnateSet;
    //events->AfterMaxStagnateSets = BackpropTrainer_PrintfAfterMaxStagnateSets;
    //events->AfterStubbornSet = BackpropTrainer_PrintfAfterStubbornSet;
    //
    //events->AfterStagnateBatch = BackpropTrainer_PrintfAfterStagnateBatch;
    //events->AfterMaxStagnateBatches = BackpropTrainer_PrintfAfterMaxStagnateBatches;
    //events->AfterStubbornBatch = BackpropTrainer_PrintfAfterStubbornBatch;
    //
    //events->BeforeTrainSet = BackpropTrainer_PrintfBeforeTrainSet;
    //events->AfterTrainSet = BackpropTrainer_PrintfAfterTrainSet;
    //
    //events->BeforeTrainPair = BackpropTrainer_PrintfBeforeTrainPair;
    //events->AfterTrainPair = BackpropTrainer_PrintfAfterTrainPair;
    //
    //events->BeforeTeachPair = BackpropTrainer_PrintfBeforeTeachPair;
    //events->AfterTeachPair = BackpropTrainer_PrintfAfterTeachPair;
  }
}






/*-------------------------------------------------------------------*
 *
 * BackpropExerciseStats
 *
 *-------------------------------------------------------------------*/

#pragma mark BackpropExerciseStats


size_t BackpropExerciseStats_Fprintf(const struct BackpropExerciseStats* stats, FILE* file)
{
  BACKPROP_IO_ASSERT(file);
  BACKPROP_IO_ASSERT(stats);

  return fprintf(file,
         "exercise_stats: "
         "{ error: %f"
         ", exercise_clock_ticks: %ld"
         ", activate_count: %ld"
         " }"
         , stats->error
         , stats->exercise_clock_ticks
         , stats->activate_count);
}




size_t BackpropExerciseStats_Printf(const struct BackpropExerciseStats* stats)
{
  return BackpropExerciseStats_Fprintf(stats, stdout);
}




size_t BackpropExerciseStats_Puts(const struct BackpropExerciseStats* stats)
{
  size_t result = BackpropExerciseStats_Printf(stats);
  result += printf("\n");
  return result;
}








/*-------------------------------------------------------------------*
 *
 * BackpropTrainingStats
 *
 *-------------------------------------------------------------------*/

#pragma mark BackpropTrainingStats


size_t BackpropTrainingStats_Fprintf(const struct BackpropTrainingStats* stats, FILE* file)
{
  BACKPROP_IO_ASSERT(stats);

  return fprintf( file
                , "training_stats: "
                  "{ set_weight_correction_total: %f"
                  ", batch_weight_correction_total: %f"
                  ", pair_total: %lu"
                  ", set_total: %lu"
                  ", batches_total: %lu"
                  ", train_clock: %ld"
                  " }"
                , stats->set_weight_correction_total
                , stats->batch_weight_correction_total
                , stats->pair_total
                , stats->set_total
                , stats->batches_total
                , stats->train_clock);
}




size_t BackpropTrainingStats_Printf(const struct BackpropTrainingStats* stats)
{
  return BackpropTrainingStats_Fprintf(stats, stdout);
}




size_t BackpropTrainingStats_Puts(const struct BackpropTrainingStats* stats)
{
  size_t result = BackpropTrainingStats_Printf(stats);
  result += printf("\n");
  return result;
}








/*-------------------------------------------------------------------*
 *
 * BackpropEvolutionStats
 *
 *-------------------------------------------------------------------*/

#pragma mark BackpropEvolutionStats


size_t BackpropEvolutionStats_Fprintf(const struct BackpropEvolutionStats* stats, FILE* file)
{
  BACKPROP_IO_ASSERT(stats);
  BACKPROP_IO_ASSERT(file);

  return fprintf( file
                , "evolution_stats: "
                  "{ generation_count: %lu"
                  ", mate_networks_count: %lu"
                  ", evolve_clock: %ld}"
                  " }"
                , stats->generation_count
                , stats->mate_networks_count
                , stats->evolve_clock);
}




size_t BackpropEvolutionStats_Printf(const struct BackpropEvolutionStats* stats)
{
  BACKPROP_IO_ASSERT(stats);

  return BackpropEvolutionStats_Fprintf(stats, stdout);
}




size_t BackpropEvolutionStats_Puts(const struct BackpropEvolutionStats* stats)
{
  BACKPROP_IO_ASSERT(stats);
  {
    size_t result = BackpropEvolutionStats_Printf(stats);
    result += printf("\n");
    return result;
  }
}
