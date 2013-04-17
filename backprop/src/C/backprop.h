/** backprop.h

Defines interface for backpropagation network (backprop)
The network can be any size, but must be intialized before using any of the functions.

For most applications of the backprop, there will
be two programs: a training program and the application program.
The application program will only need to load the weight data,
input data to the network and retrieve the output.  For a training program,
the weights must be randomized, then trained with an appropriate training set and the weights saved to an output file.

When using creating a network with backprop.h, the first layer is implied.  Thus,
for a three 'layer' network, only two layers need to be defined.  In the case of
a three layer network, the hidden layer input serves also as the input layer.

The learning method used by the backprop is a gradient descent without
momentum.  For stable training used a learning rate close to 0 (e.g. 0.1 or 0.2),
for faster training use a learning rate close to 1 (e.g. 0.9 or 0.8).
Use caution when increasing the learning rate however because too large a rate
can cause the network never to converge on a set of weights.

An evolutionary algorithm can also be used in conjuction with gradient descent learning.
In some cases, the evolutionary algorithm far outperforms gradient descent alone.



 Limitations

 The convergence obtained from backpropagation learning is very slow.
 The convergence in backpropagation learning is not guaranteed.
 The result may generally converge to any local minimum on the error surface, since stochastic gradient descent exists on a surface which is not flat.
 Backpropagation learning requires input scaling or normalization.


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

#ifndef BACKPROP_H
#define BACKPROP_H


#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>




#ifndef NDEBUG
  #include <assert.h>
  #define BACKPROP_ASSERT(_arg_)    assert(_arg_)
#else
  #define BACKPROP_ASSERT(_arg_)
#endif




/* Backprop library size definitions.
 * Adjust this for the desired target architecture.
 */

#define BACKPROP_SIZE_T    size_t
//#define BACKPROP_FLOAT_T    float
#define BACKPROP_FLOAT_T    double
#define BACKPROP_BYTE_T    uint8_t



/** A simple size, pointer pair.
 */
typedef struct BackpropByteArray
{
  size_t size;
  BACKPROP_BYTE_T* data;

} BackpropByteArray_t;




typedef struct BackpropConstByteArray
{
  size_t size;
  const BACKPROP_BYTE_T* data;

} BackpropConstByteArray_t;




/** A simple size, pointer pair.
 */
typedef struct BackpropFloatArray
{
  size_t size;
  BACKPROP_FLOAT_T* data;

} BackpropFloatArray_t;




/**
 * Backprop Layer structure.
 *
 * y = sig(W * x)
 *
 */
typedef struct BackpropLayer
{
	BACKPROP_SIZE_T x_count; ///< Number of inputs to each neuron (M).
	BACKPROP_SIZE_T y_count; ///< Number of neurons in the layer (N).

	BACKPROP_FLOAT_T* W; ///< Pointer to weight matrix  [NxM].
	BACKPROP_FLOAT_T* g; ///< Pointer to layer gradient [Nx1].

	BACKPROP_FLOAT_T* x; ///< Pointer to layer input    [Mx1].
	BACKPROP_FLOAT_T* y; ///< Pointer to layer output   [Nx1].

} BackpropLayer_t;




/**
 * Structure to hold an array of BackpropLayer.
 */
typedef struct BackpropLayersArray
{
  BACKPROP_SIZE_T count;  ///< Number of layers.
  struct BackpropLayer* data;  ///< Pointer to layer array.

} BackpropLayersArray_t;












/*-------------------------------------------------------------------*
 *
 * GLOBAL CONFIGURATION FUNCTIONS
 *
 * These settings may affect all other library functions.
 * You should configure these once before calling other library functions.
 *
 * By default the library uses malloc() and free().
 * Calls to malloc() and free() are counted, so the amount of dynamic
 * memory used can be determined by calling Backprop_GetMallocInUse();
 *
 *-------------------------------------------------------------------*/


/** Set callback function to malloc().
 *  Can be used to override default behavior.
 */
void Backprop_SetMalloc(void* (*) (size_t));


/** Set callback function to free().
 *  Can be used to override default behavior.
 */
void Backprop_SetFree(void (*) (void*));


/** Called if Backprop_Malloc() returns a NULL pointer.
 */
void Backprop_SetOnMallocFail(void (*) (size_t));


/** Returns the total number of bytes allocated by malloc.
 *  Will wrap eventually.
 */
size_t Backprop_GetMallocTotal(void);


/** Returns the total number of bytes allocated by malloc.
 *  Will wrap eventually.
 */
size_t Backprop_GetFreeTotal(void);


/** Reset the total counters.
 */
void Backprop_ResetTotals(void);


/** Returns the number of bytes currently in use.
 */
size_t Backprop_GetMallocInUse(void);








/*-------------------------------------------------------------------*
 *
 * MATH FUNCTIONS
 *
 *-------------------------------------------------------------------*/


/** Evaluate sigmoid function.
 *  1.0 / (1.0 + exp(-x))
 *  Returns a number in the range (0, 1) and where 0.5 == Backprop_Sigmoid(0).
 */
BACKPROP_FLOAT_T Backprop_Sigmoid(BACKPROP_FLOAT_T x);








/*-------------------------------------------------------------------*
 *
 * RANDOM FUNCTIONS
 *
 *-------------------------------------------------------------------*/


/** Use pseudo-random number generator (PRNG) to generate a random integer.
 */
int Backprop_UniformRandomInt(void);


/** Seed the PRNG.
 */
void Backprop_RandomSeed(unsigned long seed);


/** Use the current time to seed the PRNG.
 */
void Backprop_RandomSeedTime(void);


/** Use pseudo-random number generator (PRNG) to generate a random float.
 */
BACKPROP_FLOAT_T Backprop_UniformRandomFloat(void);


/** Generate a uniform random number in the range of (-1, 1).
 */
BACKPROP_FLOAT_T BackpropLayer_RandomWeight(void);


/** Generate an index between the values of [lower, upper).
 */
BACKPROP_SIZE_T Backprop_RandomArrayIndex(size_t lower, size_t upper);








/*-------------------------------------------------------------------*
 *
 * NETWORK MEMORY MANAGEMENT FUNCTIONS
 *
 *-------------------------------------------------------------------*/


/** Returns the number of bytes allocated for a network with the given sizes.
 */
size_t BackpropNetwork_MallocSize(BACKPROP_SIZE_T x_size, BACKPROP_SIZE_T y_size, BACKPROP_SIZE_T layers_count);


/** Dynamically allocate memory for a BackpropNetwork.
 *  Returns pointer to malloc'ed memory.
 *  Returns NULL if error.
 *  Must call BackpropNetwork_Free() with pointer returned from this function.
 */
struct BackpropNetwork* BackpropNetwork_Malloc(BACKPROP_SIZE_T x_size, BACKPROP_SIZE_T y_size, BACKPROP_SIZE_T layers_count, bool chain_layers);


/** Free memory allocated by BackpropNetwork_Malloc().
 */
void BackpropNetwork_Free(struct BackpropNetwork* network);








/*-------------------------------------------------------------------*
 *
 * NETWORK MANIPULATION FUNCTIONS
 *
 * These are the core functions for using the backpropagation network.
 *
 *-------------------------------------------------------------------*/


/** Returns 1 if the network is valid, 0 otherwise.
 */
int BackpropNetwork_IsValid(const struct BackpropNetwork* self);


/** Input byte values to a network.
 */
void BackpropNetwork_Input(struct BackpropNetwork* self, const BACKPROP_BYTE_T* values, BACKPROP_SIZE_T values_count);


/** Input C string byte values to a network.
 */
void BackpropNetwork_InputCStr(struct BackpropNetwork* self, const char* values);


/** Activate a network.
 *  Will calculate output based on current input.
 */
void BackpropNetwork_Activate(struct BackpropNetwork* self);


/** Output bytes values from a network.
 *  Returns number of input bytes used.
 */
BACKPROP_SIZE_T BackpropNetwork_GetOutput(const struct BackpropNetwork* self, BACKPROP_BYTE_T* values, BACKPROP_SIZE_T values_count);


/** Output byte values from a network as a NULL terminated string.
 *  Writes up to str_size bytes into str.
 *  Returns number of bytes written.
 */
size_t BackpropNetwork_GetOutputCStr(const struct BackpropNetwork* self, char* str, size_t str_size);


/** Get pointer to first layer in the network.  For 1 layer networks, this is also the last layer.
 */
BackpropLayer_t* BackpropNetwork_GetFirstLayer(struct BackpropNetwork* self);


/** Get pointer to last layer in the network.  For 1 layer networks, this is also the first layer.
 */
BackpropLayer_t* BackpropNetwork_GetLastLayer(struct BackpropNetwork* self);


/** Get const pointer to last layer in the network.  For 1 layer networks, this is also the first layer.
 */
const BackpropLayer_t* BackpropNetwork_GetConstLastLayer(const struct BackpropNetwork* self);


/** Randomize weights for a network.
 */
void BackpropNetwork_Randomize(struct BackpropNetwork* self, unsigned int seed);


/** Round weights to nearest whole numbers.
 */
void BackpropNetwork_Round(struct BackpropNetwork* self);


/** Set network input and output bytes to 0.
 *  Set input and output values to 0 for all layers in the network.
 *  Does not affect network layer weights.
 */
void BackpropNetwork_Reset(struct BackpropNetwork* self);


/** Set weight values that are less than the given threshold to 0.0.
 */
void BackpropNetwork_Prune(struct BackpropNetwork* self, BACKPROP_FLOAT_T threshold);


BACKPROP_SIZE_T BackpropNetwork_GetWeightsCount(const struct BackpropNetwork* self);


BACKPROP_SIZE_T BackpropNetwork_GetWeightsSize(const struct BackpropNetwork* self);


BACKPROP_FLOAT_T BackpropNetwork_GetWeightsSum(const struct BackpropNetwork* self);


BACKPROP_FLOAT_T BackpropNetwork_GetWeightsMean(const struct BackpropNetwork* self);


BACKPROP_FLOAT_T BackpropNetwork_GetWeightsStdDev(const struct BackpropNetwork* self);


/** Get read-only pointer the network input.
 */
const BackpropByteArray_t* BackpropNetwork_GetX(const struct BackpropNetwork* self);


/** Get the size of input.
 */
BACKPROP_SIZE_T BackpropNetwork_GetXSize(const struct BackpropNetwork* self);


/** Get read-only pointer the network input.
 */
const BackpropByteArray_t* BackpropNetwork_GetY(const struct BackpropNetwork* self);


/** Get the size of output.
 */
BACKPROP_SIZE_T BackpropNetwork_GetYSize(const struct BackpropNetwork* self);


/** Get read-only pointer to the network layers.
 */
const struct BackpropLayersArray* BackpropNetwork_GetLayers(const struct BackpropNetwork* self);


/** Get the amount of jitter that is applied to the input for the network is activated.
 */
BACKPROP_FLOAT_T BackpropNetwork_GetJitter(const struct BackpropNetwork* self);


/** Set the amount of jitter that is applied to the input before the network is activated.
 */
void BackpropNetwork_SetJitter(struct BackpropNetwork* self, BACKPROP_FLOAT_T jitter);








/*-------------------------------------------------------------------*
 *
 * NETWORK STATISTICS
 *
 * Functions for gathering statistics about a BackpropNetwork.
 *
 *-------------------------------------------------------------------*/


/** Structure that holds statistics about a BackpropNetwork.
 */
typedef struct BackpropNetworkStats
{
  BACKPROP_SIZE_T x_size;              ///< Size of the input.
  BACKPROP_SIZE_T y_size;              ///< Size of the output.
  BACKPROP_SIZE_T layers_count;        ///< Number of network layers.
  BACKPROP_SIZE_T layers_size;         ///< Size in bytes of network layers.
  BACKPROP_SIZE_T layers_W_count;      ///< Number of all layer weights in the network.
  BACKPROP_SIZE_T layers_W_size;       ///< Size in bytes of all layer weights in the network.

  BACKPROP_FLOAT_T layers_W_avg;       ///< Mean value of all layer weights in the network.
  BACKPROP_FLOAT_T layers_W_stddev;    ///< Standard deviation of all layer weights in the network.

} BackpropNetworkStats_t;




/** Get statistics about a network.
 *  Copies data into structure pointed to by stats.
 */
void BackpropNetwork_GetStats(const struct BackpropNetwork* self, BackpropNetworkStats_t* stats);








/*-------------------------------------------------------------------*
 *
 * NETWORK TRAINING FUNCTIONS
 *
 *-------------------------------------------------------------------*/


#define BEGIN_TRAINING_SET(_name_, _count_, _in_size_, _out_size_) \
static const size_t _name_##_TRAINING_SET_COUNT = (_count_); \
static const size_t _name_##_TRAINING_SET_IN_SIZE = (_in_size_); \
static const size_t _name_##_TRAINING_SET_OUT_SIZE = (_out_size_);


#define DEFINE_TRAINING_SET_X(_name_) \
static const BACKPROP_BYTE_T _name_##_TRAINING_SET_X[_name_##_TRAINING_SET_COUNT][_name_##_TRAINING_SET_IN_SIZE] =


#define DEFINE_TRAINING_SET_Y(_name_) \
static const BACKPROP_BYTE_T _name_##_TRAINING_SET_Y[_name_##_TRAINING_SET_COUNT][_name_##_TRAINING_SET_OUT_SIZE] =


#define END_TRAINING_SET(_name_) \
static const BackpropConstTrainingSet_t _name_##_TRAINING_SET = \
{ \
  .dims = { \
    .count = _name_##_TRAINING_SET_COUNT, \
    .x_size = _name_##_TRAINING_SET_IN_SIZE, \
    .y_size = _name_##_TRAINING_SET_OUT_SIZE \
  }, \
  .x = &_name_##_TRAINING_SET_X[0][0], \
  .y = &_name_##_TRAINING_SET_Y[0][0] \
};




/** Statistics from a call to BackpropTrainer_Exercise()
 */
typedef struct BackpropExerciseStats
{
  long int exercise_clock_ticks;
  BACKPROP_SIZE_T activate_count;
  BACKPROP_FLOAT_T error;

} BackpropExerciseStats_t;




/** Structure to hold dimension information for a BackpropTrainingSet
 */
typedef struct BackpropTrainingSetDimensions
{
  BACKPROP_SIZE_T  count;  ///< Number of x:y pairs in the training set
  BACKPROP_SIZE_T  x_size;
  BACKPROP_SIZE_T  y_size;

} BackpropTrainingSetDimensions_t;




/** Training set structure.
 * Holds pointers to x:y training pairs.
 */
typedef struct BackpropTrainingSet
{
  BackpropTrainingSetDimensions_t dims;

  BACKPROP_BYTE_T* x;
  BACKPROP_BYTE_T* y;

} BackpropTrainingSet_t;





/** Allocate a BackpropTrainingSet with given dimensions from the heap.
 */
BackpropTrainingSet_t* BackpropTrainingSet_Malloc(size_t count, size_t x_size, size_t y_size);


/** Free a BackpropTrainingSet allocated with BackpropTrainingSet_Malloc.
 */
void BackpropTrainingSet_Free(BackpropTrainingSet_t* self);


/** Get the input size in bytes.
 */
size_t BackpropTrainingSet_GetXSize(BackpropTrainingSet_t* self);


/** Get the output size in bytes.
 */
size_t BackpropTrainingSet_GetYSize(BackpropTrainingSet_t* self);


/** Set training pair data.
 */
void BackpropTrainingSet_SetPair(BackpropTrainingSet_t* self, BACKPROP_BYTE_T* x, BACKPROP_BYTE_T* y);


/** Set training pair data as a NULL terminated strings.
 */
void BackpropTrainingSet_SetPairCStr(BackpropTrainingSet_t* self, const char* x, const char* y);


typedef struct BackpropConstTrainingSet
{
  BackpropTrainingSetDimensions_t dims;

  const BACKPROP_BYTE_T* x;
  const BACKPROP_BYTE_T* y;

} BackpropConstTrainingSet_t;





/** Statistics from a call to BackpropTrainer_Train()
 */
typedef struct BackpropTrainingStats
{
  BACKPROP_FLOAT_T set_weight_correction_total;
  BACKPROP_FLOAT_T batch_weight_correction_total;
  BACKPROP_SIZE_T teach_total;                      ///< Total teaching.
  BACKPROP_SIZE_T pair_total;                       ///< Total number of training pairs.
  BACKPROP_SIZE_T set_total;                        ///< Total number of training sets.
  BACKPROP_SIZE_T batches_total;                    ///< Total number of training batches.

  BACKPROP_SIZE_T stubborn_batches_total;           ///< Total number of stubborn batches encountered during training.
  BACKPROP_SIZE_T stagnate_batches_total;           ///< Total number of stagnate batches encountered during training.

  long int train_clock;  ///< Clock ticks used in training.

} BackpropTrainingStats_t;




struct BackpropTrainer;


typedef struct BackpropTrainerEvents
{
  /* Callback pointers */
  void (*AfterInput)(const struct BackpropNetwork*); ///< Called after trainer calls BackpropNetwork_Input.
  void (*AfterActivate)(const struct BackpropNetwork*); ///< Called after trainer calls BackpropNetwork_Activate.
  void (*AfterExercisePair)(const struct BackpropNetwork*, BACKPROP_FLOAT_T error); ///< Called after trainer exercises a training set x:y pair.
  void (*AfterExercise)(const struct BackpropNetwork*, BACKPROP_FLOAT_T error); ///< Called after trainer exercises a training set x:y pair.

  void (*BeforeTrain)(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set);
  void (*AfterTrainSuccess)(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_FLOAT_T error);
  void (*AfterTrainFailure)(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_FLOAT_T error);
  void (*AfterTrain)(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_FLOAT_T error);


  void (*BeforeTrainBatch)(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set);
  void (*AfterTrainBatch)(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_SIZE_T batches, BACKPROP_FLOAT_T error);

  void (*AfterStagnateSet)(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_SIZE_T batches, BACKPROP_SIZE_T stagnate_sets, BACKPROP_FLOAT_T error);

  void (*AfterMaxStagnateSets)(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_SIZE_T batches, BACKPROP_SIZE_T stagnate_sets, BACKPROP_FLOAT_T error);

  void (*AfterStubbornSet)(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_FLOAT_T error);

  void (*AfterStagnateBatch)(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_SIZE_T stagnate_batches, BACKPROP_FLOAT_T error);
  void (*AfterMaxStagnateBatches)(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_SIZE_T stagnate_batches, BACKPROP_FLOAT_T error);

  void (*AfterStubbornBatch)(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_FLOAT_T error);


  void (*BeforeTrainSet)(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set);
  void (*AfterTrainSet)(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_FLOAT_T error);

  void (*BeforeTrainPair)(const struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, const struct BackpropNetwork* network, const BACKPROP_BYTE_T* x, const BACKPROP_SIZE_T x_size, const BACKPROP_BYTE_T* yd, const BACKPROP_SIZE_T yd_size);

  void (*AfterTrainPair)(const struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, const struct BackpropNetwork* network, const BACKPROP_BYTE_T* x, const BACKPROP_SIZE_T x_size, const BACKPROP_BYTE_T* yd, const BACKPROP_SIZE_T yd_size, const BACKPROP_BYTE_T* y, const BACKPROP_SIZE_T y_size, BACKPROP_FLOAT_T error);

  void (*BeforeTeachPair)(const struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, const struct BackpropNetwork* network, const BACKPROP_BYTE_T* x, const BACKPROP_SIZE_T x_size, const BACKPROP_BYTE_T* yd, const BACKPROP_SIZE_T yd_size);

  void (*AfterTeachPair)(const struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, const struct BackpropNetwork* network, const BACKPROP_BYTE_T* x, const BACKPROP_SIZE_T x_size, const BACKPROP_BYTE_T* yd, const BACKPROP_SIZE_T yd_size, const BACKPROP_BYTE_T* y, const BACKPROP_SIZE_T y_size, BACKPROP_FLOAT_T error, BACKPROP_FLOAT_T weight_correction);


} BackpropTrainerEvents_t;




/** Structure for holding information about BackpropTrainer learning_rate accelerator algorithm.
 */
typedef struct BackpropLearningAccelerator
{
  BACKPROP_FLOAT_T min_learning_rate;
  BACKPROP_FLOAT_T max_learning_rate;
  BACKPROP_FLOAT_T acceleration;

} BackpropLearningAccelerator_t;




/** Set the BackpropLearningAccelerator attributes to default values.
 */
void BackpropLearningAccelerator_SetToDefault(BackpropLearningAccelerator_t* self);


/** Returns the learning rate calcuated by a BackpropLearningAccelerator.
 */
BACKPROP_FLOAT_T BackpropLearningAccelerator_Accelerate(BackpropLearningAccelerator_t* self, BACKPROP_FLOAT_T learning_rate, BACKPROP_FLOAT_T error_now, BACKPROP_FLOAT_T error_prev);




/** Returns the number of bytes allocated for a trainer for a given network.
 */
size_t BackpropTrainer_MallocSize(const struct BackpropNetwork* network);


/** Dynamically allocate memory for a BackpropTrainer.
 *  Returns pointer to malloc'ed memory.
 *  Returns NULL if error.
 *  Must call BackpropTrainer_Free() with pointer returned from this function.
 */
struct BackpropTrainer* BackpropTrainer_Malloc(struct BackpropNetwork* network);


/** Free memory allocated by BackpropNetwork_Malloc().
 */
void BackpropTrainer_Free(struct BackpropTrainer* network);


/** Set the default values for a BackpropTrainer.
 */
void BackpropTrainer_SetToDefault(struct BackpropTrainer* self);


/** Exercise a network with a given training set and return the total error for the training set.
 */
BACKPROP_FLOAT_T BackpropTrainer_Exercise(struct BackpropTrainer* self, BackpropExerciseStats_t* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set);


/** Exercise a network with a given training set and return the total error for the training set.
 */
BACKPROP_FLOAT_T BackpropTrainer_ExerciseConst(struct BackpropTrainer* self, BackpropExerciseStats_t* stats, struct BackpropNetwork* network, const BackpropConstTrainingSet_t* training_set);



/** Use Backpropagation learning to train network for a given x:y pair.
 */
BACKPROP_FLOAT_T BackpropTrainer_TrainPair( struct BackpropTrainer* self, BackpropTrainingStats_t* stats
                                          , struct BackpropNetwork* network
                                          , const BACKPROP_BYTE_T* x, size_t x_size
                                          , const BACKPROP_BYTE_T* y, size_t y_size);


/** Use Backpropagation learning to train network for the given training set.
 */
BACKPROP_FLOAT_T BackpropTrainer_Train( struct BackpropTrainer* trainer
                                      , BackpropTrainingStats_t* stats
                                      , BackpropExerciseStats_t* exercise_stats
                                      , struct BackpropNetwork* network
                                      , const BackpropTrainingSet_t* training_set);


/** Use the BackpropTrainer to prune the network.
 */
void BackpropTrainer_Prune(struct BackpropTrainer* trainer, struct BackpropNetwork* network, BACKPROP_FLOAT_T threshold);


/** Structure to hold statistics about a BackpropNetwork evolution.
 */
typedef struct BackpropEvolutionStats
{
  BACKPROP_SIZE_T generation_count;
  BACKPROP_SIZE_T mate_networks_count;

  long int evolve_clock;

} BackpropEvolutionStats_t;




/** Backprop Evolver structure.
 *  Holds parameters that affect network evolution.
 */
typedef struct BackpropEvolver
{
  BACKPROP_SIZE_T pool_count;      ///< Number of networks in the network pool.
  BACKPROP_SIZE_T max_generations; ///< Maximum number of generations to run.
  BACKPROP_FLOAT_T mate_rate;      ///< Proportion of alpha weight to beta weight.  0.5 is equal alpha and beta weights.  0.75 alpha is (0.75 * alpha) + (0.25 * beta).
  BACKPROP_FLOAT_T mutation_limit; ///< The maximum mutation in a single neuron weight.
  unsigned int seed;               ///< Seed used for random number generator.


  void (*BeforeMateNetworks)(const struct BackpropEvolver*, const BackpropEvolutionStats_t* stats, const struct BackpropNetwork* network);
  void (*AfterMateNetworks)(const struct BackpropEvolver*, const BackpropEvolutionStats_t* stats, const struct BackpropNetwork* network, const struct BackpropNetwork* best);

  void (*BeforeMateLayers)(const struct BackpropEvolver*, const BackpropEvolutionStats_t* stats, const struct BackpropNetwork* beta, const struct BackpropNetwork* alpha);
  void (*AfterMateLayers)(const struct BackpropEvolver*, const BackpropEvolutionStats_t* stats, const struct BackpropNetwork* beta, const struct BackpropNetwork* alpha);

  void (*BeforeGeneration)(const struct BackpropEvolver*, const BackpropEvolutionStats_t* stats, BACKPROP_SIZE_T generation);
  void (*AfterGeneration)(const struct BackpropEvolver*, const BackpropEvolutionStats_t* stats, BACKPROP_SIZE_T generation);


} BackpropEvolver_t;




/** Set the default values for a BackpropEvolver.
 */
void BackpropEvolver_SetToDefault(BackpropEvolver_t* self);



/** Use an evolutionary algorithm to evolve a network trained for the given training set.
 */
BACKPROP_FLOAT_T BackpropEvolver_Evolve( BackpropEvolver_t* evolver
                                       , BackpropEvolutionStats_t* evolution_stats
                                       , struct BackpropTrainer* trainer
                                       , BackpropTrainingStats_t* training_stats
                                       , BackpropExerciseStats_t* exercise_stats
                                       , struct BackpropNetwork* network
                                       , const BackpropTrainingSet_t* training_set);




#endif //BACKPROP_H
