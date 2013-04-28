/** backprop_io.h

Defines I/O functions for backprop.h.


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


#ifndef BACKPROP_IO_H
#define BACKPROP_IO_H


#include <stdio.h>
#include "backprop.h"




/*-------------------------------------------------------------------*
 *
 * Backprop
 *
 *-------------------------------------------------------------------*/


void Backprop_PutsOnMallocFail(size_t);




/*-------------------------------------------------------------------*
 *
 * BackpropLayer
 *
 *-------------------------------------------------------------------*/


/** Print the layer input.
 */
void BackpropLayer_PrintfInput(const struct BackpropLayer* self);


/** Print the layer output.
 */
void BackpropLayer_PrintfOutput(const struct BackpropLayer* self);


/** Print layer weights.
 *  Returns the number of bytes printed.
 */
size_t BackpropLayer_FprintfWeights(FILE* file, const struct BackpropLayer* self);


void BackpropLayer_PrintfWeights(const struct BackpropLayer* self);


void BackpropLayer_PutsWeights(const struct BackpropLayer* self);




/*-------------------------------------------------------------------*
 *
 * BackpropNetwork
 *
 *-------------------------------------------------------------------*/


/** Print the entire network.
 */
void BackpropNetwork_Printf(const struct BackpropNetwork* self);


/** Print the network input bytes.
 */
void BackpropNetwork_PrintfInput(const struct BackpropNetwork* self);


/** Puts the network input bytes. Adds newline character.
 */
void BackpropNetwork_PutsInput(const struct BackpropNetwork* self);


/** Prints the first network layer input values.
 */
void BackpropNetwork_PrintfLayersInput(const struct BackpropNetwork* self);


/** Prints the first network layer output values. Adds newline character.
 */
void BackpropNetwork_PutsLayersInput(const struct BackpropNetwork* self);


/** Prints the first network layer input values.
 */
void BackpropNetwork_PrintfLayersOutput(const struct BackpropNetwork* self);


/** Prints the last network layer output values. Adds newline character.
 */
void BackpropNetwork_PutsLayersOutput(const struct BackpropNetwork* self);


/** Print the network output.
 */
void BackpropNetwork_PrintfOutput(const struct BackpropNetwork* self);


/** Puts the network output. Adds newline character.
 */
void BackpropNetwork_PutsOutput(const struct BackpropNetwork* self);


/** Print the network x:y pair.
 */
void BackpropNetwork_PrintfInputOutput(const struct BackpropNetwork* self);


/** Print the network x:y pair.
 */
void BackpropNetwork_PutsInputOutput(const struct BackpropNetwork* self);


/** Print the layer weights.
 */
size_t BackpropNetwork_FprintfWeights(FILE* file, const struct BackpropNetwork* self);


void BackpropNetwork_PrintfWeights(const struct BackpropNetwork* self);


void BackpropNetwork_PutsWeights(const struct BackpropNetwork* self);


/** Save the network weights to a file.
 *  Returns number of bytes written.
 */
size_t BackpropNetwork_SaveWeights(const struct BackpropNetwork* self, const char* filename);


/** Load the network weights from a file.
 *  Returns number of bytes read.
 */
size_t BackpropNetwork_LoadWeights(struct BackpropNetwork* self, const char* filename);




/*-------------------------------------------------------------------*
 *
 * BackpropNetworkStats
 *
 *-------------------------------------------------------------------*/


size_t BackpropNetworkStats_Fprintf(const BackpropNetworkStats_t* self, FILE* file);


size_t BackpropNetworkStats_Printf(const BackpropNetworkStats_t* self);


/** Puts BackpropNetworkStats.
 */
size_t BackpropNetworkStats_Puts(const BackpropNetworkStats_t* self);




/*-------------------------------------------------------------------*
 *
 * BackpropTrainer
 *
 *-------------------------------------------------------------------*/

/** Set trainer to have default I/O settings.
 *  Will report results, errors, and abnormal conditions.
 */
void BackpropTrainer_SetToDefaultIO(struct BackpropTrainer* trainer);


/** Set trainer to verbose I/O settings.
 *  Will report all reportable data.
 */
void BackpropTrainer_SetToVerboseIO(struct BackpropTrainer* trainer);

void BackpropTrainer_PrintfAfterInput( const struct BackpropTrainer* trainer
                                     , const struct BackpropNetwork* network
                                     , const BACKPROP_BYTE_T* x
                                     , BACKPROP_SIZE_T x_size);

void BackpropTainer_PrintfAfterActivate(struct BackpropTrainer* trainer);
void BackpropTrainer_PrintfAfterExercisePair(struct BackpropTrainer* trainer);
void BackpropTrainer_PrintfAfterExercise(struct BackpropTrainer* trainer);

void BackpropTrainer_PrintfBeforeTrain(struct BackpropTrainer* trainer);
void BackpropTrainer_PrinfAfterTrainSuccess(struct BackpropTrainer* trainer);

void BackpropTrainer_PrintfAfterTrainFailure( struct BackpropTrainer* trainer
                                            , const struct BackpropTrainingStats* stats
                                            , struct BackpropNetwork* network
                                            , const BackpropTrainingSet_t* training_set
                                            , BACKPROP_FLOAT_T error);

void BackpropTrainer_PrintfAfterTrain(struct BackpropTrainer* trainer);

void BackpropTrainer_PrintfBeforeTrainBatch(struct BackpropTrainer* trainer);
void BackpropTrainer_PrintfAfterTrainBatch(struct BackpropTrainer* trainer);


void BackpropTrainer_PrintfAfterStagnateSet( struct BackpropTrainer* trainer
                                           , const struct BackpropTrainingStats* stats
                                           , struct BackpropNetwork* network
                                           , const BackpropTrainingSet_t* training_set
                                           , BACKPROP_SIZE_T batches
                                           , BACKPROP_SIZE_T stagnate_sets
                                           , BACKPROP_FLOAT_T error);

void BackpropTrainer_PrintfAfterMaxStagnateSets(struct BackpropTrainer* trainer);
void BackpropTrainer_PrintfAfterStubbornSet(struct BackpropTrainer* trainer);

void BackpropTrainer_PrintfAfterStagnateBatch( struct BackpropTrainer* trainer
                                             , const struct BackpropTrainingStats* stats
                                             , struct BackpropNetwork* network
                                             , const BackpropTrainingSet_t* training_set
                                             , BACKPROP_SIZE_T batches
                                             , BACKPROP_FLOAT_T error);

void BackpropTrainer_PrintfAfterMaxStagnateBatches(struct BackpropTrainer* trainer);
void BackpropTrainer_PrintfAfterStubbornBatch(struct BackpropTrainer* trainer);

void BackpropTrainer_PrintfBeforeTrainSet(struct BackpropTrainer* trainer);

void BackpropTrainer_PrintfBeforeTrainPair(struct BackpropTrainer* trainer);
void BackpropTrainer_PrintfAfterTrainPair(struct BackpropTrainer* trainer);

void BackpropTrainer_PrintfBeforeTeachPair(struct BackpropTrainer* trainer);


void BackpropTrainer_PrintfAfterTeachPair( const struct BackpropTrainer* trainer
                                         , const BackpropTrainingStats_t* stats
                                         , const struct BackpropNetwork* network
                                         , const BACKPROP_BYTE_T* x, const BACKPROP_SIZE_T x_size
                                         , const BACKPROP_BYTE_T* yd, const BACKPROP_SIZE_T yd_size
                                         , const BACKPROP_BYTE_T* y, const BACKPROP_SIZE_T y_size
                                         , BACKPROP_FLOAT_T error, BACKPROP_FLOAT_T weight_correction);


void BackpropTrainer_PutsAfterTeachPair( const struct BackpropTrainer* trainer
                                       , const struct BackpropTrainingStats* stats
                                       , const struct BackpropNetwork* network
                                       , const BACKPROP_BYTE_T* x, const BACKPROP_SIZE_T x_size
                                       , const BACKPROP_BYTE_T* yd, const BACKPROP_SIZE_T yd_size
                                       , const BACKPROP_BYTE_T* y, const BACKPROP_SIZE_T y_size
                                       , BACKPROP_FLOAT_T error, BACKPROP_FLOAT_T weight_correction);


void BackpropTrainer_FprintfAfterTrainSet( FILE* file
                                         , struct BackpropTrainer* trainer
                                         , const struct BackpropTrainingStats* stats
                                         , struct BackpropNetwork* network
                                         , const BackpropTrainingSet_t* training_set
                                         , BACKPROP_FLOAT_T error);


void BackpropTrainer_PrintfAfterTrainSet( struct BackpropTrainer* trainer
                                        , const struct BackpropTrainingStats* stats
                                        , struct BackpropNetwork* network
                                        , const BackpropTrainingSet_t* training_set
                                        , BACKPROP_FLOAT_T error);


void BackpropTrainer_PutsAfterTrainSet( struct BackpropTrainer* trainer
                                      , const struct BackpropTrainingStats* stats
                                      , struct BackpropNetwork* network
                                      , const BackpropTrainingSet_t* training_set
                                      , BACKPROP_FLOAT_T error);


void BackpropTrainer_AfterTrainBatch( struct BackpropTrainer* trainer
                                    , const struct BackpropTrainingStats* stats
                                    , struct BackpropNetwork* network
                                    , const BackpropTrainingSet_t* training_set
                                    , BACKPROP_SIZE_T batches
                                    , BACKPROP_FLOAT_T error);


void BackpropTrainer_PrintfAfterStagnateSet(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_SIZE_T batches, BACKPROP_SIZE_T stagnate_sets, BACKPROP_FLOAT_T error);


void BackpropTrainer_PutsAfterStagnateSet(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_SIZE_T batches, BACKPROP_SIZE_T stagnate_sets, BACKPROP_FLOAT_T error);


void BackpropTrainer_PrintfAfterStagnateBatch(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_SIZE_T batches, BACKPROP_FLOAT_T error);


void BackpropTrainer_PutsAfterStagnateBatch(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_SIZE_T batches, BACKPROP_FLOAT_T error);


void BackpropTrainer_PrintfAfterTrainSuccess(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_FLOAT_T error);


void BackpropTrainer_PutsAfterTrainSuccess(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_FLOAT_T error);

void BackpropTrainer_PrintfAfterTrainFailure(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_FLOAT_T error);


void BackpropTrainer_PutsAfterTrainFailure(struct BackpropTrainer* trainer, const struct BackpropTrainingStats* stats, struct BackpropNetwork* network, const BackpropTrainingSet_t* training_set, BACKPROP_FLOAT_T error);




/*-------------------------------------------------------------------*
 *
 * BackpropTrainingSet
 *
 *-------------------------------------------------------------------*/


size_t BackpropTrainingSet_Fprintf(const BackpropTrainingSet_t* self, FILE* file);


size_t BackpropTrainingSet_Printf(const BackpropTrainingSet_t* self);


size_t BackpropTrainingSet_Puts(const BackpropTrainingSet_t* self);


size_t BackpropTrainingSet_Fparsef(BackpropTrainingSet_t* self, FILE* file);


size_t BackpropTrainingSet_LoadDimensions(BackpropTrainingSetDimensions_t* dims, const char* filename);


size_t BackpropTrainingSet_Load(BackpropTrainingSet_t* self, const char* filename);


size_t BackpropTrainingSet_Save(const BackpropTrainingSet_t* self, const char* filename);




/*-------------------------------------------------------------------*
 *
 * BackpropExerciseStats
 *
 *-------------------------------------------------------------------*/


size_t BackpropExerciseStats_Fprintf(const struct BackpropExerciseStats* stats, FILE* file);


size_t BackpropExerciseStats_Printf(const struct BackpropExerciseStats* stats);


size_t BackpropExerciseStats_Puts(const struct BackpropExerciseStats* stats);




/*-------------------------------------------------------------------*
 *
 * BackpropTrainingStats
 *
 *-------------------------------------------------------------------*/


size_t BackpropTrainingStats_Fprintf(const struct BackpropTrainingStats* stats, FILE* file);


size_t BackpropTrainingStats_Printf(const struct BackpropTrainingStats* stats);


size_t BackpropTrainingStats_Puts(const struct BackpropTrainingStats* stats);




/*-------------------------------------------------------------------*
 *
 * BackpropEvolutionStats
 *
 *-------------------------------------------------------------------*/


size_t BackpropEvolutionStats_Fprintf(const struct BackpropEvolutionStats* stats, FILE* file);


size_t BackpropEvolutionStats_Printf(const struct BackpropEvolutionStats* stats);


size_t BackpropEvolutionStats_Puts(const struct BackpropEvolutionStats* stats);




#endif/*BACKPROP_IO_H*/
