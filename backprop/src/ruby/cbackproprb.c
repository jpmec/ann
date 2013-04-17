// Include the Ruby headers and goodies
#include "ruby.h"
#include "../C/backprop.h"
#include "../C/backprop.c"
#include "../C/backprop_io.h"
#include "../C/backprop_io.c"




#define VALUE_TO_C_PTR(_c_type_, _c_val_, _rb_val_) \
  _c_type_* _c_val_; \
  Data_Get_Struct(_rb_val_, _c_type_, _c_val_); \
  if (!_c_val_) { \
    rb_raise(rb_eArgError, "invalid arg"); \
    return Qnil; \
  }




typedef struct BackpropNetwork BackpropNetwork_t;


// Defining a space for information and references about the module to be stored internally
static VALUE cBackproprb = Qnil;
static VALUE cBackpropLayer = Qnil;
static VALUE cBackpropNetwork = Qnil;
static VALUE cBackpropNetworkStats = Qnil;
static VALUE cBackpropTrainer = Qnil;
static VALUE cBackpropTrainingSet = Qnil;
static VALUE cBackpropTrainingStats = Qnil;
static VALUE cBackpropExerciseStats = Qnil;
static VALUE cBackpropEvolutionStats = Qnil;
static VALUE cBackpropEvolver = Qnil;


#if defined(USE_BACKPROP_TRACE)
#define BACKPROP_TRACE(_arg_)    puts(_arg_)
#else
#define BACKPROP_TRACE(_arg_)
#endif




static void* cBackprop_Malloc (size_t size)
{
  BACKPROP_TRACE(__FUNCTION__);
  return xmalloc(size);
}




static void cBackprop_Free(void* obj)
{
  BACKPROP_TRACE(__FUNCTION__);
  xfree(obj);
}




VALUE cBackprop_sigmoid(VALUE self, VALUE x)
{
  BACKPROP_TRACE(__FUNCTION__);

  const BACKPROP_FLOAT_T y = Backprop_Sigmoid(NUM2DBL(x));
  return rb_float_new(y);
}




VALUE cBackprop_uniform_random_int(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  const int i = Backprop_UniformRandomInt();
  return rb_float_new(i);
}




VALUE cBackprop_used(void)
{
  BACKPROP_TRACE(__FUNCTION__);

  const size_t in_use = Backprop_GetMallocInUse();
  return INT2NUM(in_use);
}





//------------------------------------------------------------------------------
//
// BackpropLayer
//
//------------------------------------------------------------------------------


static VALUE cBackpropLayer_get_W(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  VALUE a = rb_ary_new();

  BACKPROP_FLOAT_T* w = layer->W;

  size_t y = layer->y_count;
  do
  {
    VALUE row = rb_ary_new();

    size_t x = layer->x_count;
    do
    {
      rb_ary_push(row, rb_float_new(*w));
      ++w;
    } while(--x);

    rb_ary_push(a, row);
  } while (--y);

  return a;
}




static VALUE cBackpropLayer_set_W(VALUE self, VALUE vals)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  const ID f = rb_intern("length");
  VALUE length_val = rb_funcall(vals, f, 0, 0);
  size_t length = NUM2INT(length_val);

  const BACKPROP_SIZE_T W_count = BackpropLayer_GetWeightsCount(layer);

  const long end = (W_count < length) ? W_count : length;

  for (long i = 0; i < end; ++i)
  {
    VALUE val = rb_ary_entry(vals, i);
    layer->W[i] = NUM2DBL(val);
  }

  return self;
}




static VALUE cBackpropLayer_get_g(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  VALUE a = rb_ary_new();

  BACKPROP_FLOAT_T* g = layer->g;

  size_t i = layer->y_count;
  do
  {
    rb_ary_push(a, rb_float_new(*g));
    ++g;
  } while(--i);

  return a;
}




static VALUE cBackpropLayer_set_g(VALUE self, VALUE vals)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  const ID f = rb_intern("length");
  VALUE length_val = rb_funcall(vals, f, 0, 0);
  size_t length = NUM2INT(length_val);

  const long end = (layer->y_count < length) ? layer->y_count : length;

  for (long i = 0; i < end; ++i)
  {
    VALUE val = rb_ary_entry(vals, i);
    layer->g[i] = NUM2DBL(val);
  }

  return self;
}




static VALUE cBackpropLayer_get_x(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  VALUE a = rb_ary_new();

  BACKPROP_FLOAT_T* x = layer->x;

  size_t i = layer->x_count;
  do
  {
    rb_ary_push(a, rb_float_new(*x));
    ++x;
  } while(--i);

  return a;
}




static VALUE cBackpropLayer_set_x(VALUE self, VALUE vals)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  const ID f = rb_intern("length");
  VALUE length_val = rb_funcall(vals, f, 0, 0);
  size_t length = NUM2INT(length_val);

  const long end = (layer->x_count < length) ? layer->x_count : length;

  for (long i = 0; i < end; ++i)
  {
    VALUE val = rb_ary_entry(vals, i);
    layer->x[i] = NUM2DBL(val);
  }

  return self;
}




static VALUE cBackpropLayer_get_y(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  VALUE a = rb_ary_new();

  const ID f = rb_intern("push");

  BACKPROP_FLOAT_T* y = layer->y;

  size_t i = layer->y_count;
  do
  {
    rb_funcall(a, f, 1, rb_float_new(*y));
    ++y;
  } while(--i);

  return a;
}




static VALUE cBackpropLayer_set_y(VALUE self, VALUE vals)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  const ID f = rb_intern("length");
  VALUE length_val = rb_funcall(vals, f, 0, 0);
  size_t length = NUM2INT(length_val);

  const long end = (layer->y_count < length) ? layer->y_count : length;

  for (long i = 0; i < end; ++i)
  {
    VALUE val = rb_ary_entry(vals, i);
    layer->y[i] = NUM2DBL(val);
  }

  return self;
}




static VALUE cBackpropLayer_get_W_count(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  const BACKPROP_SIZE_T count = BackpropLayer_GetWeightsCount(layer);

  return INT2NUM(count);
}




static VALUE cBackpropLayer_get_W_sum(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  const BACKPROP_FLOAT_T sum = BackpropLayer_GetWeightsSum(layer);

  return rb_float_new(sum);
}




static VALUE cBackpropLayer_get_W_mean(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  const BACKPROP_FLOAT_T mean = BackpropLayer_GetWeightsMean(layer);

  return rb_float_new(mean);
}




static VALUE cBackpropLayer_get_W_stddev(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  const BACKPROP_FLOAT_T value = BackpropLayer_GetWeightsStdDev(layer);

  return rb_float_new(value);
}




static VALUE cBackpropLayer_get_x_count(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  return INT2NUM(layer->x_count);
}




static VALUE cBackpropLayer_get_y_count(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  return INT2NUM(layer->y_count);
}




static VALUE cBackpropLayer_to_hash(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  VALUE hash = rb_hash_new();
  rb_hash_aset(hash, rb_str_new2("w"), cBackpropLayer_get_W(self));
  rb_hash_aset(hash, rb_str_new2("x"), cBackpropLayer_get_x(self));
  rb_hash_aset(hash, rb_str_new2("y"), cBackpropLayer_get_y(self));

  return hash;
}




static VALUE cBackpropLayer_randomize(VALUE self, VALUE gain_val)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  BACKPROP_FLOAT_T gain = NUM2DBL(gain_val);

  BackpropLayer_Randomize(layer, gain);

  return self;
}




static VALUE cBackpropLayer_identity(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  BackpropLayer_Identity(layer);

  return self;
}




static VALUE cBackpropLayer_reset(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  BackpropLayer_Reset(layer);

  return self;
}




static VALUE cBackpropLayer_prune(VALUE self, VALUE threshold_val)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  BACKPROP_FLOAT_T threshold = NUM2DBL(threshold_val);

  BackpropLayer_Prune(layer, threshold);

  return self;
}




static VALUE cBackpropLayer_activate(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  BackpropLayer_Activate(layer);

  return self;
}




static void cBackpropLayer_Free(struct BackpropLayer* layer)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropLayer_Free(layer);
}




static VALUE cBackpropLayer_initialize(VALUE self, VALUE x_size, VALUE y_size)
{
  BACKPROP_TRACE(__FUNCTION__);

  return self;
}




static VALUE cBackpropLayer_new(VALUE klass, VALUE x_count_val, VALUE y_count_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BACKPROP_SIZE_T x_count = NUM2UINT(x_count_val);
  BACKPROP_SIZE_T  y_count = NUM2UINT(y_count_val);

  // allocate structure
  struct BackpropLayer* layer = BackpropLayer_Malloc(x_count, y_count);

  // wrap it in a ruby object, this will cause GC to call free function
  VALUE tdata = Data_Wrap_Struct(klass, 0, cBackpropLayer_Free, layer);

  // call initialize
  VALUE argv[2] = {x_count_val, y_count_val};
  rb_obj_call_init(tdata, 2, argv);

  return tdata;
}








//------------------------------------------------------------------------------
//
// BackpropNetworkStats
//
//------------------------------------------------------------------------------




static VALUE CBackpropNetworkStats_get_x_size(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetworkStats_t* stats;
  Data_Get_Struct(self, BackpropNetworkStats_t, stats);

  return INT2NUM(stats->x_size);
}




static VALUE CBackpropNetworkStats_set_x_size(VALUE self, VALUE val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetworkStats_t* stats;
  Data_Get_Struct(self, BackpropNetworkStats_t, stats);

  stats->x_size = NUM2INT(val);

  return self;
}



static VALUE CBackpropNetworkStats_get_y_size(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetworkStats_t* stats;
  Data_Get_Struct(self, BackpropNetworkStats_t, stats);

  return INT2NUM(stats->y_size);
}




static VALUE CBackpropNetworkStats_set_y_size(VALUE self, VALUE val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetworkStats_t* stats;
  Data_Get_Struct(self, BackpropNetworkStats_t, stats);

  stats->y_size = NUM2INT(val);

  return self;
}




static VALUE CBackpropNetworkStats_get_layers_count(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetworkStats_t* stats;
  Data_Get_Struct(self, BackpropNetworkStats_t, stats);

  return INT2NUM(stats->layers_count);
}




static VALUE CBackpropNetworkStats_set_layers_count(VALUE self, VALUE val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetworkStats_t* stats;
  Data_Get_Struct(self, BackpropNetworkStats_t, stats);

  stats->layers_count = NUM2INT(val);

  return self;
}




static VALUE CBackpropNetworkStats_get_layers_size(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetworkStats_t* stats;
  Data_Get_Struct(self, BackpropNetworkStats_t, stats);

  return INT2NUM(stats->layers_size);
}




static VALUE CBackpropNetworkStats_set_layers_size(VALUE self, VALUE val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetworkStats_t* stats;
  Data_Get_Struct(self, BackpropNetworkStats_t, stats);

  stats->layers_size = NUM2INT(val);

  return self;
}




static VALUE CBackpropNetworkStats_get_layers_w_count(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetworkStats_t* stats;
  Data_Get_Struct(self, BackpropNetworkStats_t, stats);

  return INT2NUM(stats->layers_W_count);
}




static VALUE CBackpropNetworkStats_set_layers_w_count(VALUE self, VALUE val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetworkStats_t* stats;
  Data_Get_Struct(self, BackpropNetworkStats_t, stats);

  stats->layers_W_count = NUM2INT(val);

  return self;
}




static VALUE CBackpropNetworkStats_get_layers_w_size(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetworkStats_t* stats;
  Data_Get_Struct(self, BackpropNetworkStats_t, stats);

  return INT2NUM(stats->layers_W_size);
}




static VALUE CBackpropNetworkStats_set_layers_w_size(VALUE self, VALUE val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetworkStats_t* stats;
  Data_Get_Struct(self, BackpropNetworkStats_t, stats);

  stats->layers_W_size = NUM2INT(val);

  return self;
}




static VALUE CBackpropNetworkStats_get_layers_w_avg(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetworkStats_t* stats;
  Data_Get_Struct(self, BackpropNetworkStats_t, stats);

  return rb_float_new(stats->layers_W_avg);
}




static VALUE CBackpropNetworkStats_set_layers_w_avg(VALUE self, VALUE val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetworkStats_t* stats;
  Data_Get_Struct(self, BackpropNetworkStats_t, stats);

  stats->layers_W_avg = NUM2DBL(val);

  return self;
}




static VALUE CBackpropNetworkStats_get_layers_w_stddev(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetworkStats_t* stats;
  Data_Get_Struct(self, BackpropNetworkStats_t, stats);

  return rb_float_new(stats->layers_W_stddev);
}




static VALUE CBackpropNetworkStats_set_layers_w_stddev(VALUE self, VALUE val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetworkStats_t* stats;
  Data_Get_Struct(self, BackpropNetworkStats_t, stats);

  stats->layers_W_stddev = NUM2DBL(val);

  return self;
}




static VALUE CBackpropNetworkStats_to_hash(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetworkStats_t* stats;
  Data_Get_Struct(self, BackpropNetworkStats_t, stats);

  VALUE hash = rb_hash_new();
  rb_hash_aset(hash, rb_str_new2("x_size"), CBackpropNetworkStats_get_x_size(self));
  rb_hash_aset(hash, rb_str_new2("y_size"), CBackpropNetworkStats_get_y_size(self));
  rb_hash_aset(hash, rb_str_new2("layers_count"), CBackpropNetworkStats_get_layers_count(self));
  rb_hash_aset(hash, rb_str_new2("layers_size"), CBackpropNetworkStats_get_layers_size(self));
  rb_hash_aset(hash, rb_str_new2("layers_w_count"), CBackpropNetworkStats_get_layers_w_count(self));
  rb_hash_aset(hash, rb_str_new2("layers_w_size"), CBackpropNetworkStats_get_layers_w_size(self));
  rb_hash_aset(hash, rb_str_new2("layers_w_avg"), CBackpropNetworkStats_get_layers_w_avg(self));
  rb_hash_aset(hash, rb_str_new2("layers_w_stddev"), CBackpropNetworkStats_get_layers_w_stddev(self));

  return hash;
}




static VALUE CBackpropNetworkStats_new(VALUE klass)
{
  BACKPROP_TRACE(__FUNCTION__);

  // allocate structure
  BackpropNetworkStats_t* instance = xmalloc(sizeof(BackpropNetworkStats_t));

  // initialize structure
  memset(instance, 0, sizeof(BackpropNetworkStats_t));

  // wrap it in a ruby object, this will cause GC to call free function
  VALUE tdata = Data_Wrap_Struct(klass, 0, xfree, instance);

  // call initialize
  rb_obj_call_init(tdata, 0, 0);

  return tdata;
}








//------------------------------------------------------------------------------
//
// BackpropNetwork
//
//------------------------------------------------------------------------------


static VALUE cBackpropNetwork_activate(VALUE self, VALUE input)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  char* cstr_in = StringValueCStr(input);

  BackpropNetwork_InputCStr(network, cstr_in);

  BackpropNetwork_Activate(network);


  // TODO MAKE THIS VARIABLE LENGTH
  char cstr_out[2] = {0};

  BackpropNetwork_GetOutputCStr(network, cstr_out, sizeof(cstr_out) - 1);

  return rb_str_new2(cstr_out);
}




static VALUE cBackpropNetwork_x_size(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  BACKPROP_SIZE_T x_size = BackpropNetwork_GetXSize(network);

  return INT2NUM(x_size);
}




static VALUE cBackpropNetwork_get_x(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  const size_t x_size = network->x.size;

  VALUE x_array = rb_ary_new2(x_size);

  for (size_t i = 0; i < x_size; ++i)
  {
    rb_ary_store(x_array, i, INT2NUM(network->x.data[i]));
  }

  return x_array;
}




static VALUE cBackpropNetwork_y_size(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  const BACKPROP_SIZE_T y_size = BackpropNetwork_GetXSize(network);

  return INT2NUM(y_size);
}




static VALUE cBackpropNetwork_get_y(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  const size_t y_size = network->y.size;

  VALUE y_array = rb_ary_new2(y_size);

  for (size_t i = 0; i < y_size; ++i)
  {
    rb_ary_store(y_array, i, INT2NUM(network->y.data[i]));
  }

  return y_array;
}




static VALUE cBackpropNetwork_layers_count(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  const struct BackpropLayersArray* layers = BackpropNetwork_GetLayers(network);

  const BACKPROP_SIZE_T layers_count = layers->count;

  return INT2NUM(layers_count);
}



static VALUE cBackpropNetwork_get_layer(VALUE self, VALUE index_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  int index = NUM2INT(index_val);

  struct BackpropLayersArray* layers = BackpropNetwork_GetLayers(network);
  const BACKPROP_SIZE_T layers_count = layers->count;

  if (index < 0)
  {
    index = layers_count + index;

    if (index < 0)
    {
    return Qnil;
    }
  }

  if (index > layers_count)
  {
    return Qnil;
  }

  struct BackpropLayer* layer = layers->data + index;

  VALUE tdata = Data_Wrap_Struct(cBackpropLayer, 0, 0, layer);

  return tdata;
}


static VALUE cBackpropNetwork_get_jitter(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  BACKPROP_FLOAT_T jitter = BackpropNetwork_GetJitter(network);

  return rb_float_new(jitter);
}




static VALUE cBackpropNetwork_set_jitter(VALUE self, VALUE value)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  BACKPROP_FLOAT_T jitter = NUM2DBL(value);

  BackpropNetwork_SetJitter(network, jitter);

  return self;
}




static VALUE cBackpropNetwork_randomize(VALUE self, VALUE seed_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  unsigned int seed = NUM2UINT(seed_val);

  BackpropNetwork_Randomize(network, seed);

  return self;
}




static VALUE cBackpropNetwork_identity(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  BackpropNetwork_Identity(network);

  return self;
}




static VALUE cBackpropNetwork_reset(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  BackpropNetwork_Reset(network);

  return self;
}




static VALUE cBackpropNetwork_prune(VALUE self, VALUE threshold_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  BACKPROP_FLOAT_T threshold = NUM2DBL(threshold_val);

  BackpropNetwork_Prune(network, threshold);

  return self;
}




static VALUE cBackpropNetwork_get_stats(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  VALUE stats_value = CBackpropNetworkStats_new(cBackpropNetworkStats);

  BackpropNetworkStats_t* stats;
  Data_Get_Struct(stats_value, BackpropNetworkStats_t, stats);

  BackpropNetwork_GetStats(network, stats);

  return stats_value;
}




static VALUE cBackpropNetwork_to_hash(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  VALUE x_array = cBackpropNetwork_get_x(self);
  VALUE y_array = cBackpropNetwork_get_y(self);

  const BACKPROP_SIZE_T layers_count = network->layers.count;

  VALUE layers_array = rb_ary_new2(layers_count);

  for (BACKPROP_SIZE_T i = 0; i < layers_count; ++i)
  {
    VALUE layer = cBackpropNetwork_get_layer(self, INT2NUM(i));
    rb_ary_store(layers_array, i, cBackpropLayer_to_hash(layer));
  }

  VALUE hash = rb_hash_new();
  rb_hash_aset(hash, rb_str_new2("layers"), layers_array);
  rb_hash_aset(hash, rb_str_new2("x"), x_array);
  rb_hash_aset(hash, rb_str_new2("y"), y_array);
  rb_hash_aset(hash, rb_str_new2("jitter"), cBackpropNetwork_get_jitter(self));

  return hash;
}




static VALUE cBackpropNetwork_to_file(VALUE self, VALUE file_name_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  VALUE_TO_C_PTR(BackpropNetwork_t, network, self);

  const char* file_name = StringValueCStr(file_name_val);

  BackpropNetwork_SaveWeights(network, file_name);

  return self;
}




static VALUE cBackpropNetwork_from_file(VALUE self, VALUE file_name_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  VALUE_TO_C_PTR(BackpropNetwork_t, network, self);

  const char* file_name = StringValueCStr(file_name_val);

  BackpropNetwork_LoadWeights(network, file_name);

  return self;
}




static VALUE cBackpropNetwork_initialize(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  cBackpropNetwork_reset(self);

  return self;
}




static void cBackpropNetwork_free(struct BackpropNetwork* network)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_Free(network);
}




static VALUE cBackpropNetwork_new(VALUE klass, VALUE x_size, VALUE y_size, VALUE layer_count)
{
  BACKPROP_TRACE(__FUNCTION__);

  // allocate structure
  struct BackpropNetwork* network = BackpropNetwork_Malloc( NUM2INT(x_size)
                                                          , NUM2INT(y_size)
                                                          , NUM2INT(layer_count)
                                                          , true);

  // wrap it in a ruby object, this will cause GC to call free function
  VALUE tdata = Data_Wrap_Struct(klass, 0, cBackpropNetwork_free, network);

  // call initialize
  VALUE argv[3] = {x_size, y_size, layer_count};
  rb_obj_call_init(tdata, 3, argv);

  return tdata;
}








//------------------------------------------------------------------------------
//
// BackpropTrainingSet
//
//------------------------------------------------------------------------------


static VALUE cBackpropTrainingSet_x_at(VALUE self, VALUE index)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingSet_t* training_set;
  Data_Get_Struct(self, BackpropTrainingSet_t, training_set);

  const size_t x_size = training_set->dims.x_size;
  char* result = malloc(x_size);

  memcpy( result
        , training_set->x + NUM2INT(index) * x_size
        , x_size);

  VALUE result_value = rb_str_new2(result);

  free(result);

  return result_value;
}




static VALUE cBackpropTrainingSet_y_at(VALUE self, VALUE index)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingSet_t* training_set;
  Data_Get_Struct(self, BackpropTrainingSet_t, training_set);

  const size_t y_size = training_set->dims.y_size;
  char* result = malloc(y_size);

  memcpy( result
        , training_set->y + NUM2INT(index) * y_size
        , y_size);

  VALUE result_value = rb_str_new2(result);

  free(result);

  return result_value;
}




static VALUE cBackpropTrainingSet_count(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingSet_t* training_set;
  Data_Get_Struct(self, BackpropTrainingSet_t, training_set);

  return INT2NUM(training_set->dims.count);
}




static VALUE cBackpropTrainingSet_x_size(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingSet_t* training_set;
  Data_Get_Struct(self, BackpropTrainingSet_t, training_set);

  return INT2NUM(training_set->dims.x_size);
}




static VALUE cBackpropTrainingSet_y_size(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingSet_t* training_set;
  Data_Get_Struct(self, BackpropTrainingSet_t, training_set);

  return INT2NUM(training_set->dims.y_size);
}




static VALUE cBackpropTrainingSet_to_hash(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingSet_t* training_set;
  Data_Get_Struct(self, BackpropTrainingSet_t, training_set);

  const BACKPROP_SIZE_T count = training_set->dims.count;

  VALUE hash = rb_hash_new();

  VALUE x_array = rb_ary_new2(count);
  VALUE y_array = rb_ary_new2(count);

  for (BACKPROP_SIZE_T i = 0; i < count; ++i)
  {
    rb_ary_store(x_array, i, cBackpropTrainingSet_x_at(self, INT2NUM(i)));
    rb_ary_store(y_array, i, cBackpropTrainingSet_y_at(self, INT2NUM(i)));
  }

  rb_hash_aset(hash, rb_str_new2("count"), INT2NUM(count));
  rb_hash_aset(hash, rb_str_new2("x_size"), INT2NUM(training_set->dims.x_size));
  rb_hash_aset(hash, rb_str_new2("y_size"), INT2NUM(training_set->dims.y_size));
  rb_hash_aset(hash, rb_str_new2("x"), x_array);
  rb_hash_aset(hash, rb_str_new2("y"), y_array);

  return hash;
}




static VALUE cBackpropTrainingSet_initialize(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  return self;
}




static void cBackpropTrainingSet_Free(struct BackpropTrainingSet* self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingSet_Free(self);
}




static VALUE cBackpropTrainingSet_new(VALUE klass, VALUE x_value, VALUE y_value)
{
  BACKPROP_TRACE(__FUNCTION__);

  VALUE x_count = rb_funcall(x_value, rb_intern("size"),  0);
  VALUE y_count = rb_funcall(y_value, rb_intern("size"),  0);

  if (x_count != y_count)
  {
    return Qnil;
  }

  VALUE x_str_val = rb_funcall(x_value, rb_intern("at"), 1, INT2NUM(0));
  VALUE y_str_val = rb_funcall(y_value, rb_intern("at"), 1, INT2NUM(0));

  VALUE x_size_val = rb_funcall(x_str_val, rb_intern("size"), 0);
  VALUE y_size_val = rb_funcall(y_str_val, rb_intern("size"), 0);

  size_t count = NUM2INT(x_count);
  size_t x_size = NUM2INT(x_size_val);
  size_t y_size = NUM2INT(y_size_val);

  BackpropTrainingSet_t* instance = BackpropTrainingSet_Malloc(count, x_size, y_size);

  // wrap it in a ruby object, this will cause GC to call free function
  VALUE tdata = Data_Wrap_Struct(klass, 0, cBackpropTrainingSet_Free, instance);

  // call initialize
  //VALUE argv[3] = {count, x_size, y_size};
  //  rb_obj_call_init(tdata, 0, 0);

  for (size_t i = 0; i < count; ++i)
  {
    VALUE x_str_val = rb_funcall(x_value, rb_intern("at"), 1, INT2NUM(i));
    char* x_str = StringValueCStr(x_str_val);

    BACKPROP_BYTE_T* x = instance->x + i * x_size;
    memcpy(x, x_str, x_size);
  }

  for (size_t j = 0; j < count; ++j)
  {
    VALUE y_str_val = rb_funcall(y_value, rb_intern("at"), 1, INT2NUM(j));
    char* y_str = StringValueCStr(y_str_val);

    BACKPROP_BYTE_T* y = instance->y + j * y_size;
    memcpy(y, y_str, y_size);
  }

  return tdata;
}




//------------------------------------------------------------------------------
//
// BackpropExerciseStats
//
//------------------------------------------------------------------------------


static VALUE cBackpropExerciseStats_exercise_clock_ticks(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropExerciseStats_t* stats;
  Data_Get_Struct(self, BackpropExerciseStats_t, stats);

  return INT2NUM(stats->exercise_clock_ticks);
}




static VALUE cBackpropExerciseStats_activate_count(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropExerciseStats_t* stats;
  Data_Get_Struct(self, BackpropExerciseStats_t, stats);

  return INT2NUM(stats->activate_count);
}



static VALUE cBackpropExerciseStats_error(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropExerciseStats_t* stats;
  Data_Get_Struct(self, BackpropExerciseStats_t, stats);

  return rb_float_new(stats->error);
}




static VALUE cBackpropExerciseStats_to_hash(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropExerciseStats_t* stats;
  Data_Get_Struct(self, BackpropExerciseStats_t, stats);

  VALUE hash = rb_hash_new();
  rb_hash_aset(hash, rb_str_new2("exercise_clock_ticks"), cBackpropExerciseStats_exercise_clock_ticks(self));
  rb_hash_aset(hash, rb_str_new2("activate_count"), cBackpropExerciseStats_activate_count(self));
  rb_hash_aset(hash, rb_str_new2("error"), cBackpropExerciseStats_error(self));

  return hash;
}




static VALUE cBackpropExerciseStats_new(VALUE klass)
{
  BACKPROP_TRACE(__FUNCTION__);

  // allocate structure
  BackpropExerciseStats_t* instance = xmalloc(sizeof(BackpropExerciseStats_t));

  // initialize structure
  memset(instance, 0, sizeof(BackpropExerciseStats_t));

  // wrap it in a ruby object, this will cause GC to call free function
  VALUE tdata = Data_Wrap_Struct(klass, 0, xfree, instance);

  // call initialize
  rb_obj_call_init(tdata, 0, 0);

  return tdata;
}




//------------------------------------------------------------------------------
//
// BackpropTrainingStats
//
//------------------------------------------------------------------------------


static VALUE cBackpropTrainingStats_set_weight_correction_total(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingStats_t* stats;
  Data_Get_Struct(self, BackpropTrainingStats_t, stats);

  return rb_float_new(stats->set_weight_correction_total);
}




static VALUE cBackpropTrainingStats_batch_weight_correction_total(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingStats_t* stats;
  Data_Get_Struct(self, BackpropTrainingStats_t, stats);

  return rb_float_new(stats->batch_weight_correction_total);
}




static VALUE cBackpropTrainingStats_teach_total(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingStats_t* stats;
  Data_Get_Struct(self, BackpropTrainingStats_t, stats);

  return INT2NUM(stats->teach_total);
}




static VALUE cBackpropTrainingStats_pair_total(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingStats_t* stats;
  Data_Get_Struct(self, BackpropTrainingStats_t, stats);

  return INT2NUM(stats->pair_total);
}




static VALUE cBackpropTrainingStats_set_total(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingStats_t* stats;
  Data_Get_Struct(self, BackpropTrainingStats_t, stats);

  return INT2NUM(stats->set_total);
}




static VALUE cBackpropTrainingStats_batches_total(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingStats_t* stats;
  Data_Get_Struct(self, BackpropTrainingStats_t, stats);

  return INT2NUM(stats->batches_total);
}




static VALUE cBackpropTrainingStats_stubborn_batches_total(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingStats_t* stats;
  Data_Get_Struct(self, BackpropTrainingStats_t, stats);

  return INT2NUM(stats->stubborn_batches_total);
}




static VALUE cBackpropTrainingStats_stagnate_batches_total(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingStats_t* stats;
  Data_Get_Struct(self, BackpropTrainingStats_t, stats);

  return INT2NUM(stats->stagnate_batches_total);
}




static VALUE cBackpropTrainingStats_train_clock(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingStats_t* stats;
  Data_Get_Struct(self, BackpropTrainingStats_t, stats);

  return INT2NUM(stats->train_clock);
}




static VALUE cBackpropTrainingStats_to_hash(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingStats_t* stats;
  Data_Get_Struct(self, BackpropTrainingStats_t, stats);

  VALUE hash = rb_hash_new();
  rb_hash_aset(hash, rb_str_new2("set_weight_correction_total"), cBackpropTrainingStats_set_weight_correction_total(self));
  rb_hash_aset(hash, rb_str_new2("batch_weight_correction_total"), cBackpropTrainingStats_batch_weight_correction_total(self));
  rb_hash_aset(hash, rb_str_new2("teach_total"), cBackpropTrainingStats_teach_total(self));
  rb_hash_aset(hash, rb_str_new2("pair_total"), cBackpropTrainingStats_pair_total(self));
  rb_hash_aset(hash, rb_str_new2("set_total"), cBackpropTrainingStats_set_total(self));
  rb_hash_aset(hash, rb_str_new2("batches_total"), cBackpropTrainingStats_batches_total(self));
  rb_hash_aset(hash, rb_str_new2("stubborn_batches_total"), cBackpropTrainingStats_stubborn_batches_total(self));
  rb_hash_aset(hash, rb_str_new2("stagnate_batches_total"), cBackpropTrainingStats_stagnate_batches_total(self));
  rb_hash_aset(hash, rb_str_new2("train_clock"), cBackpropTrainingStats_train_clock(self));

  return hash;
}




static VALUE cBackpropTrainingStats_new(VALUE klass)
{
  BACKPROP_TRACE(__FUNCTION__);

  // allocate structure
  BackpropTrainingStats_t* instance = xmalloc(sizeof(BackpropTrainingStats_t));

  // initialize structure
  memset(instance, 0, sizeof(BackpropTrainingStats_t));

  // wrap it in a ruby object, this will cause GC to call free function
  VALUE tdata = Data_Wrap_Struct(klass, 0, xfree, instance);

  // call initialize
  rb_obj_call_init(tdata, 0, 0);

  return tdata;
}




//------------------------------------------------------------------------------
//
// BackpropTrainer
//
//------------------------------------------------------------------------------


static VALUE cBackpropTrainer_teach_pair( VALUE trainer_val
                                        , VALUE training_stats_val
                                        , VALUE network_val
                                        , VALUE x_val
                                        , VALUE y_desired_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer = NULL;
  Data_Get_Struct(trainer_val, BackpropTrainer_t, trainer);

  if (!trainer)
  {
    rb_raise(rb_eArgError, "NULL trainer");
    return Qnil;
  }

  BackpropTrainingStats_t* training_stats = NULL;
  Data_Get_Struct(training_stats_val, BackpropTrainingStats_t, training_stats);

  if (!training_stats)
  {
    rb_raise(rb_eArgError, "NULL training_stats");
    return Qnil;
  }

  BackpropNetwork_t* network = NULL;
  Data_Get_Struct(network_val, BackpropNetwork_t, network);

  if (!network)
  {
    rb_raise(rb_eArgError, "NULL network");
    return Qnil;
  }

  char* x_str = StringValueCStr(x_val);

  if (!x_str)
  {
    rb_raise(rb_eArgError, "NULL x");
    return Qnil;
  }

  BACKPROP_SIZE_T x_len = strlen(x_str);

  if (x_len <= 0)
  {
    rb_raise(rb_eArgError, "invalid x length");
    return Qnil;
  }

  char* y_desired_str = StringValueCStr(y_desired_val);

  if (!y_desired_str)
  {
    rb_raise(rb_eArgError, "NULL y");
    return Qnil;
  }

  BACKPROP_SIZE_T y_desired_len = strlen(y_desired_str);

  if (y_desired_len <= 0)
  {
    rb_raise(rb_eArgError, "invalid y length");
    return Qnil;
  }

  BACKPROP_FLOAT_T result = BackpropTrainer_TeachPair( trainer
                                                     , training_stats
                                                     , network
                                                     , x_str, x_len
                                                     , y_desired_str, y_desired_len);

  return rb_float_new(result);
}




static VALUE cBackpropTrainer_train_pair( VALUE trainer_val
                                        , VALUE training_stats_val
                                        , VALUE network_val
                                        , VALUE x_val
                                        , VALUE y_desired_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer = NULL;
  Data_Get_Struct(trainer_val, BackpropTrainer_t, trainer);

  if (!trainer)
  {
    rb_raise(rb_eArgError, "NULL trainer");
    return Qnil;
  }

  BackpropTrainingStats_t* training_stats = NULL;
  Data_Get_Struct(training_stats_val, BackpropTrainingStats_t, training_stats);

  if (!training_stats)
  {
    rb_raise(rb_eArgError, "NULL training_stats");
    return Qnil;
  }

  BackpropNetwork_t* network = NULL;
  Data_Get_Struct(network_val, BackpropNetwork_t, network);

  if (!network)
  {
    rb_raise(rb_eArgError, "NULL network");
    return Qnil;
  }

  char* x_str = StringValueCStr(x_val);

  if (!x_str)
  {
    rb_raise(rb_eArgError, "NULL x");
    return Qnil;
  }

  BACKPROP_SIZE_T x_len = strlen(x_str);

  if (x_len <= 0)
  {
    rb_raise(rb_eArgError, "invalid x length");
    return Qnil;
  }

  char* y_desired_str = StringValueCStr(y_desired_val);

  if (!y_desired_str)
  {
    rb_raise(rb_eArgError, "NULL y");
    return Qnil;
  }

  BACKPROP_SIZE_T y_desired_len = strlen(y_desired_str);

  if (y_desired_len <= 0)
  {
    rb_raise(rb_eArgError, "invalid y length");
    return Qnil;
  }

  BACKPROP_FLOAT_T result = BackpropTrainer_TrainPair( trainer
                                                     , training_stats
                                                     , network
                                                     , x_str, x_len
                                                     , y_desired_str, y_desired_len);

  return rb_float_new(result);
}






static VALUE cBackpropTrainer_train_set( VALUE trainer_val
                                       , VALUE training_stats_val
                                       , VALUE network_val
                                       , VALUE training_set_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer = NULL;
  Data_Get_Struct(trainer_val, BackpropTrainer_t, trainer);

  if (!trainer)
  {
    rb_raise(rb_eArgError, "NULL trainer");
    return Qnil;
  }

  BackpropTrainingStats_t* training_stats = NULL;
  Data_Get_Struct(training_stats_val, BackpropTrainingStats_t, training_stats);

  if (!training_stats)
  {
    rb_raise(rb_eArgError, "NULL training stats");
    return Qnil;
  }

  BackpropNetwork_t* network = NULL;
  Data_Get_Struct(network_val, BackpropNetwork_t, network);

  if (!network)
  {
    rb_raise(rb_eArgError, "NULL network");
    return Qnil;
  }


  BackpropTrainingSet_t* training_set = NULL;
  Data_Get_Struct(training_set_val, BackpropTrainingSet_t, training_set);

  if (!training_set)
  {
    rb_raise(rb_eArgError, "NULL training set");
    return Qnil;
  }

  BACKPROP_FLOAT_T error = BackpropTrainer_TrainSet( trainer
                                                   , training_stats
                                                   , network
                                                   , training_set);

  return rb_float_new(error);
}






static VALUE cBackpropTrainer_train_batch( VALUE trainer_val
                                         , VALUE training_stats_val
                                         , VALUE exercise_stats_val
                                         , VALUE network_val
                                         , VALUE training_set_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer = NULL;
  Data_Get_Struct(trainer_val, BackpropTrainer_t, trainer);

  if (!trainer)
  {
    rb_raise(rb_eArgError, "NULL trainer");
    return Qnil;
  }

  BackpropTrainingStats_t* training_stats = NULL;
  Data_Get_Struct(training_stats_val, BackpropTrainingStats_t, training_stats);

  if (!training_stats)
  {
    rb_raise(rb_eArgError, "NULL training stats");
    return Qnil;
  }

  BackpropExerciseStats_t* exercise_stats = NULL;
  Data_Get_Struct(exercise_stats_val, BackpropExerciseStats_t, exercise_stats);

  if (!exercise_stats)
  {
    rb_raise(rb_eArgError, "NULL exercise stats");
    return Qnil;
  }

  BackpropNetwork_t* network = NULL;
  Data_Get_Struct(network_val, BackpropNetwork_t, network);

  if (!network)
  {
    rb_raise(rb_eArgError, "NULL network");
    return Qnil;
  }


  BackpropTrainingSet_t* training_set = NULL;
  Data_Get_Struct(training_set_val, BackpropTrainingSet_t, training_set);

  if (!training_set)
  {
    rb_raise(rb_eArgError, "NULL training set");
    return Qnil;
  }

  BACKPROP_FLOAT_T error = BackpropTrainer_TrainBatch( trainer
                                                     , training_stats
                                                     , exercise_stats
                                                     , network
                                                     , training_set);

  return rb_float_new(error);
}




static VALUE cBackpropTrainer_train( VALUE trainer_val
                                   , VALUE training_stats_val
                                   , VALUE exercise_stats_val
                                   , VALUE network_val
                                   , VALUE training_set_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer = NULL;
  Data_Get_Struct(trainer_val, BackpropTrainer_t, trainer);

  if (!trainer)
  {
    rb_raise(rb_eArgError, "NULL trainer");
    return Qnil;
  }

  BackpropTrainingStats_t* training_stats = NULL;
  Data_Get_Struct(training_stats_val, BackpropTrainingStats_t, training_stats);

  if (!training_stats)
  {
    rb_raise(rb_eArgError, "NULL training stats");
    return Qnil;
  }

  BackpropExerciseStats_t* exercise_stats = NULL;
  Data_Get_Struct(exercise_stats_val, BackpropExerciseStats_t, exercise_stats);

  if (!exercise_stats)
  {
    rb_raise(rb_eArgError, "NULL exercise stats");
    return Qnil;
  }

  BackpropNetwork_t* network = NULL;
  Data_Get_Struct(network_val, BackpropNetwork_t, network);

  if (!network)
  {
    rb_raise(rb_eArgError, "NULL network");
    return Qnil;
  }


  BackpropTrainingSet_t* training_set = NULL;
  Data_Get_Struct(training_set_val, BackpropTrainingSet_t, training_set);

  if (!training_set)
  {
    rb_raise(rb_eArgError, "NULL training set");
    return Qnil;
  }

  BACKPROP_FLOAT_T error = BackpropTrainer_Train( trainer
                                                , training_stats
                                                , exercise_stats
                                                , network
                                                , training_set);

  return rb_float_new(error);
}




static VALUE cBackpropTrainer_exercise( VALUE self_val
                                      , VALUE stats_val
                                      , VALUE network_val
                                      , VALUE training_set_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer = NULL;
  Data_Get_Struct(self_val, BackpropTrainer_t, trainer);

  if (!trainer)
  {
    rb_raise(rb_eArgError, "NULL self");
    return Qnil;
  }

  BackpropExerciseStats_t* stats = NULL;
  Data_Get_Struct(stats_val, BackpropExerciseStats_t, stats);

  if (!stats)
  {
    rb_raise(rb_eArgError, "NULL stats");
    return Qnil;
  }

  BackpropNetwork_t* network = NULL;
  Data_Get_Struct(network_val, BackpropNetwork_t, network);

  if (!network)
  {
    rb_raise(rb_eArgError, "NULL network");
    return Qnil;
  }


  BackpropTrainingSet_t* training_set = NULL;
  Data_Get_Struct(training_set_val, BackpropTrainingSet_t, training_set);

  if (!training_set)
  {
    rb_raise(rb_eArgError, "NULL training set");
    return Qnil;
  }


  BACKPROP_FLOAT_T error = 0.0;

  error = BackpropTrainer_Exercise(trainer, stats, network, training_set);

  return rb_float_new(error);
}




static VALUE cBackpropTrainer_get_error_tolerance(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer;
  Data_Get_Struct(self, BackpropTrainer_t, trainer);

  return rb_float_new(trainer->error_tolerance);
}




static VALUE cBackpropTrainer_get_learning_rate(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer;
  Data_Get_Struct(self, BackpropTrainer_t, trainer);

  return rb_float_new(trainer->learning_rate);
}




static VALUE cBackpropTrainer_get_mutation_rate(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer;
  Data_Get_Struct(self, BackpropTrainer_t, trainer);

  return rb_float_new(trainer->mutation_rate);
}




static VALUE cBackpropTrainer_get_momentum_rate(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer;
  Data_Get_Struct(self, BackpropTrainer_t, trainer);

  return rb_float_new(trainer->momentum_rate);
}




static VALUE cBackpropTrainer_get_max_reps(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer;
  Data_Get_Struct(self, BackpropTrainer_t, trainer);

  return INT2NUM(trainer->max_reps);
}




static VALUE cBackpropTrainer_get_max_batch_sets(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer;
  Data_Get_Struct(self, BackpropTrainer_t, trainer);

  return INT2NUM(trainer->max_batch_sets);
}




static VALUE cBackpropTrainer_set_max_batch_sets(VALUE self, VALUE value)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer;
  Data_Get_Struct(self, BackpropTrainer_t, trainer);

  trainer->max_batch_sets = NUM2INT(value);

  return self;
}




static VALUE cBackpropTrainer_get_max_batches(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer;
  Data_Get_Struct(self, BackpropTrainer_t, trainer);

  return INT2NUM(trainer->max_batches);
}




static VALUE cBackpropTrainer_set_max_batches(VALUE self, VALUE value)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer;
  Data_Get_Struct(self, BackpropTrainer_t, trainer);

  trainer->max_batches = NUM2INT(value);

  return self;
}




static VALUE cBackpropTrainer_get_stagnate_tolerance(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer;
  Data_Get_Struct(self, BackpropTrainer_t, trainer);

  return rb_float_new(trainer->stagnate_tolerance);
}




static VALUE cBackpropTrainer_get_max_stagnate_sets(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer;
  Data_Get_Struct(self, BackpropTrainer_t, trainer);

  return INT2NUM(trainer->max_stagnate_sets);
}




static VALUE cBackpropTrainer_get_max_stagnate_batches(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer;
  Data_Get_Struct(self, BackpropTrainer_t, trainer);

  return INT2NUM(trainer->max_stagnate_batches);
}




static VALUE cBackpropTrainer_get_min_set_weight_correction_limit(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer;
  Data_Get_Struct(self, BackpropTrainer_t, trainer);

  return rb_float_new(trainer->min_set_weight_correction_limit);
}




static VALUE cBackpropTrainer_get_min_batch_weight_correction_limit(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer;
  Data_Get_Struct(self, BackpropTrainer_t, trainer);

  return rb_float_new(trainer->min_batch_weight_correction_limit);
}




static VALUE cBackpropTrainer_get_batch_prune_threshold(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer;
  Data_Get_Struct(self, BackpropTrainer_t, trainer);

  return rb_float_new(trainer->batch_prune_threshold);
}




static VALUE cBackpropTrainer_get_training_ratio(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer;
  Data_Get_Struct(self, BackpropTrainer_t, trainer);

  return rb_float_new(trainer->training_ratio);
}





static VALUE cBackpropTrainer_get_batch_prune_rate(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer;
  Data_Get_Struct(self, BackpropTrainer_t, trainer);

  return rb_float_new(trainer->batch_prune_rate);
}




static VALUE cBackpropTrainer_set_batch_prune_rate(VALUE self, VALUE value)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer;
  Data_Get_Struct(self, BackpropTrainer_t, trainer);

  trainer->batch_prune_rate = NUM2DBL(value);

  return self;
}




static VALUE cBackpropTrainer_to_hash(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer;
  Data_Get_Struct(self, BackpropTrainer_t, trainer);

  VALUE hash = rb_hash_new();
  rb_hash_aset(hash, rb_str_new2("error_tolerance"), cBackpropTrainer_get_error_tolerance(self));
  rb_hash_aset(hash, rb_str_new2("learning_rate"), cBackpropTrainer_get_learning_rate(self));
  rb_hash_aset(hash, rb_str_new2("mutation_rate"), cBackpropTrainer_get_mutation_rate(self));
  rb_hash_aset(hash, rb_str_new2("momentum_rate"), cBackpropTrainer_get_momentum_rate(self));
  rb_hash_aset(hash, rb_str_new2("max_reps"), cBackpropTrainer_get_max_reps(self));
  rb_hash_aset(hash, rb_str_new2("max_batch_sets"), cBackpropTrainer_get_max_batch_sets(self));
  rb_hash_aset(hash, rb_str_new2("max_batches"), cBackpropTrainer_get_max_batches(self));
  rb_hash_aset(hash, rb_str_new2("stagnate_tolerance"), cBackpropTrainer_get_stagnate_tolerance(self));
  rb_hash_aset(hash, rb_str_new2("max_stagnate_sets"), cBackpropTrainer_get_max_stagnate_sets(self));
  rb_hash_aset(hash, rb_str_new2("max_stagnate_batches"), cBackpropTrainer_get_max_stagnate_batches(self));
  rb_hash_aset(hash, rb_str_new2("min_set_weight_correction_limit"), cBackpropTrainer_get_min_set_weight_correction_limit(self));
  rb_hash_aset(hash, rb_str_new2("min_batch_weight_correction_limit"), cBackpropTrainer_get_min_batch_weight_correction_limit(self));
  rb_hash_aset(hash, rb_str_new2("batch_prune_threshold"), cBackpropTrainer_get_batch_prune_threshold(self));
  rb_hash_aset(hash, rb_str_new2("batch_prune_rate"), cBackpropTrainer_get_batch_prune_rate(self));
  rb_hash_aset(hash, rb_str_new2("training_ratio"), cBackpropTrainer_get_training_ratio(self));


//  rb_hash_aset(hash, rb_str_new2("teach_total"), cBackpropTrainingStats_teach_total(self));
//  rb_hash_aset(hash, rb_str_new2("pair_total"), cBackpropTrainingStats_pair_total(self));
//  rb_hash_aset(hash, rb_str_new2("set_total"), cBackpropTrainingStats_set_total(self));
//  rb_hash_aset(hash, rb_str_new2("batches_total"), cBackpropTrainingStats_batches_total(self));
//  rb_hash_aset(hash, rb_str_new2("stubborn_batches_total"), cBackpropTrainingStats_stubborn_batches_total(self));
//  rb_hash_aset(hash, rb_str_new2("stagnate_batches_total"), cBackpropTrainingStats_stagnate_batches_total(self));
//  rb_hash_aset(hash, rb_str_new2("train_clock"), cBackpropTrainingStats_train_clock(self));

  return hash;
}




static VALUE cBackpropTrainer_initialize(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  return self;
}




static void cBackpropTrainer_Free(struct BackpropTrainer* trainer)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_Free(trainer);
}




static VALUE cBackpropTrainer_new(VALUE klass, VALUE network_value)
{
  BACKPROP_TRACE(__FUNCTION__);

  // get the network
  BackpropNetwork_t* network;
  Data_Get_Struct(network_value, BackpropNetwork_t, network);

  // allocate a trainer
  BackpropTrainer_t* trainer = BackpropTrainer_Malloc(network);

  BackpropTrainer_SetToDefault(trainer);
  BackpropTrainer_SetToDefaultIO(trainer);

  // wrap it in a ruby object, this will cause GC to call free function
  VALUE tdata = Data_Wrap_Struct(klass, 0, cBackpropTrainer_Free, trainer);

  // call initialize
  VALUE argv[1] = {network_value};
  rb_obj_call_init(tdata, 1, argv);

  return tdata;
}




//------------------------------------------------------------------------------
//
// BackpropEvolutionStats
//
//------------------------------------------------------------------------------


static VALUE cBackpropEvolutionStats_get_generation_count(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropEvolutionStats_t* stats;
  Data_Get_Struct(self, BackpropEvolutionStats_t, stats);

  return INT2NUM(stats->generation_count);
}




static VALUE cBackpropEvolutionStats_get_mate_networks_count(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropEvolutionStats_t* stats;
  Data_Get_Struct(self, BackpropEvolutionStats_t, stats);

  return INT2NUM(stats->mate_networks_count);
}




static VALUE cBackpropEvolutionStats_get_evolve_clock(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropEvolutionStats_t* stats;
  Data_Get_Struct(self, BackpropEvolutionStats_t, stats);

  return INT2NUM(stats->evolve_clock);
}




static VALUE cBackpropEvolutionStats_to_hash(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  VALUE hash = rb_hash_new();

  rb_hash_aset(hash, rb_str_new2("generation_count"), cBackpropEvolutionStats_get_generation_count(self));
  rb_hash_aset(hash, rb_str_new2("mate_networks_count"), cBackpropEvolutionStats_get_mate_networks_count(self));
  rb_hash_aset(hash, rb_str_new2("evolve_clock"), cBackpropEvolutionStats_get_evolve_clock(self));

  return hash;
}




static VALUE cBackpropEvolutionStats_new(VALUE klass)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropEvolutionStats_t* obj = xmalloc(sizeof(BackpropEvolutionStats_t));

  // wrap it in a ruby object, this will cause GC to call free function
  VALUE tdata = Data_Wrap_Struct(klass, 0, xfree, obj);

  // call initialize
  //rb_obj_call_init(tdata, 0, 0);

  return tdata;
}



//------------------------------------------------------------------------------
//
// BackpropEvolutionStats
//
//------------------------------------------------------------------------------


static VALUE cBackpropEvolver_get_pool_count(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropEvolver_t* obj;
  Data_Get_Struct(self, BackpropEvolver_t, obj);

  return INT2NUM(obj->pool_count);
}




static VALUE cBackpropEvolver_get_max_generations(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropEvolver_t* obj;
  Data_Get_Struct(self, BackpropEvolver_t, obj);

  return INT2NUM(obj->max_generations);
}




static VALUE cBackpropEvolver_get_mate_rate(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropEvolver_t* obj;
  Data_Get_Struct(self, BackpropEvolver_t, obj);

  return rb_float_new(obj->mate_rate);
}




static VALUE cBackpropEvolver_get_mutation_limit(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropEvolver_t* obj;
  Data_Get_Struct(self, BackpropEvolver_t, obj);

  return rb_float_new(obj->mutation_limit);
}




static VALUE cBackpropEvolver_get_seed(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropEvolver_t* obj;
  Data_Get_Struct(self, BackpropEvolver_t, obj);

  return rb_float_new(obj->mutation_limit);
}




static VALUE cBackpropEvolver_to_hash(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  VALUE hash = rb_hash_new();

  rb_hash_aset(hash, rb_str_new2("pool_count"), cBackpropEvolver_get_pool_count(self));
  rb_hash_aset(hash, rb_str_new2("max_generations"), cBackpropEvolver_get_max_generations(self));
  rb_hash_aset(hash, rb_str_new2("mate_rate"), cBackpropEvolver_get_mate_rate(self));
  rb_hash_aset(hash, rb_str_new2("mutation_limit"), cBackpropEvolver_get_mutation_limit(self));
  rb_hash_aset(hash, rb_str_new2("seed"), cBackpropEvolver_get_seed(self));

  return hash;
}




static VALUE cBackpropEvolver_set_to_default(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropEvolver_t* obj;
  Data_Get_Struct(self, BackpropEvolver_t, obj);

  BackpropEvolver_SetToDefault(obj);

  return self;
}






static VALUE cBackpropEvolver_evolve( VALUE evolver_val
                                    , VALUE evolution_stats_val
                                    , VALUE trainer_val
                                    , VALUE training_stats_val
                                    , VALUE exercise_stats_val
                                    , VALUE network_val
                                    , VALUE training_set_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  VALUE_TO_C_PTR(BackpropEvolver_t, evolver, evolver_val);
  VALUE_TO_C_PTR(BackpropEvolutionStats_t, evolution_stats, evolution_stats_val);
  VALUE_TO_C_PTR(BackpropTrainer_t, trainer, trainer_val);
  VALUE_TO_C_PTR(BackpropTrainingStats_t, training_stats, training_stats_val);
  VALUE_TO_C_PTR(BackpropExerciseStats_t, exercise_stats, exercise_stats_val);
  VALUE_TO_C_PTR(BackpropNetwork_t, network, network_val);
  VALUE_TO_C_PTR(BackpropTrainingSet_t, training_set, training_set_val);

  BACKPROP_FLOAT_T result = BackpropEvolver_Evolve( evolver
                                                  , evolution_stats
                                                  , trainer
                                                  , training_stats
                                                  , exercise_stats
                                                  , network
                                                  , training_set);

  return rb_float_new(result);
}




static VALUE cBackpropEvolver_new(VALUE klass)
{
  BACKPROP_TRACE(__FUNCTION__);

  struct BackpropEvolver* obj = xmalloc(sizeof(struct BackpropEvolver));

  // wrap it in a ruby object, this will cause GC to call free function
  VALUE tdata = Data_Wrap_Struct(klass, 0, xfree, obj);

  // call initialize
  //rb_obj_call_init(tdata, 0, 0);

  return tdata;
}




//------------------------------------------------------------------------------
//
// module initialization
//
//------------------------------------------------------------------------------

// The initialization method for this module
void Init_cbackproprb()
{
  BACKPROP_TRACE(__FUNCTION__);
  // configure Backprop libary
  //Backprop_SetMalloc(cBackprop_Malloc);
  //Backprop_SetFree(cBackprop_Free);


  // Define module methods
  cBackproprb = rb_define_module("CBackproprb");
  rb_define_module_function(cBackproprb, "used", cBackprop_used, 0);
  rb_define_module_function(cBackproprb, "sigmoid", cBackprop_sigmoid, 1);
  rb_define_module_function(cBackproprb, "uniform_random_int", cBackprop_uniform_random_int, 0);


  // Define class CBackproprb::CBackpropLayer
  cBackpropLayer = rb_define_class_under(cBackproprb, "CLayer", rb_cObject);
  rb_define_singleton_method(cBackpropLayer, "new", cBackpropLayer_new, 2);
  rb_define_method(cBackpropLayer, "initialize", cBackpropLayer_initialize, 2);
  rb_define_method(cBackpropLayer, "w", cBackpropLayer_get_W, 0);
  rb_define_method(cBackpropLayer, "w=", cBackpropLayer_set_W, 1);
  rb_define_method(cBackpropLayer, "w_count", cBackpropLayer_get_W_count, 0);
  rb_define_method(cBackpropLayer, "w_sum", cBackpropLayer_get_W_sum, 0);
  rb_define_method(cBackpropLayer, "w_mean", cBackpropLayer_get_W_mean, 0);
  rb_define_method(cBackpropLayer, "w_stddev", cBackpropLayer_get_W_stddev, 0);
  rb_define_method(cBackpropLayer, "g", cBackpropLayer_get_g, 0);
  rb_define_method(cBackpropLayer, "g=", cBackpropLayer_set_g, 1);
  rb_define_method(cBackpropLayer, "x", cBackpropLayer_get_x, 0);
  rb_define_method(cBackpropLayer, "x=", cBackpropLayer_set_x, 1);
  rb_define_method(cBackpropLayer, "x_count", cBackpropLayer_get_x_count, 0);
  rb_define_method(cBackpropLayer, "y", cBackpropLayer_get_y, 0);
  rb_define_method(cBackpropLayer, "y=", cBackpropLayer_set_y, 1);
  rb_define_method(cBackpropLayer, "y_count", cBackpropLayer_get_y_count, 0);
  rb_define_method(cBackpropLayer, "randomize", cBackpropLayer_randomize, 1);
  rb_define_method(cBackpropLayer, "identity", cBackpropLayer_identity, 0);
  rb_define_method(cBackpropLayer, "reset", cBackpropLayer_reset, 0);
  rb_define_method(cBackpropLayer, "prune", cBackpropLayer_prune, 1);
  rb_define_method(cBackpropLayer, "activate", cBackpropLayer_activate, 0);
  rb_define_method(cBackpropLayer, "to_hash", cBackpropLayer_to_hash, 0);


  // Define class CBackproprb::CNetwork
  cBackpropNetwork = rb_define_class_under(cBackproprb, "CNetwork", rb_cObject);
  rb_define_singleton_method(cBackpropNetwork, "new", cBackpropNetwork_new, 3);
  rb_define_method(cBackpropNetwork, "initialize", cBackpropNetwork_initialize, 3);
  rb_define_method(cBackpropNetwork, "activate", cBackpropNetwork_activate, 1);
  rb_define_method(cBackpropNetwork, "x", cBackpropNetwork_get_x, 0);
  rb_define_method(cBackpropNetwork, "x_size", cBackpropNetwork_x_size, 0);
  rb_define_method(cBackpropNetwork, "y", cBackpropNetwork_get_y, 0);
  rb_define_method(cBackpropNetwork, "y_size", cBackpropNetwork_y_size, 0);
  rb_define_method(cBackpropNetwork, "layers_count", cBackpropNetwork_layers_count, 0);
  rb_define_method(cBackpropNetwork, "layer_get", cBackpropNetwork_get_layer, 1);

  rb_define_method(cBackpropNetwork, "jitter", cBackpropNetwork_get_jitter, 0);
  rb_define_method(cBackpropNetwork, "jitter=", cBackpropNetwork_set_jitter, 1);
  rb_define_method(cBackpropNetwork, "randomize", cBackpropNetwork_randomize, 1);
  rb_define_method(cBackpropNetwork, "identity", cBackpropNetwork_identity, 0);
  rb_define_method(cBackpropNetwork, "reset", cBackpropNetwork_reset, 0);
  rb_define_method(cBackpropNetwork, "prune", cBackpropNetwork_prune, 1);
  rb_define_method(cBackpropNetwork, "stats", cBackpropNetwork_get_stats, 0);
  rb_define_method(cBackpropNetwork, "to_hash", cBackpropNetwork_to_hash, 0);
  rb_define_method(cBackpropNetwork, "to_file", cBackpropNetwork_to_file, 1);
  rb_define_method(cBackpropNetwork, "from_file", cBackpropNetwork_from_file, 1);

  // Define class CBackproprb::CNetworkStats
  cBackpropNetworkStats = rb_define_class_under(cBackproprb, "CNetworkStats", rb_cObject);
  rb_define_singleton_method(cBackpropNetworkStats, "new", CBackpropNetworkStats_new, 0);

  rb_define_method(cBackpropNetworkStats, "x_size", CBackpropNetworkStats_get_x_size, 0);
  rb_define_method(cBackpropNetworkStats, "y_size", CBackpropNetworkStats_get_y_size, 0);
  rb_define_method(cBackpropNetworkStats, "layers_count", CBackpropNetworkStats_get_layers_count, 0);
  rb_define_method(cBackpropNetworkStats, "layers_size", CBackpropNetworkStats_get_layers_size, 0);
  rb_define_method(cBackpropNetworkStats, "layers_w_count", CBackpropNetworkStats_get_layers_w_count, 0);
  rb_define_method(cBackpropNetworkStats, "layers_w_size", CBackpropNetworkStats_get_layers_w_size, 0);
  rb_define_method(cBackpropNetworkStats, "layers_w_avg", CBackpropNetworkStats_get_layers_w_avg, 0);
  rb_define_method(cBackpropNetworkStats, "layers_w_stddev", CBackpropNetworkStats_get_layers_w_stddev, 0);

  rb_define_method(cBackpropNetworkStats, "x_size=", CBackpropNetworkStats_set_x_size, 0);
  rb_define_method(cBackpropNetworkStats, "y_size=", CBackpropNetworkStats_set_y_size, 0);
  rb_define_method(cBackpropNetworkStats, "layers_count=", CBackpropNetworkStats_set_layers_count, 0);
  rb_define_method(cBackpropNetworkStats, "layers_size=", CBackpropNetworkStats_set_layers_size, 0);
  rb_define_method(cBackpropNetworkStats, "layers_w_count=", CBackpropNetworkStats_set_layers_w_count, 0);
  rb_define_method(cBackpropNetworkStats, "layers_w_size=", CBackpropNetworkStats_set_layers_w_size, 0);
  rb_define_method(cBackpropNetworkStats, "layers_w_avg=", CBackpropNetworkStats_set_layers_w_avg, 0);
  rb_define_method(cBackpropNetworkStats, "layers_w_stddev=", CBackpropNetworkStats_set_layers_w_stddev, 0);

  rb_define_method(cBackpropNetworkStats, "to_hash", CBackpropNetworkStats_to_hash, 0);


  // Define class CBackproprb::CTrainingSet
  cBackpropTrainingSet = rb_define_class_under(cBackproprb, "CTrainingSet", rb_cObject);
  rb_define_singleton_method(cBackpropTrainingSet, "new", cBackpropTrainingSet_new, 2);
  rb_define_method(cBackpropTrainingSet, "initialize", cBackpropTrainingSet_initialize, 2);
  rb_define_method(cBackpropTrainingSet, "count", cBackpropTrainingSet_count, 0);
  rb_define_method(cBackpropTrainingSet, "x_size", cBackpropTrainingSet_x_size, 0);
  rb_define_method(cBackpropTrainingSet, "y_size", cBackpropTrainingSet_y_size, 0);
  rb_define_method(cBackpropTrainingSet, "x_at", cBackpropTrainingSet_x_at, 1);
  rb_define_method(cBackpropTrainingSet, "y_at", cBackpropTrainingSet_y_at, 1);

  rb_define_method(cBackpropTrainingSet, "to_hash", cBackpropTrainingSet_to_hash, 0);

  // Define class CBackproprb::CExerciseStats
  cBackpropExerciseStats = rb_define_class_under(cBackproprb, "CExerciseStats", rb_cObject);
  rb_define_singleton_method(cBackpropExerciseStats, "new", cBackpropExerciseStats_new, 0);
  rb_define_method(cBackpropExerciseStats, "exercise_clock_ticks", cBackpropExerciseStats_exercise_clock_ticks, 0);
  rb_define_method(cBackpropExerciseStats, "activate_count", cBackpropExerciseStats_activate_count, 0);
  rb_define_method(cBackpropExerciseStats, "error", cBackpropExerciseStats_error, 0);
  rb_define_method(cBackpropExerciseStats, "to_hash", cBackpropExerciseStats_to_hash, 0);


  // Define class CBackproprb::CTrainingStats
  cBackpropTrainingStats = rb_define_class_under(cBackproprb, "CTrainingStats", rb_cObject);
  rb_define_singleton_method(cBackpropTrainingStats, "new", cBackpropTrainingStats_new, 0);
  rb_define_method(cBackpropTrainingStats, "set_weight_correction_total", cBackpropTrainingStats_set_weight_correction_total, 0);
  rb_define_method(cBackpropTrainingStats, "batch_weight_correction_total", cBackpropTrainingStats_batch_weight_correction_total, 0);
  rb_define_method(cBackpropTrainingStats, "teach_total", cBackpropTrainingStats_teach_total, 0);
  rb_define_method(cBackpropTrainingStats, "pair_total", cBackpropTrainingStats_pair_total, 0);
  rb_define_method(cBackpropTrainingStats, "set_total", cBackpropTrainingStats_set_total, 0);
  rb_define_method(cBackpropTrainingStats, "batches_total", cBackpropTrainingStats_batches_total, 0);
  rb_define_method(cBackpropTrainingStats, "stubborn_batches_total", cBackpropTrainingStats_stubborn_batches_total, 0);
  rb_define_method(cBackpropTrainingStats, "stagnate_batches_total", cBackpropTrainingStats_stagnate_batches_total, 0);
  rb_define_method(cBackpropTrainingStats, "train_clock", cBackpropTrainingStats_train_clock, 0);
  rb_define_method(cBackpropTrainingStats, "to_hash", cBackpropTrainingStats_to_hash, 0);

  // Define class CBackprop::CTrainer
  cBackpropTrainer = rb_define_class_under(cBackproprb, "CTrainer", rb_cObject);
  rb_define_singleton_method(cBackpropTrainer, "new", cBackpropTrainer_new, 1);
  rb_define_method(cBackpropTrainer, "initialize", cBackpropTrainer_initialize, 1);

  rb_define_method(cBackpropTrainer, "error_tolerance", cBackpropTrainer_get_error_tolerance, 0);
  rb_define_method(cBackpropTrainer, "learning_rate", cBackpropTrainer_get_learning_rate, 0);
  rb_define_method(cBackpropTrainer, "mutation_rate", cBackpropTrainer_get_mutation_rate, 0);
  rb_define_method(cBackpropTrainer, "momentum_rate", cBackpropTrainer_get_momentum_rate, 0);
  rb_define_method(cBackpropTrainer, "max_reps", cBackpropTrainer_get_max_reps, 0);
  rb_define_method(cBackpropTrainer, "max_batch_sets", cBackpropTrainer_get_max_batch_sets, 0);
  rb_define_method(cBackpropTrainer, "max_batches", cBackpropTrainer_get_max_batches, 0);
  rb_define_method(cBackpropTrainer, "stagnate_tolerance", cBackpropTrainer_get_stagnate_tolerance, 0);
  rb_define_method(cBackpropTrainer, "max_stagnate_sets", cBackpropTrainer_get_max_stagnate_sets, 0);
  rb_define_method(cBackpropTrainer, "max_stagnate_batches", cBackpropTrainer_get_max_stagnate_batches, 0);
  rb_define_method(cBackpropTrainer, "min_set_weight_correction_limit", cBackpropTrainer_get_min_set_weight_correction_limit, 0);
  rb_define_method(cBackpropTrainer, "min_batch_weight_correction_limit", cBackpropTrainer_get_min_batch_weight_correction_limit, 0);
  rb_define_method(cBackpropTrainer, "batch_prune_threshold", cBackpropTrainer_get_batch_prune_threshold, 0);
  rb_define_method(cBackpropTrainer, "batch_prune_rate", cBackpropTrainer_get_batch_prune_rate, 0);
  rb_define_method(cBackpropTrainer, "training_ratio", cBackpropTrainer_get_training_ratio, 0);

  rb_define_method(cBackpropTrainer, "max_batch_sets=", cBackpropTrainer_set_max_batch_sets, 1);
  rb_define_method(cBackpropTrainer, "max_batches=", cBackpropTrainer_set_max_batches, 1);
  rb_define_method(cBackpropTrainer, "batch_prune_rate=", cBackpropTrainer_set_batch_prune_rate, 1);

  rb_define_method(cBackpropTrainer, "exercise", cBackpropTrainer_exercise, 3);
  rb_define_method(cBackpropTrainer, "teach_pair", cBackpropTrainer_teach_pair, 4);
  rb_define_method(cBackpropTrainer, "train_pair", cBackpropTrainer_train_pair, 4);
  rb_define_method(cBackpropTrainer, "train_set", cBackpropTrainer_train_set, 3);
  rb_define_method(cBackpropTrainer, "train_batch", cBackpropTrainer_train_batch, 4);
  rb_define_method(cBackpropTrainer, "train", cBackpropTrainer_train, 4);

  rb_define_method(cBackpropTrainer, "to_hash", cBackpropTrainer_to_hash, 0);


  cBackpropEvolutionStats = rb_define_class_under(cBackproprb, "CEvolutionStats", rb_cObject);
  rb_define_singleton_method(cBackpropEvolutionStats, "new", cBackpropEvolutionStats_new, 0);
  rb_define_method(cBackpropEvolutionStats, "generation_count", cBackpropEvolutionStats_get_generation_count, 0);
  rb_define_method(cBackpropEvolutionStats, "mate_networks_count", cBackpropEvolutionStats_get_mate_networks_count, 0);
  rb_define_method(cBackpropEvolutionStats, "evolve_clock", cBackpropEvolutionStats_get_evolve_clock, 0);
  rb_define_method(cBackpropEvolutionStats, "to_hash", cBackpropEvolutionStats_to_hash, 0);


  cBackpropEvolver = rb_define_class_under(cBackproprb, "CEvolver", rb_cObject);
  rb_define_singleton_method(cBackpropEvolver, "new", cBackpropEvolver_new, 0);
  rb_define_method(cBackpropEvolver, "pool_count", cBackpropEvolver_get_pool_count, 0);
  rb_define_method(cBackpropEvolver, "max_generations", cBackpropEvolver_get_max_generations, 0);
  rb_define_method(cBackpropEvolver, "mate_rate", cBackpropEvolver_get_mate_rate, 0);
  rb_define_method(cBackpropEvolver, "mutation_limit", cBackpropEvolver_get_mutation_limit, 0);
  rb_define_method(cBackpropEvolver, "seed", cBackpropEvolver_get_seed, 0);
  rb_define_method(cBackpropEvolver, "to_hash", cBackpropEvolver_to_hash, 0);

  rb_define_method(cBackpropEvolver, "set_to_default", cBackpropEvolver_set_to_default, 0);
  rb_define_method(cBackpropEvolver, "evolve", cBackpropEvolver_evolve, 6);
}

