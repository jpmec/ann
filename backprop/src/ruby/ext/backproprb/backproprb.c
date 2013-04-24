// Include the Ruby headers and goodies
#include "ruby.h"
#include "backprop.h"
#include "backprop_io.h"




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




static void* CBackprop_Malloc (size_t size)
{
  BACKPROP_TRACE(__FUNCTION__);
  return xmalloc(size);
}




static void CBackprop_Free(void* obj)
{
  BACKPROP_TRACE(__FUNCTION__);
  xfree(obj);
}




VALUE CBackprop_sigmoid(VALUE self, VALUE x)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    const BACKPROP_FLOAT_T y = Backprop_Sigmoid(NUM2DBL(x));
    return rb_float_new(y);
  }
}




VALUE CBackprop_uniform_random_int(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    const int i = Backprop_UniformRandomInt();
    return INT2NUM(i);
  }
}




VALUE CBackprop_used(void)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    const size_t in_use = Backprop_GetMallocInUse();
    return INT2NUM(in_use);
  }
}





//------------------------------------------------------------------------------
//
// BackpropLayer
//
//------------------------------------------------------------------------------


static VALUE CBackpropLayer_get_W(VALUE self)
{
  BackpropLayer_t* layer = NULL;

  Data_Get_Struct(self, BackpropLayer_t, layer);
  if (!layer)
  {
    return Qnil;
  }

  else
  {
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
}




static VALUE CBackpropLayer_set_W(VALUE self, VALUE vals)
{
  const ID f = rb_intern("length");
  VALUE length_val = rb_funcall(vals, f, 0, 0);
  size_t length = NUM2INT(length_val);

  BackpropLayer_t* layer = NULL;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  if (!layer)
  {
    return Qnil;
  }
  else
  {
    const BACKPROP_SIZE_T W_count = BackpropLayer_GetWeightsCount(layer);

    const long end = (W_count < length) ? W_count : length;

    for (long i = 0; i < end; ++i)
    {
      VALUE val = rb_ary_entry(vals, i);
      layer->W[i] = NUM2DBL(val);
    }

    return self;
  }
}




static VALUE CBackpropLayer_get_g(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  if (!layer)
  {
    return Qnil;
  }
  else
  {
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
}




static VALUE CBackpropLayer_set_g(VALUE self, VALUE vals)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  if (!layer)
  {
    return Qnil;
  }

  else
  {
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
}




static VALUE CBackpropLayer_get_x(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  if (!self)
  {
    return Qnil;
  }

  else
  {
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
}




static VALUE CBackpropLayer_set_x(VALUE self, VALUE vals)
{
  const ID f = rb_intern("length");
  VALUE length_val = rb_funcall(vals, f, 0, 0);
  size_t length = NUM2INT(length_val);

  BackpropLayer_t* layer = NULL;
  Data_Get_Struct(self, BackpropLayer_t, layer);
  if (!layer)
  {
    return Qnil;
  }
  else
  {
    const long end = (layer->x_count < length) ? layer->x_count : length;

    for (long i = 0; i < end; ++i)
    {
      VALUE val = rb_ary_entry(vals, i);
      layer->x[i] = NUM2DBL(val);
    }

    return self;
  }
}




static VALUE CBackpropLayer_get_y(VALUE self)
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




static VALUE CBackpropLayer_set_y(VALUE self, VALUE vals)
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




static VALUE CBackpropLayer_get_W_count(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  const BACKPROP_SIZE_T count = BackpropLayer_GetWeightsCount(layer);

  return INT2NUM(count);
}




static VALUE CBackpropLayer_get_W_sum(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  const BACKPROP_FLOAT_T sum = BackpropLayer_GetWeightsSum(layer);

  return rb_float_new(sum);
}




static VALUE CBackpropLayer_get_W_mean(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  const BACKPROP_FLOAT_T mean = BackpropLayer_GetWeightsMean(layer);

  return rb_float_new(mean);
}




static VALUE CBackpropLayer_get_W_stddev(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  const BACKPROP_FLOAT_T value = BackpropLayer_GetWeightsStdDev(layer);

  return rb_float_new(value);
}




static VALUE CBackpropLayer_get_x_count(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  return INT2NUM(layer->x_count);
}




static VALUE CBackpropLayer_get_y_count(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  return INT2NUM(layer->y_count);
}




static VALUE CBackpropLayer_to_hash(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  VALUE hash = rb_hash_new();
  rb_hash_aset(hash, rb_str_new2("w"), CBackpropLayer_get_W(self));
  rb_hash_aset(hash, rb_str_new2("x"), CBackpropLayer_get_x(self));
  rb_hash_aset(hash, rb_str_new2("y"), CBackpropLayer_get_y(self));

  return hash;
}




static VALUE CBackpropLayer_randomize(VALUE self, VALUE gain_val)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  BACKPROP_FLOAT_T gain = NUM2DBL(gain_val);

  BackpropLayer_Randomize(layer, gain);

  return self;
}




static VALUE CBackpropLayer_identity(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  BackpropLayer_Identity(layer);

  return self;
}




static VALUE CBackpropLayer_reset(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  BackpropLayer_Reset(layer);

  return self;
}




static VALUE CBackpropLayer_prune(VALUE self, VALUE threshold_val)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  BACKPROP_FLOAT_T threshold = NUM2DBL(threshold_val);

  BackpropLayer_Prune(layer, threshold);

  return self;
}




static VALUE CBackpropLayer_activate(VALUE self)
{
  BackpropLayer_t* layer;
  Data_Get_Struct(self, BackpropLayer_t, layer);

  BackpropLayer_Activate(layer);

  return self;
}




static void CBackpropLayer_Free(struct BackpropLayer* layer)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropLayer_Free(layer);
}




static VALUE CBackpropLayer_initialize(VALUE self, VALUE x_size, VALUE y_size)
{
  BACKPROP_TRACE(__FUNCTION__);

  return self;
}




static VALUE CBackpropLayer_new(VALUE klass, VALUE x_count_val, VALUE y_count_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BACKPROP_SIZE_T x_count = NUM2UINT(x_count_val);
  BACKPROP_SIZE_T  y_count = NUM2UINT(y_count_val);

  // allocate structure
  struct BackpropLayer* layer = BackpropLayer_Malloc(x_count, y_count);

  // wrap it in a ruby object, this will cause GC to call free function
  VALUE tdata = Data_Wrap_Struct(klass, 0, CBackpropLayer_Free, layer);

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


static VALUE CBackpropNetwork_activate(VALUE self, VALUE input)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  char* cstr_in = StringValueCStr(input);

  BackpropNetwork_InputCStr(network, cstr_in);

  BackpropNetwork_Activate(network);

  const size_t len = BackpropNetwork_GetYSize(network) + 1;
  char* cstr_out = malloc(len);
  memset(cstr_out, 0, len);

  BackpropNetwork_GetOutputCStr(network, cstr_out, len);

  printf("output = %s\n", cstr_out);

  VALUE return_value = rb_str_new2(cstr_out);

  free(cstr_out);

  return return_value;
}




static VALUE CBackpropNetwork_x_size(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  BACKPROP_SIZE_T x_size = BackpropNetwork_GetXSize(network);

  return INT2NUM(x_size);
}




static VALUE CBackpropNetwork_get_x(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  const size_t x_size = BackpropNetwork_GetXSize(network);

  VALUE x_array = rb_ary_new2(x_size);

  for (size_t i = 0; i < x_size; ++i)
  {
    const BackpropByteArray_t* x = BackpropNetwork_GetX(network);
    rb_ary_store(x_array, i, INT2NUM(x->data[i]));
  }

  return x_array;
}




static VALUE CBackpropNetwork_y_size(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  const BACKPROP_SIZE_T y_size = BackpropNetwork_GetYSize(network);

  return INT2NUM(y_size);
}




static VALUE CBackpropNetwork_get_y(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  const size_t y_size = BackpropNetwork_GetYSize(network);

  VALUE y_array = rb_ary_new2(y_size);

  for (size_t i = 0; i < y_size; ++i)
  {
    const BackpropByteArray_t* y = BackpropNetwork_GetY(network);
    rb_ary_store(y_array, i, INT2NUM(y->data[i]));
  }

  return y_array;
}




static VALUE CBackpropNetwork_layers_count(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  const struct BackpropLayersArray* layers = BackpropNetwork_GetLayers(network);

  const BACKPROP_SIZE_T layers_count = layers->count;

  return INT2NUM(layers_count);
}



static VALUE CBackpropNetwork_get_layer(VALUE self, VALUE index_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  int index = NUM2INT(index_val);

  const struct BackpropLayersArray* layers = BackpropNetwork_GetLayers(network);
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


static VALUE CBackpropNetwork_get_jitter(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  BACKPROP_FLOAT_T jitter = BackpropNetwork_GetJitter(network);

  return rb_float_new(jitter);
}




static VALUE CBackpropNetwork_set_jitter(VALUE self, VALUE value)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  BACKPROP_FLOAT_T jitter = NUM2DBL(value);

  BackpropNetwork_SetJitter(network, jitter);

  return self;
}




static VALUE CBackpropNetwork_randomize(VALUE self, VALUE seed_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  unsigned int seed = NUM2UINT(seed_val);

  BackpropNetwork_Randomize(network, seed);

  return self;
}




static VALUE CBackpropNetwork_identity(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  BackpropNetwork_Identity(network);

  return self;
}




static VALUE CBackpropNetwork_reset(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  BackpropNetwork_Reset(network);

  return self;
}




static VALUE CBackpropNetwork_prune(VALUE self, VALUE threshold_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  BACKPROP_FLOAT_T threshold = NUM2DBL(threshold_val);

  BackpropNetwork_Prune(network, threshold);

  return self;
}




static VALUE CBackpropNetwork_get_stats(VALUE self)
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




static VALUE CBackpropNetwork_to_hash(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_t* network;
  Data_Get_Struct(self, BackpropNetwork_t, network);

  VALUE x_array = CBackpropNetwork_get_x(self);
  VALUE y_array = CBackpropNetwork_get_y(self);

  const BACKPROP_SIZE_T layers_count = BackpropNetwork_GetLayersCount(network);

  VALUE layers_array = rb_ary_new2(layers_count);

  for (BACKPROP_SIZE_T i = 0; i < layers_count; ++i)
  {
    VALUE layer = CBackpropNetwork_get_layer(self, INT2NUM(i));
    rb_ary_store(layers_array, i, CBackpropLayer_to_hash(layer));
  }

  VALUE hash = rb_hash_new();
  rb_hash_aset(hash, rb_str_new2("layers"), layers_array);
  rb_hash_aset(hash, rb_str_new2("x"), x_array);
  rb_hash_aset(hash, rb_str_new2("y"), y_array);
  rb_hash_aset(hash, rb_str_new2("jitter"), CBackpropNetwork_get_jitter(self));

  return hash;
}




static VALUE CBackpropNetwork_to_file(VALUE self, VALUE file_name_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  VALUE_TO_C_PTR(BackpropNetwork_t, network, self);

  const char* file_name = StringValueCStr(file_name_val);

  BackpropNetwork_SaveWeights(network, file_name);

  return self;
}




static VALUE CBackpropNetwork_from_file(VALUE self, VALUE file_name_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  VALUE_TO_C_PTR(BackpropNetwork_t, network, self);

  const char* file_name = StringValueCStr(file_name_val);

  BackpropNetwork_LoadWeights(network, file_name);

  return self;
}




static VALUE CBackpropNetwork_initialize(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  CBackpropNetwork_reset(self);

  return self;
}




static void CBackpropNetwork_free(struct BackpropNetwork* network)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropNetwork_Free(network);
}




static VALUE CBackpropNetwork_new(VALUE klass, VALUE args)
{
  BACKPROP_TRACE(__FUNCTION__);

  VALUE x_size_val = rb_hash_aref(args, rb_str_new2("x_size"));
  VALUE y_size_val = rb_hash_aref(args, rb_str_new2("y_size"));
  VALUE layer_count_val = rb_hash_aref(args, rb_str_new2("layer_count"));

  BACKPROP_SIZE_T x_size = NUM2INT(x_size_val);
  BACKPROP_SIZE_T y_size = NUM2INT(y_size_val);
  BACKPROP_SIZE_T layer_count = NUM2INT(layer_count_val);

  // allocate structure
  struct BackpropNetwork* network = BackpropNetwork_Malloc( x_size
                                                          , y_size
                                                          , layer_count
                                                          , true);

  // wrap it in a ruby object, this will cause GC to call free function
  VALUE tdata = Data_Wrap_Struct(klass, 0, CBackpropNetwork_free, network);

  // call initialize
  VALUE argv[1] = {args};
  rb_obj_call_init(tdata, 1, argv);

  return tdata;
}








//------------------------------------------------------------------------------
//
// BackpropTrainingSet
//
//------------------------------------------------------------------------------


static VALUE CBackpropTrainingSet_x_at(VALUE self, VALUE index)
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




static VALUE CBackpropTrainingSet_y_at(VALUE self, VALUE index)
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




static VALUE CBackpropTrainingSet_count(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingSet_t* training_set;
  Data_Get_Struct(self, BackpropTrainingSet_t, training_set);

  return INT2NUM(training_set->dims.count);
}




static VALUE CBackpropTrainingSet_x_size(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingSet_t* training_set;
  Data_Get_Struct(self, BackpropTrainingSet_t, training_set);

  return INT2NUM(training_set->dims.x_size);
}




static VALUE CBackpropTrainingSet_y_size(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingSet_t* training_set;
  Data_Get_Struct(self, BackpropTrainingSet_t, training_set);

  return INT2NUM(training_set->dims.y_size);
}




static VALUE CBackpropTrainingSet_to_hash(VALUE self)
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
    rb_ary_store(x_array, i, CBackpropTrainingSet_x_at(self, INT2NUM(i)));
    rb_ary_store(y_array, i, CBackpropTrainingSet_y_at(self, INT2NUM(i)));
  }

  rb_hash_aset(hash, rb_str_new2("count"), INT2NUM(count));
  rb_hash_aset(hash, rb_str_new2("x_size"), INT2NUM(training_set->dims.x_size));
  rb_hash_aset(hash, rb_str_new2("y_size"), INT2NUM(training_set->dims.y_size));
  rb_hash_aset(hash, rb_str_new2("x"), x_array);
  rb_hash_aset(hash, rb_str_new2("y"), y_array);

  return hash;
}




static VALUE CBackpropTrainingSet_to_file(VALUE self, VALUE file_name_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  VALUE_TO_C_PTR(BackpropTrainingSet_t, training_set, self);

  const char* file_name = StringValueCStr(file_name_val);

  BackpropTrainingSet_Save(training_set, file_name);

  return self;
}




static VALUE CBackpropTrainingSet_from_file(VALUE self, VALUE file_name_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  VALUE_TO_C_PTR(BackpropTrainingSet_t, training_set, self);

  const char* file_name = StringValueCStr(file_name_val);

  BackpropTrainingSet_Load(training_set, file_name);

  return self;
}




static VALUE CBackpropTrainingSet_initialize(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  return self;
}




static void CBackpropTrainingSet_free(struct BackpropTrainingSet* self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingSet_Free(self);
}




static VALUE CBackpropTrainingSet_new(VALUE klass, VALUE x_value, VALUE y_value)
{
  BACKPROP_TRACE(__FUNCTION__);

  size_t count = 0;
  size_t x_size = 0;
  size_t y_size = 0;

  if ((x_value != Qnil) || (y_value != Qnil))
  {
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

    count = NUM2INT(x_count);
    x_size = NUM2INT(x_size_val);
    y_size = NUM2INT(y_size_val);
  }

  BackpropTrainingSet_t* instance = BackpropTrainingSet_Malloc(count, x_size, y_size);

  // wrap it in a ruby object, this will cause GC to call free function
  VALUE tdata = Data_Wrap_Struct(klass, 0, CBackpropTrainingSet_free, instance);

  // call initialize
  //VALUE argv[3] = {count, x_size, y_size};
  //  rb_obj_call_init(tdata, 0, 0);

  if ((x_value != Qnil) || (y_value != Qnil))
  {
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
  }

  return tdata;
}




//------------------------------------------------------------------------------
//
// BackpropExerciseStats
//
//------------------------------------------------------------------------------


static VALUE CBackpropExerciseStats_exercise_clock_ticks(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropExerciseStats_t* stats;
  Data_Get_Struct(self, BackpropExerciseStats_t, stats);

  return INT2NUM(stats->exercise_clock_ticks);
}




static VALUE CBackpropExerciseStats_activate_count(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropExerciseStats_t* stats;
  Data_Get_Struct(self, BackpropExerciseStats_t, stats);

  return INT2NUM(stats->activate_count);
}



static VALUE CBackpropExerciseStats_error(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropExerciseStats_t* stats;
  Data_Get_Struct(self, BackpropExerciseStats_t, stats);

  return rb_float_new(stats->error);
}




static VALUE CBackpropExerciseStats_to_hash(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropExerciseStats_t* stats;
  Data_Get_Struct(self, BackpropExerciseStats_t, stats);

  VALUE hash = rb_hash_new();
  rb_hash_aset(hash, rb_str_new2("exercise_clock_ticks"), CBackpropExerciseStats_exercise_clock_ticks(self));
  rb_hash_aset(hash, rb_str_new2("activate_count"), CBackpropExerciseStats_activate_count(self));
  rb_hash_aset(hash, rb_str_new2("error"), CBackpropExerciseStats_error(self));

  return hash;
}




static VALUE CBackpropExerciseStats_new(VALUE klass)
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


static VALUE CBackpropTrainingStats_set_weight_correction_total(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingStats_t* stats;
  Data_Get_Struct(self, BackpropTrainingStats_t, stats);

  return rb_float_new(stats->set_weight_correction_total);
}




static VALUE CBackpropTrainingStats_batch_weight_correction_total(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingStats_t* stats;
  Data_Get_Struct(self, BackpropTrainingStats_t, stats);

  return rb_float_new(stats->batch_weight_correction_total);
}




static VALUE CBackpropTrainingStats_teach_total(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingStats_t* stats;
  Data_Get_Struct(self, BackpropTrainingStats_t, stats);

  return INT2NUM(stats->teach_total);
}




static VALUE CBackpropTrainingStats_pair_total(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingStats_t* stats;
  Data_Get_Struct(self, BackpropTrainingStats_t, stats);

  return INT2NUM(stats->pair_total);
}




static VALUE CBackpropTrainingStats_set_total(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingStats_t* stats;
  Data_Get_Struct(self, BackpropTrainingStats_t, stats);

  return INT2NUM(stats->set_total);
}




static VALUE CBackpropTrainingStats_batches_total(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingStats_t* stats;
  Data_Get_Struct(self, BackpropTrainingStats_t, stats);

  return INT2NUM(stats->batches_total);
}




static VALUE CBackpropTrainingStats_stubborn_batches_total(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingStats_t* stats;
  Data_Get_Struct(self, BackpropTrainingStats_t, stats);

  return INT2NUM(stats->stubborn_batches_total);
}




static VALUE CBackpropTrainingStats_stagnate_batches_total(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingStats_t* stats;
  Data_Get_Struct(self, BackpropTrainingStats_t, stats);

  return INT2NUM(stats->stagnate_batches_total);
}




static VALUE CBackpropTrainingStats_train_clock(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingStats_t* stats;
  Data_Get_Struct(self, BackpropTrainingStats_t, stats);

  return INT2NUM(stats->train_clock);
}




static VALUE CBackpropTrainingStats_to_hash(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainingStats_t* stats;
  Data_Get_Struct(self, BackpropTrainingStats_t, stats);

  VALUE hash = rb_hash_new();
  rb_hash_aset(hash, rb_str_new2("set_weight_correction_total"), CBackpropTrainingStats_set_weight_correction_total(self));
  rb_hash_aset(hash, rb_str_new2("batch_weight_correction_total"), CBackpropTrainingStats_batch_weight_correction_total(self));
  rb_hash_aset(hash, rb_str_new2("teach_total"), CBackpropTrainingStats_teach_total(self));
  rb_hash_aset(hash, rb_str_new2("pair_total"), CBackpropTrainingStats_pair_total(self));
  rb_hash_aset(hash, rb_str_new2("set_total"), CBackpropTrainingStats_set_total(self));
  rb_hash_aset(hash, rb_str_new2("batches_total"), CBackpropTrainingStats_batches_total(self));
  rb_hash_aset(hash, rb_str_new2("stubborn_batches_total"), CBackpropTrainingStats_stubborn_batches_total(self));
  rb_hash_aset(hash, rb_str_new2("stagnate_batches_total"), CBackpropTrainingStats_stagnate_batches_total(self));
  rb_hash_aset(hash, rb_str_new2("train_clock"), CBackpropTrainingStats_train_clock(self));

  return hash;
}




static VALUE CBackpropTrainingStats_new(VALUE klass)
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


static VALUE CBackpropTrainer_teach_pair( VALUE trainer_val
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




static VALUE CBackpropTrainer_train_pair( VALUE trainer_val
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






static VALUE CBackpropTrainer_train_set( VALUE trainer_val
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






static VALUE CBackpropTrainer_train_batch( VALUE trainer_val
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




static VALUE CBackpropTrainer_train( VALUE trainer_val
                                   , VALUE training_stats_val
                                   , VALUE exercise_stats_val
                                   , VALUE network_val
                                   , VALUE training_set_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer = NULL;
  BackpropTrainingStats_t* training_stats = NULL;
  BackpropExerciseStats_t* exercise_stats = NULL;
  BackpropNetwork_t* network = NULL;
  BackpropTrainingSet_t* training_set = NULL;
  BACKPROP_FLOAT_T error = 0;

  Data_Get_Struct(trainer_val, BackpropTrainer_t, trainer);
  if (!trainer)
  {
    rb_raise(rb_eArgError, "NULL trainer");
    return Qnil;
  }

  Data_Get_Struct(training_stats_val, BackpropTrainingStats_t, training_stats);
  if (!training_stats)
  {
    rb_raise(rb_eArgError, "NULL training stats");
    return Qnil;
  }

  Data_Get_Struct(exercise_stats_val, BackpropExerciseStats_t, exercise_stats);
  if (!exercise_stats)
  {
    rb_raise(rb_eArgError, "NULL exercise stats");
    return Qnil;
  }

  Data_Get_Struct(network_val, BackpropNetwork_t, network);

  if (!network)
  {
    rb_raise(rb_eArgError, "NULL network");
    return Qnil;
  }

  Data_Get_Struct(training_set_val, BackpropTrainingSet_t, training_set);
  if (!training_set)
  {
    rb_raise(rb_eArgError, "NULL training set");
    return Qnil;
  }


  error = BackpropTrainer_Train( trainer
                               , training_stats
                               , exercise_stats
                               , network
                               , training_set);

  return rb_float_new(error);
}




static VALUE CBackpropTrainer_exercise( VALUE self_val
                                      , VALUE stats_val
                                      , VALUE network_val
                                      , VALUE training_set_val)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_t* trainer = NULL;
  BackpropExerciseStats_t* stats = NULL;
  BackpropNetwork_t* network = NULL;
  BackpropTrainingSet_t* training_set = NULL;
  BACKPROP_FLOAT_T error = 0.0;

  Data_Get_Struct(self_val, BackpropTrainer_t, trainer);
  if (!trainer)
  {
    rb_raise(rb_eArgError, "NULL self");
    return Qnil;
  }

  Data_Get_Struct(stats_val, BackpropExerciseStats_t, stats);
  if (!stats)
  {
    rb_raise(rb_eArgError, "NULL stats");
    return Qnil;
  }

  Data_Get_Struct(network_val, BackpropNetwork_t, network);
  if (!network)
  {
    rb_raise(rb_eArgError, "NULL network");
    return Qnil;
  }

  Data_Get_Struct(training_set_val, BackpropTrainingSet_t, training_set);
  if (!training_set)
  {
    rb_raise(rb_eArgError, "NULL training set");
    return Qnil;
  }

  error = BackpropTrainer_Exercise(trainer, stats, network, training_set);

  return rb_float_new(error);
}




static VALUE CBackpropTrainer_get_error_tolerance(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropTrainer_t* trainer;
    Data_Get_Struct(self, BackpropTrainer_t, trainer);

    return rb_float_new(BackpropTrainer_GetErrorTolerance(trainer));
  }
}




static VALUE CBackpropTrainer_get_learning_rate(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropTrainer_t* trainer;
    Data_Get_Struct(self, BackpropTrainer_t, trainer);

    return rb_float_new(BackpropTrainer_GetLearningRate(trainer));
  }
}




static VALUE CBackpropTrainer_get_mutation_rate(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropTrainer_t* trainer;
    Data_Get_Struct(self, BackpropTrainer_t, trainer);

    return rb_float_new(BackpropTrainer_GetMutationRate(trainer));
  }
}




static VALUE CBackpropTrainer_get_momentum_rate(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropTrainer_t* trainer;
    Data_Get_Struct(self, BackpropTrainer_t, trainer);

    return rb_float_new(BackpropTrainer_GetMomentumRate(trainer));
  }
}




static VALUE CBackpropTrainer_get_max_reps(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropTrainer_t* trainer;
    Data_Get_Struct(self, BackpropTrainer_t, trainer);

    return INT2NUM(BackpropTrainer_GetMaxReps(trainer));
  }
}




static VALUE CBackpropTrainer_get_max_batch_sets(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropTrainer_t* trainer;
    Data_Get_Struct(self, BackpropTrainer_t, trainer);

    return INT2NUM(BackpropTrainer_GetMaxBatchSets(trainer));
  }
}




static VALUE CBackpropTrainer_set_max_batch_sets(VALUE self, VALUE value)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropTrainer_t* trainer;
    Data_Get_Struct(self, BackpropTrainer_t, trainer);

    BackpropTrainer_SetMaxBatchSets(trainer, NUM2INT(value));

    return self;
  }
}




static VALUE CBackpropTrainer_get_max_batches(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropTrainer_t* trainer;
    Data_Get_Struct(self, BackpropTrainer_t, trainer);

    return INT2NUM(BackpropTrainer_GetMaxBatches(trainer));
  }
}




static VALUE CBackpropTrainer_set_max_batches(VALUE self, VALUE value)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropTrainer_t* trainer;
    Data_Get_Struct(self, BackpropTrainer_t, trainer);

    BackpropTrainer_SetMaxBatches(trainer, NUM2INT(value));

    return self;
  }
}




static VALUE CBackpropTrainer_get_stagnate_tolerance(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropTrainer_t* trainer;
    Data_Get_Struct(self, BackpropTrainer_t, trainer);

    return rb_float_new(BackpropTrainer_GetStagnateTolerance(trainer));
  }
}




static VALUE CBackpropTrainer_get_max_stagnate_sets(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropTrainer_t* trainer;
    Data_Get_Struct(self, BackpropTrainer_t, trainer);

    return INT2NUM(BackpropTrainer_GetMaxStagnateSets(trainer));
  }
}




static VALUE CBackpropTrainer_get_max_stagnate_batches(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropTrainer_t* trainer;
    Data_Get_Struct(self, BackpropTrainer_t, trainer);

    return INT2NUM(BackpropTrainer_GetMaxStagnateBatches(trainer));
  }
}




static VALUE CBackpropTrainer_get_min_set_weight_correction_limit(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropTrainer_t* trainer;
    Data_Get_Struct(self, BackpropTrainer_t, trainer);

    return rb_float_new(BackpropTrainer_GetMinSetWeightCorrectionLimit(trainer));
  }
}




static VALUE CBackpropTrainer_get_min_batch_weight_correction_limit(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropTrainer_t* trainer;
    Data_Get_Struct(self, BackpropTrainer_t, trainer);

    return rb_float_new(BackpropTrainer_GetMinBatchWeightCorrectionLimit(trainer));
  }
}




static VALUE CBackpropTrainer_get_batch_prune_threshold(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropTrainer_t* trainer;
    Data_Get_Struct(self, BackpropTrainer_t, trainer);

    return rb_float_new(BackpropTrainer_GetBatchPruneThreshold(trainer));
  }
}




static VALUE CBackpropTrainer_get_training_ratio(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropTrainer_t* trainer;
    Data_Get_Struct(self, BackpropTrainer_t, trainer);

    return rb_float_new(BackpropTrainer_GetTrainingRatio(trainer));
  }
}





static VALUE CBackpropTrainer_get_batch_prune_rate(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropTrainer_t* trainer;
    Data_Get_Struct(self, BackpropTrainer_t, trainer);

    return rb_float_new(BackpropTrainer_GetBatchPruneRate(trainer));
  }
}




static VALUE CBackpropTrainer_set_batch_prune_rate(VALUE self, VALUE value)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropTrainer_t* trainer;
    Data_Get_Struct(self, BackpropTrainer_t, trainer);

    BackpropTrainer_SetBatchPruneRate(trainer, NUM2DBL(value));

    return self;
  }
}




static VALUE CBackpropTrainer_to_hash(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropTrainer_t* trainer;
    Data_Get_Struct(self, BackpropTrainer_t, trainer);

    VALUE hash = rb_hash_new();
    rb_hash_aset(hash, rb_str_new2("error_tolerance"), CBackpropTrainer_get_error_tolerance(self));
    rb_hash_aset(hash, rb_str_new2("learning_rate"), CBackpropTrainer_get_learning_rate(self));
    rb_hash_aset(hash, rb_str_new2("mutation_rate"), CBackpropTrainer_get_mutation_rate(self));
    rb_hash_aset(hash, rb_str_new2("momentum_rate"), CBackpropTrainer_get_momentum_rate(self));
    rb_hash_aset(hash, rb_str_new2("max_reps"), CBackpropTrainer_get_max_reps(self));
    rb_hash_aset(hash, rb_str_new2("max_batch_sets"), CBackpropTrainer_get_max_batch_sets(self));
    rb_hash_aset(hash, rb_str_new2("max_batches"), CBackpropTrainer_get_max_batches(self));
    rb_hash_aset(hash, rb_str_new2("stagnate_tolerance"), CBackpropTrainer_get_stagnate_tolerance(self));
    rb_hash_aset(hash, rb_str_new2("max_stagnate_sets"), CBackpropTrainer_get_max_stagnate_sets(self));
    rb_hash_aset(hash, rb_str_new2("max_stagnate_batches"), CBackpropTrainer_get_max_stagnate_batches(self));
    rb_hash_aset(hash, rb_str_new2("min_set_weight_correction_limit"), CBackpropTrainer_get_min_set_weight_correction_limit(self));
    rb_hash_aset(hash, rb_str_new2("min_batch_weight_correction_limit"), CBackpropTrainer_get_min_batch_weight_correction_limit(self));
    rb_hash_aset(hash, rb_str_new2("batch_prune_threshold"), CBackpropTrainer_get_batch_prune_threshold(self));
    rb_hash_aset(hash, rb_str_new2("batch_prune_rate"), CBackpropTrainer_get_batch_prune_rate(self));
    rb_hash_aset(hash, rb_str_new2("training_ratio"), CBackpropTrainer_get_training_ratio(self));

    return hash;
  }
}




static VALUE CBackpropTrainer_initialize(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);

  return self;
}




static void CBackpropTrainer_Free(struct BackpropTrainer* trainer)
{
  BACKPROP_TRACE(__FUNCTION__);

  BackpropTrainer_Free(trainer);
}




static VALUE CBackpropTrainer_new(VALUE klass, VALUE network_value)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    // get the network
    BackpropNetwork_t* network;
    Data_Get_Struct(network_value, BackpropNetwork_t, network);

    // allocate a trainer
    BackpropTrainer_t* trainer = BackpropTrainer_Malloc(network);

    BackpropTrainer_SetToDefault(trainer);
    BackpropTrainer_SetToDefaultIO(trainer);

    // wrap it in a ruby object, this will cause GC to call free function
    VALUE tdata = Data_Wrap_Struct(klass, 0, CBackpropTrainer_Free, trainer);

    // call initialize
    VALUE argv[1] = {network_value};
    rb_obj_call_init(tdata, 1, argv);

    return tdata;
  }
}




//------------------------------------------------------------------------------
//
// BackpropEvolutionStats
//
//------------------------------------------------------------------------------


static VALUE CBackpropEvolutionStats_get_generation_count(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropEvolutionStats_t* stats;
    Data_Get_Struct(self, BackpropEvolutionStats_t, stats);

    return INT2NUM(stats->generation_count);
  }
}




static VALUE CBackpropEvolutionStats_get_mate_networks_count(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropEvolutionStats_t* stats;
    Data_Get_Struct(self, BackpropEvolutionStats_t, stats);

    return INT2NUM(stats->mate_networks_count);
  }
}




static VALUE CBackpropEvolutionStats_get_evolve_clock(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropEvolutionStats_t* stats;
    Data_Get_Struct(self, BackpropEvolutionStats_t, stats);

    return INT2NUM(stats->evolve_clock);
  }
}




static VALUE CBackpropEvolutionStats_to_hash(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    VALUE hash = rb_hash_new();

    rb_hash_aset(hash, rb_str_new2("generation_count"), CBackpropEvolutionStats_get_generation_count(self));
    rb_hash_aset(hash, rb_str_new2("mate_networks_count"), CBackpropEvolutionStats_get_mate_networks_count(self));
    rb_hash_aset(hash, rb_str_new2("evolve_clock"), CBackpropEvolutionStats_get_evolve_clock(self));

    return hash;
  }
}




static VALUE CBackpropEvolutionStats_new(VALUE klass)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropEvolutionStats_t* obj = xmalloc(sizeof(BackpropEvolutionStats_t));

    // wrap it in a ruby object, this will cause GC to call free function
    VALUE tdata = Data_Wrap_Struct(klass, 0, xfree, obj);

    // call initialize
    //rb_obj_call_init(tdata, 0, 0);

    return tdata;
  }
}



//------------------------------------------------------------------------------
//
// BackpropEvolutionStats
//
//------------------------------------------------------------------------------


static VALUE CBackpropEvolver_get_pool_count(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropEvolver_t* obj;
    Data_Get_Struct(self, BackpropEvolver_t, obj);

    return INT2NUM(obj->pool_count);
  }
}




static VALUE CBackpropEvolver_get_max_generations(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropEvolver_t* obj;
    Data_Get_Struct(self, BackpropEvolver_t, obj);

    return INT2NUM(obj->max_generations);
  }
}




static VALUE CBackpropEvolver_get_mate_rate(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropEvolver_t* obj;
    Data_Get_Struct(self, BackpropEvolver_t, obj);

    return rb_float_new(obj->mate_rate);
  }
}




static VALUE CBackpropEvolver_get_mutation_limit(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropEvolver_t* obj;
    Data_Get_Struct(self, BackpropEvolver_t, obj);

    return rb_float_new(obj->mutation_limit);
  }
}




static VALUE CBackpropEvolver_get_seed(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropEvolver_t* obj;
    Data_Get_Struct(self, BackpropEvolver_t, obj);

    return rb_float_new(obj->mutation_limit);
  }
}




static VALUE CBackpropEvolver_to_hash(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    VALUE hash = rb_hash_new();

    rb_hash_aset(hash, rb_str_new2("pool_count"), CBackpropEvolver_get_pool_count(self));
    rb_hash_aset(hash, rb_str_new2("max_generations"), CBackpropEvolver_get_max_generations(self));
    rb_hash_aset(hash, rb_str_new2("mate_rate"), CBackpropEvolver_get_mate_rate(self));
    rb_hash_aset(hash, rb_str_new2("mutation_limit"), CBackpropEvolver_get_mutation_limit(self));
    rb_hash_aset(hash, rb_str_new2("seed"), CBackpropEvolver_get_seed(self));

    return hash;
  }
}




static VALUE CBackpropEvolver_set_to_default(VALUE self)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    BackpropEvolver_t* obj;
    Data_Get_Struct(self, BackpropEvolver_t, obj);

    BackpropEvolver_SetToDefault(obj);

    return self;
  }
}






static VALUE CBackpropEvolver_evolve( VALUE evolver_val
                                    , VALUE evolution_stats_val
                                    , VALUE trainer_val
                                    , VALUE training_stats_val
                                    , VALUE exercise_stats_val
                                    , VALUE network_val
                                    , VALUE training_set_val)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    VALUE_TO_C_PTR(BackpropEvolver_t, evolver, evolver_val);
    VALUE_TO_C_PTR(BackpropEvolutionStats_t, evolution_stats, evolution_stats_val);
    VALUE_TO_C_PTR(BackpropTrainer_t, trainer, trainer_val);
    VALUE_TO_C_PTR(BackpropTrainingStats_t, training_stats, training_stats_val);
    VALUE_TO_C_PTR(BackpropExerciseStats_t, exercise_stats, exercise_stats_val);
    VALUE_TO_C_PTR(BackpropNetwork_t, network, network_val);
    VALUE_TO_C_PTR(BackpropTrainingSet_t, training_set, training_set_val);
    {
      BACKPROP_FLOAT_T result = BackpropEvolver_Evolve( evolver
                                                      , evolution_stats
                                                      , trainer
                                                      , training_stats
                                                      , exercise_stats
                                                      , network
                                                      , training_set);

      return rb_float_new(result);
    }
  }
}




static VALUE CBackpropEvolver_new(VALUE klass)
{
  BACKPROP_TRACE(__FUNCTION__);
  {
    struct BackpropEvolver* obj = xmalloc(sizeof(struct BackpropEvolver));

    // wrap it in a ruby object, this will cause GC to call free function
    VALUE tdata = Data_Wrap_Struct(klass, 0, xfree, obj);

    // call initialize
    //rb_obj_call_init(tdata, 0, 0);

    return tdata;
  }
}




//------------------------------------------------------------------------------
//
// module initialization
//
//------------------------------------------------------------------------------

// The initialization method for this module
void Init_backproprb()
{
  BACKPROP_TRACE(__FUNCTION__);
  // configure Backprop libary
  //Backprop_SetMalloc(CBackprop_Malloc);
  //Backprop_SetFree(CBackprop_Free);


  // Define module methods
  cBackproprb = rb_define_module("CBackproprb");
  rb_define_module_function(cBackproprb, "used", CBackprop_used, 0);
  rb_define_module_function(cBackproprb, "sigmoid", CBackprop_sigmoid, 1);
  rb_define_module_function(cBackproprb, "uniform_random_int", CBackprop_uniform_random_int, 0);


  // Define class CBackproprb::CBackpropLayer
  cBackpropLayer = rb_define_class_under(cBackproprb, "CLayer", rb_cObject);
  rb_define_singleton_method(cBackpropLayer, "new", CBackpropLayer_new, 2);
  rb_define_method(cBackpropLayer, "initialize", CBackpropLayer_initialize, 2);
  rb_define_method(cBackpropLayer, "w", CBackpropLayer_get_W, 0);
  rb_define_method(cBackpropLayer, "w=", CBackpropLayer_set_W, 1);
  rb_define_method(cBackpropLayer, "w_count", CBackpropLayer_get_W_count, 0);
  rb_define_method(cBackpropLayer, "w_sum", CBackpropLayer_get_W_sum, 0);
  rb_define_method(cBackpropLayer, "w_mean", CBackpropLayer_get_W_mean, 0);
  rb_define_method(cBackpropLayer, "w_stddev", CBackpropLayer_get_W_stddev, 0);
  rb_define_method(cBackpropLayer, "g", CBackpropLayer_get_g, 0);
  rb_define_method(cBackpropLayer, "g=", CBackpropLayer_set_g, 1);
  rb_define_method(cBackpropLayer, "x", CBackpropLayer_get_x, 0);
  rb_define_method(cBackpropLayer, "x=", CBackpropLayer_set_x, 1);
  rb_define_method(cBackpropLayer, "x_count", CBackpropLayer_get_x_count, 0);
  rb_define_method(cBackpropLayer, "y", CBackpropLayer_get_y, 0);
  rb_define_method(cBackpropLayer, "y=", CBackpropLayer_set_y, 1);
  rb_define_method(cBackpropLayer, "y_count", CBackpropLayer_get_y_count, 0);
  rb_define_method(cBackpropLayer, "randomize", CBackpropLayer_randomize, 1);
  rb_define_method(cBackpropLayer, "identity", CBackpropLayer_identity, 0);
  rb_define_method(cBackpropLayer, "reset", CBackpropLayer_reset, 0);
  rb_define_method(cBackpropLayer, "prune", CBackpropLayer_prune, 1);
  rb_define_method(cBackpropLayer, "activate", CBackpropLayer_activate, 0);
  rb_define_method(cBackpropLayer, "to_hash", CBackpropLayer_to_hash, 0);


  // Define class CBackproprb::CNetwork
  cBackpropNetwork = rb_define_class_under(cBackproprb, "CNetwork", rb_cObject);
  rb_define_singleton_method(cBackpropNetwork, "new", CBackpropNetwork_new, 1);
  rb_define_method(cBackpropNetwork, "initialize", CBackpropNetwork_initialize, 1);
  rb_define_method(cBackpropNetwork, "activate", CBackpropNetwork_activate, 1);
  rb_define_method(cBackpropNetwork, "x", CBackpropNetwork_get_x, 0);
  rb_define_method(cBackpropNetwork, "x_size", CBackpropNetwork_x_size, 0);
  rb_define_method(cBackpropNetwork, "y", CBackpropNetwork_get_y, 0);
  rb_define_method(cBackpropNetwork, "y_size", CBackpropNetwork_y_size, 0);
  rb_define_method(cBackpropNetwork, "layers_count", CBackpropNetwork_layers_count, 0);
  rb_define_method(cBackpropNetwork, "layer_get", CBackpropNetwork_get_layer, 1);

  rb_define_method(cBackpropNetwork, "jitter", CBackpropNetwork_get_jitter, 0);
  rb_define_method(cBackpropNetwork, "jitter=", CBackpropNetwork_set_jitter, 1);
  rb_define_method(cBackpropNetwork, "randomize", CBackpropNetwork_randomize, 1);
  rb_define_method(cBackpropNetwork, "identity", CBackpropNetwork_identity, 0);
  rb_define_method(cBackpropNetwork, "reset", CBackpropNetwork_reset, 0);
  rb_define_method(cBackpropNetwork, "prune", CBackpropNetwork_prune, 1);
  rb_define_method(cBackpropNetwork, "stats", CBackpropNetwork_get_stats, 0);
  rb_define_method(cBackpropNetwork, "to_hash", CBackpropNetwork_to_hash, 0);
  rb_define_method(cBackpropNetwork, "to_file", CBackpropNetwork_to_file, 1);
  rb_define_method(cBackpropNetwork, "from_file", CBackpropNetwork_from_file, 1);

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
  rb_define_singleton_method(cBackpropTrainingSet, "new", CBackpropTrainingSet_new, 2);
  rb_define_method(cBackpropTrainingSet, "initialize", CBackpropTrainingSet_initialize, 2);
  rb_define_method(cBackpropTrainingSet, "count", CBackpropTrainingSet_count, 0);
  rb_define_method(cBackpropTrainingSet, "x_size", CBackpropTrainingSet_x_size, 0);
  rb_define_method(cBackpropTrainingSet, "y_size", CBackpropTrainingSet_y_size, 0);
  rb_define_method(cBackpropTrainingSet, "x_at", CBackpropTrainingSet_x_at, 1);
  rb_define_method(cBackpropTrainingSet, "y_at", CBackpropTrainingSet_y_at, 1);

  rb_define_method(cBackpropTrainingSet, "to_hash", CBackpropTrainingSet_to_hash, 0);
  rb_define_method(cBackpropTrainingSet, "to_file", CBackpropTrainingSet_to_file, 1);
  rb_define_method(cBackpropTrainingSet, "from_file", CBackpropTrainingSet_from_file, 1);


  // Define class CBackproprb::CExerciseStats
  cBackpropExerciseStats = rb_define_class_under(cBackproprb, "CExerciseStats", rb_cObject);
  rb_define_singleton_method(cBackpropExerciseStats, "new", CBackpropExerciseStats_new, 0);
  rb_define_method(cBackpropExerciseStats, "exercise_clock_ticks", CBackpropExerciseStats_exercise_clock_ticks, 0);
  rb_define_method(cBackpropExerciseStats, "activate_count", CBackpropExerciseStats_activate_count, 0);
  rb_define_method(cBackpropExerciseStats, "error", CBackpropExerciseStats_error, 0);
  rb_define_method(cBackpropExerciseStats, "to_hash", CBackpropExerciseStats_to_hash, 0);


  // Define class CBackproprb::CTrainingStats
  cBackpropTrainingStats = rb_define_class_under(cBackproprb, "CTrainingStats", rb_cObject);
  rb_define_singleton_method(cBackpropTrainingStats, "new", CBackpropTrainingStats_new, 0);
  rb_define_method(cBackpropTrainingStats, "set_weight_correction_total", CBackpropTrainingStats_set_weight_correction_total, 0);
  rb_define_method(cBackpropTrainingStats, "batch_weight_correction_total", CBackpropTrainingStats_batch_weight_correction_total, 0);
  rb_define_method(cBackpropTrainingStats, "teach_total", CBackpropTrainingStats_teach_total, 0);
  rb_define_method(cBackpropTrainingStats, "pair_total", CBackpropTrainingStats_pair_total, 0);
  rb_define_method(cBackpropTrainingStats, "set_total", CBackpropTrainingStats_set_total, 0);
  rb_define_method(cBackpropTrainingStats, "batches_total", CBackpropTrainingStats_batches_total, 0);
  rb_define_method(cBackpropTrainingStats, "stubborn_batches_total", CBackpropTrainingStats_stubborn_batches_total, 0);
  rb_define_method(cBackpropTrainingStats, "stagnate_batches_total", CBackpropTrainingStats_stagnate_batches_total, 0);
  rb_define_method(cBackpropTrainingStats, "train_clock", CBackpropTrainingStats_train_clock, 0);
  rb_define_method(cBackpropTrainingStats, "to_hash", CBackpropTrainingStats_to_hash, 0);

  // Define class CBackprop::CTrainer
  cBackpropTrainer = rb_define_class_under(cBackproprb, "CTrainer", rb_cObject);
  rb_define_singleton_method(cBackpropTrainer, "new", CBackpropTrainer_new, 1);
  rb_define_method(cBackpropTrainer, "initialize", CBackpropTrainer_initialize, 1);

  rb_define_method(cBackpropTrainer, "error_tolerance", CBackpropTrainer_get_error_tolerance, 0);
  rb_define_method(cBackpropTrainer, "learning_rate", CBackpropTrainer_get_learning_rate, 0);
  rb_define_method(cBackpropTrainer, "mutation_rate", CBackpropTrainer_get_mutation_rate, 0);
  rb_define_method(cBackpropTrainer, "momentum_rate", CBackpropTrainer_get_momentum_rate, 0);
  rb_define_method(cBackpropTrainer, "max_reps", CBackpropTrainer_get_max_reps, 0);
  rb_define_method(cBackpropTrainer, "max_batch_sets", CBackpropTrainer_get_max_batch_sets, 0);
  rb_define_method(cBackpropTrainer, "max_batches", CBackpropTrainer_get_max_batches, 0);
  rb_define_method(cBackpropTrainer, "stagnate_tolerance", CBackpropTrainer_get_stagnate_tolerance, 0);
  rb_define_method(cBackpropTrainer, "max_stagnate_sets", CBackpropTrainer_get_max_stagnate_sets, 0);
  rb_define_method(cBackpropTrainer, "max_stagnate_batches", CBackpropTrainer_get_max_stagnate_batches, 0);
  rb_define_method(cBackpropTrainer, "min_set_weight_correction_limit", CBackpropTrainer_get_min_set_weight_correction_limit, 0);
  rb_define_method(cBackpropTrainer, "min_batch_weight_correction_limit", CBackpropTrainer_get_min_batch_weight_correction_limit, 0);
  rb_define_method(cBackpropTrainer, "batch_prune_threshold", CBackpropTrainer_get_batch_prune_threshold, 0);
  rb_define_method(cBackpropTrainer, "batch_prune_rate", CBackpropTrainer_get_batch_prune_rate, 0);
  rb_define_method(cBackpropTrainer, "training_ratio", CBackpropTrainer_get_training_ratio, 0);

  rb_define_method(cBackpropTrainer, "max_batch_sets=", CBackpropTrainer_set_max_batch_sets, 1);
  rb_define_method(cBackpropTrainer, "max_batches=", CBackpropTrainer_set_max_batches, 1);
  rb_define_method(cBackpropTrainer, "batch_prune_rate=", CBackpropTrainer_set_batch_prune_rate, 1);

  rb_define_method(cBackpropTrainer, "exercise", CBackpropTrainer_exercise, 3);
  rb_define_method(cBackpropTrainer, "teach_pair", CBackpropTrainer_teach_pair, 4);
  rb_define_method(cBackpropTrainer, "train_pair", CBackpropTrainer_train_pair, 4);
  rb_define_method(cBackpropTrainer, "train_set", CBackpropTrainer_train_set, 3);
  rb_define_method(cBackpropTrainer, "train_batch", CBackpropTrainer_train_batch, 4);
  rb_define_method(cBackpropTrainer, "train", CBackpropTrainer_train, 4);

  rb_define_method(cBackpropTrainer, "to_hash", CBackpropTrainer_to_hash, 0);


  cBackpropEvolutionStats = rb_define_class_under(cBackproprb, "CEvolutionStats", rb_cObject);
  rb_define_singleton_method(cBackpropEvolutionStats, "new", CBackpropEvolutionStats_new, 0);
  rb_define_method(cBackpropEvolutionStats, "generation_count", CBackpropEvolutionStats_get_generation_count, 0);
  rb_define_method(cBackpropEvolutionStats, "mate_networks_count", CBackpropEvolutionStats_get_mate_networks_count, 0);
  rb_define_method(cBackpropEvolutionStats, "evolve_clock", CBackpropEvolutionStats_get_evolve_clock, 0);
  rb_define_method(cBackpropEvolutionStats, "to_hash", CBackpropEvolutionStats_to_hash, 0);


  cBackpropEvolver = rb_define_class_under(cBackproprb, "CEvolver", rb_cObject);
  rb_define_singleton_method(cBackpropEvolver, "new", CBackpropEvolver_new, 0);
  rb_define_method(cBackpropEvolver, "pool_count", CBackpropEvolver_get_pool_count, 0);
  rb_define_method(cBackpropEvolver, "max_generations", CBackpropEvolver_get_max_generations, 0);
  rb_define_method(cBackpropEvolver, "mate_rate", CBackpropEvolver_get_mate_rate, 0);
  rb_define_method(cBackpropEvolver, "mutation_limit", CBackpropEvolver_get_mutation_limit, 0);
  rb_define_method(cBackpropEvolver, "seed", CBackpropEvolver_get_seed, 0);
  rb_define_method(cBackpropEvolver, "to_hash", CBackpropEvolver_to_hash, 0);

  rb_define_method(cBackpropEvolver, "set_to_default", CBackpropEvolver_set_to_default, 0);
  rb_define_method(cBackpropEvolver, "evolve", CBackpropEvolver_evolve, 6);
}
