# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:47:10 2020

@author: cfavr
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import json
import os
import random
import re

import enum

os.chdir("C:\\Users\\cfavr\\Documents\\Python Scripts\\QA")

import tensorflow_hub as hub
import bert_modeling as modeling
import bert_optimization as optimization
#import albert_tokenization as tokenization
import sentencepiece as spm
import copy
import tensorflow_model_optimization as tfmot


import absl
from absl import app
import sys
import numpy as np
import tensorflow as tf
import tensorflow_text
from QA_utilities import *

#!pip install -q tf-nightly --user
#output_dir = "C:\\Users\\cfavr\\Documents\\Python Scripts\\QA"
#bert_config_file = "bert_config.json"
#vocab_file = "vocab-nq.txt"

from tensorflow.python.keras import initializers

class TDense(tf.keras.layers.Layer):
    def __init__(self,
                 output_size,
                 kernel_initializer=None,
                 bias_initializer="zeros",
                **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
        if kernel_initializer is not None:
            self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        else:
            self.kernel_initializer = None
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    def build(self,input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
          raise TypeError("Unable to build `TDense` layer with "
                          "non-floating point (and non-complex) "
                          "dtype %s" % (dtype,))
        input_shape = tf.TensorShape(input_shape)
        if tf.compat.dimension_value(input_shape[-1]) is None:
          raise ValueError("The last dimension of the inputs to "
                           "`TDense` should be defined. "
                           "Found `None`.")
        last_dim = tf.compat.dimension_value(input_shape[-1])
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=3, axes={-1: last_dim})
        self.kernel = self.add_weight(
            "kernel",
            shape=[self.output_size,last_dim],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)
        self.bias = self.add_weight(
            "bias",
            shape=[self.output_size],
            initializer=self.bias_initializer,
            dtype=self.dtype,
            trainable=True)
        super(TDense, self).build(input_shape)
    def call(self,x):
        return tf.matmul(x,self.kernel,transpose_b=True)+self.bias
    
    def get_config(self):
        base_config = super(TDense, self).get_config()
        config = {
        'output_size': self.output_size,
        'kernel_initializer': self.kernel_initializer,
        'bias_initializer':self.bias_initializer
        }
        return dict(list(base_config.items()) + list(config.items()))
    

def bert_model(config):
    """
    outputs:
    'pooled_output' shape [batch_size, 768], for each sequence
    'sequence_output' shape [batch_size,max_seq_length,768], token-level representations

    Returns
    -------
    model : TYPE
        DESCRIPTION.
        
    outputs are 1) the unique_id (just to mark it, but no product of model)
                2) start_logits ()
    
    """
    #with tfmot.quantization.keras.quantize_scope({'KerasLayer':hub.KerasLayer}):
    #    quantized = tfmot.quantization.keras.quantize_apply(model)
    seq_len = config['max_position_embeddings']
    unique_id  = tf.keras.Input(shape=(1,),dtype=tf.int64,name='unique_ids')
    input_ids   = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='input_ids')
    input_mask  = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='input_mask')
    segment_ids = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='segment_ids')
    #BERT = modeling.BertModel(config=config,name='bert')
    BERT = tfmot.quantization.keras.quantize_annotate_layer(modeling.BertModel(config=config,name='bert'))
    #tfmot.quantization.keras.quantize_annotate_layer(BERT)
    #pooled_output, sequence_output = BERT(input_word_ids=input_ids,
    #                              input_mask=input_mask,
    #                              input_type_ids=segment_ids)
    
    pooled_output, sequence_output = BERT([input_ids,input_mask,segment_ids])
    
    ### NOTE: output of sequence_output is [batch_size,max_seq_length,768]
    
    logits = TDense(2,name='logits')(sequence_output)   ### I think it's calculating this for each token in the sequence
    start_logits,end_logits = tf.split(logits,axis=-1,num_or_size_splits= 2,name='split')
    
    ### These will also return one value per seq length... but should they?
    start_logits = tf.squeeze(start_logits,axis=-1,name='start_squeeze')
    end_logits   = tf.squeeze(end_logits,  axis=-1,name='end_squeeze')
    ### Do I need to add another layer for each that goes over the seq_length, to condense to 1?
    
    ans_type = tf.keras.layers.Dense(5,name='ans_type')(pooled_output)
    model= tf.keras.Model([input_ for input_ in [unique_id,input_ids,input_mask,segment_ids] 
                           if input_ is not None],
                          #[unique_id,start_logits,end_logits,ans_type],
                          [start_logits,end_logits,ans_type],
                          name='bert_model') 
    with tfmot.quantization.keras.quantize_scope({'BertModel':modeling.BertModel}):
        model = tfmot.quantization.keras.quantize_apply(model)
    return model




def albert_model():
    """
    outputs:
    'pooled_output' shape [batch_size, 768], for each sequence
    'sequence_output' shape [batch_size,max_seq_length,768], token-level representations

    Returns
    -------
    model : TYPE
        DESCRIPTION.
        
    outputs are 1) the unique_id (just to mark it, but no product of model)
                2) start_logits ()
    
    """
    
    seq_len = 512
    unique_id  = tf.keras.Input(shape=(1,),dtype=tf.int64,name='unique_ids')
    #input_ids   = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='input_word_ids')
    #input_mask  = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='input_mask')
    #segment_ids = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='input_type_ids')

    encoder_inputs = dict(
    input_word_ids=tf.keras.Input(shape=(seq_len,), dtype=tf.int32,name='inputs/input_word_ids'),
    input_mask=tf.keras.Input(shape=(seq_len,), dtype=tf.int32,name='inputs/input_mask'),
    input_type_ids=tf.keras.Input(shape=(seq_len,), dtype=tf.int32,name='inputs/input_type_ids'),
    )


    albert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/2",
                                  trainable=False)
    print("POOP",albert_layer.get_config())
    albert_outputs = albert_layer(encoder_inputs)
    ### NOTE: output of sequence_output is [batch_size,max_seq_length,768]
    
    logits = TDense(2,name='logits')(albert_outputs['sequence_output'])   ### I think it's calculating this for each token in the sequence
    start_logits,end_logits = tf.split(logits,axis=-1,num_or_size_splits= 2,name='split')
    
    ### These will also return one value per seq length... but should they?
    start_logits = tf.squeeze(start_logits,axis=-1,name='start_squeeze')
    end_logits   = tf.squeeze(end_logits,  axis=-1,name='end_squeeze')
    ### Do I need to add another layer for each that goes over the seq_length, to condense to 1?
    
    ans_type = tf.keras.layers.Dense(5,name='ans_type')(albert_outputs['pooled_output'])
    return tf.keras.Model([unique_id] + [encoder_inputs[key] for key in encoder_inputs.keys()],
                          #[unique_id,start_logits,end_logits,ans_type],
                          [start_logits,end_logits,ans_type],
                          name='albert') 


# Computes the loss for positions.
def compute_loss(logits, positions):
    one_hot_positions = tf.one_hot(
        tf.cast(positions,tf.int32),\
            depth=512,dtype=tf.float32)
            # depth=max_seq_length, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(tf.cast(logits,tf.float32), axis=-1)
    print(positions)
    print(one_hot_positions * log_probs)
    loss = -tf.reduce_mean(
        input_tensor=tf.reduce_sum(input_tensor=one_hot_positions * log_probs, axis=-1))
    #        print("LOSS!!!",loss)
    return loss

# Computes the loss for labels.
def compute_label_loss(logits, labels):
    one_hot_labels = tf.one_hot(
        tf.cast(labels,tf.int32), depth=len(AnswerType), dtype=tf.float32)
    log_probs = tf.nn.log_softmax(tf.cast(logits,tf.float32), axis=-1)
    loss = -tf.reduce_mean(
        input_tensor=tf.reduce_sum(input_tensor=one_hot_labels * log_probs, axis=-1))
    #        print("LABELLOSS!!!", loss)
    return loss


def total_loss(y_true,y_pred):
    #pred_id,start_logits,end_logits,answer_type_logits = y_pred
    #unique_id, start_positions, end_positions, answer_types = y_true
    start_logits,end_logits,answer_type_logits = y_pred
    start_positions, end_positions, answer_types = y_true
    start_loss = compute_loss(start_logits, start_positions)
    end_loss = compute_loss(end_logits, end_positions)
    answer_type_loss = compute_label_loss(answer_type_logits, answer_types)
    total_loss = tf.add_n([start_loss,end_loss,answer_type_loss])
    return total_loss




def model_fn_builder(config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, bert_yn=True):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        absl.logging.info("*** Features ***")
        for name in sorted(features.keys()):
          absl.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
    
        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
    
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    
        if bert_yn:
            model = bert_model(config.to_dict())
        else:
            model = albert_model()
        

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            checkpoint_tf = tf.train.Checkpoint(model=model)
            status = checkpoint_tf.restore(init_checkpoint)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            seq_length = modeling.get_shape_list(input_ids)[1]

            start_positions = features["start_positions"]
            end_positions = features["end_positions"]
            answer_types = features["answer_types"]
            #y_true = (unique_ids,start_positions,end_positions,answer_types)
            y_true = (start_positions,end_positions,answer_types)


          #total_loss = (start_loss + end_loss + answer_type_loss) / 3.0
    
    #       optimizer = optimization.create_optimizer(learning_rate,
    #                                                num_train_steps,
    #                                                num_warmup_steps)
    
            optimizer = create_optimizer(learning_rate,
                                    num_train_steps,
                                    num_warmup_steps)
        
    #       train_op = optimizer.minimize(total_loss, var_list)
            with tf.GradientTape() as tape:
                y_pred = model((unique_ids,input_ids,input_mask,segment_ids))
                loss = total_loss(y_true,y_pred)

            var_list = model.trainable_variables
            #print("LOSS",var_list)
            #compgrad = compute_gradients(optimizer,loss, var_list)
            grad = tape.gradient(loss, var_list)
            #print("TRAIN_OP", compgrad)
            #print("GRADS",grad)
            #print("WATCHED VARS", tape.watched_variables())

            grads_and_vars = list(zip(grad, var_list))
            
            train_op = optimizer.apply_gradients(grads_and_vars)
            #print("REAL", train_op)
    

            output_spec = tf.estimator.EstimatorSpec(
                          mode=mode,
                          loss=loss,
                          train_op=train_op,
                          scaffold=scaffold_fn)
            
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
              "unique_ids": unique_ids,
              "start_logits": start_logits,
              "end_logits": end_logits,
              "answer_type_logits": answer_type_logits,
          }
            output_spec = tf.estimator.EstimatorSpec(
              mode=mode, predictions=predictions, scaffold=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and PREDICT modes are supported: %s" % (mode))
    
        return output_spec
    
    return model_fn


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Applys a warmup schedule on a given learning rate decay schedule."""

  def __init__(
      self,
      initial_learning_rate,
      decay_schedule_fn,
      warmup_steps,
      power=1.0,
      name=None):
    super(WarmUp, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.warmup_steps = warmup_steps
    self.power = power
    self.decay_schedule_fn = decay_schedule_fn
    self.name = name

  def __call__(self, step):
    #with tf.name_scope(self.name or 'WarmUp') as name:
      # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
      # learning rate will be `global_step/num_warmup_steps * init_lr`.
      global_step_float = tf.cast(step, tf.float32)
      warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
      warmup_percent_done = global_step_float / warmup_steps_float
      warmup_learning_rate = (
          self.initial_learning_rate *
          tf.math.pow(warmup_percent_done, self.power))
      return tf.cond(global_step_float < warmup_steps_float,
                     lambda: warmup_learning_rate,
                     lambda: self.decay_schedule_fn(step),
                     name='WarmUp')

  def get_config(self):
    return {
        'initial_learning_rate': self.initial_learning_rate,
        'decay_schedule_fn': self.decay_schedule_fn,
        'warmup_steps': self.warmup_steps,
        'power': self.power,
        'name': self.name
    }


def create_optimizer(init_lr, num_train_steps, num_warmup_steps):
    """Creates an optimizer with learning rate schedule."""
    # Implements linear decay of the learning rate.
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=init_lr,
        decay_steps=num_train_steps,
        end_learning_rate=0.0)
    if num_warmup_steps:
      learning_rate_fn = WarmUp(initial_learning_rate=init_lr,
                                decay_schedule_fn=learning_rate_fn,
                                warmup_steps=num_warmup_steps)
    optimizer = optimization.AdamWeightDecay(
        learning_rate=learning_rate_fn,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=['layer_norm', 'bias'])
    return optimizer


def compute_gradients(optimizer, loss, var_list, grad_loss=None):
    with tf.GradientTape() as tape:
      if not callable(var_list):
        tape.watch(var_list)
      loss_value = loss
    if callable(var_list):
      var_list = var_list()
    var_list = tf.nest.flatten(var_list)
    print("VARLIST AFTER NEST",var_list)
    print("LOSS VALUE",loss_value)
    with tf.name_scope(optimizer._name + "/gradients"):
      grads = tape.gradient(loss_value, var_list)
    
    print("GRADS",grads)
    print("WATCHED VARS", tape.watched_variables())

    grads_and_vars = list(zip(grads, var_list))
    optimizer._assert_valid_dtypes([
        v for g, v in grads_and_vars
        if g is not None and v.dtype != dtypes.resource
    ])

    return grads_and_vars


def check_ckpt_instantiation(model,ckpt):
    #ckpt is the ckpt filename
    varpreckpt = copy.deepcopy(model.variables)
    
    checkpoint_tf = tf.train.Checkpoint(model=model)
    status = checkpoint_tf.restore(ckpt)
    
    varpostckpt = copy.deepcopy(model.variables)
    
    for i in range(len(varpreckpt)):
        print((varpreckpt[i]-varpostckpt[i]).numpy().sum())


#n = 0
#for step, y_batch_train in enumerate(train_dataset):
#    n += 1
#    print(y_batch_train.keys())
#    if n == 3:
#        break





"""
logging_hook = tf.train.LoggingTensorHook({"loss" : loss, 
    "accuracy" : accuracy}, every_n_iter=10)

# Rest of the function

return tf.estimator.EstimatorSpec(
    ...params...
    training_hooks = [logging_hook])
"""