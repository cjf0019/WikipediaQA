# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:35:41 2020

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


import absl
from absl import app
import sys
import numpy as np
import tensorflow as tf
import tensorflow_text

!pip install -q tf-nightly --user
output_dir = "C:\\Users\\cfavr\\Documents\\Python Scripts\\QA"
bert_config_file = "bert_config.json"
vocab_file = "vocab-nq.txt"

init_checkpoint = "./tf_ckpt/bert_joint.ckpt"
do_lower_case = True
max_seq_length = 384
doc_stride = 128
max_query_length = 64
do_train = False
do_predict = True
train_batch_size = 32
predict_batch_size = 8
learning_rate = 5e-5
num_train_epochs = 3.0
warmup_proportion = 0.1
save_checkpoints_steps = 1000
iterations_per_loop = 1000
n_best_size = 20
verbosity = 1
max_answer_length = 30
include_unknowns = -1.0
use_tpu = False
skip_nested_contexts = True
task_id = 0
max_contexts = 48
max_position = 50
logtostderr = True
undefok = True
use_one_hot_embeddings = False
BERT = True    ### If using BERT model, else do Albert model

if BERT:
    import bert_tokenization as tokenization
else:
    import albert_tokenization as tokenization

def instantiate_full_tokenizer(vocab_file=None,do_lower_case=True,bert_yn=False):
    if not BERT:
        albert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/2",
                                          trainable=False)
        sp_model_file = albert_layer.resolved_object.sp_model_file.asset_path.numpy()
        
        tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case,spm_model_file=sp_model_file)

    else:
        tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    return tokenizer


eval_examples = read_nq_examples(
    input_file=predict_file, is_training=False)

print("predict_file", predict_file)

eval_writer = FeatureWriter(
    filename=os.path.join(output_dir, "eval.tf_record"),
    is_training=False)
eval_features = []

def append_feature(feature):
  eval_features.append(feature)
  eval_writer.process_feature(feature)

num_spans_to_ids = convert_examples_to_features(
    examples=eval_examples,
    tokenizer=tokenizer,
    is_training=False,
    output_fn=append_feature)
eval_writer.close()
eval_filename = eval_writer.filename


