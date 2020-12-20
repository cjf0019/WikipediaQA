# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 08:41:13 2020

@author: cfavr
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # dont use GPU for now
import random
import re

import enum

os.chdir("C:\\Users\\cfavr\\Documents\\Python Scripts\\QA")

import tensorflow_hub as hub
import bert_modeling as modeling
import bert_optimization as optimization
#import albert_tokenization as tokenization
import sentencepiece as spm
import tensorflow_model_optimization as tfmot



import absl
from absl import app
import sys
import numpy as np
import tensorflow as tf
import tensorflow_text

#!pip install -q tf-nightly --user

from QA_utilities import *
from QA_modeling import *


main_dir = "C:\\Users\\cfavr\\Documents\\Python Scripts\\QA"
#output_dir = main_dir + "\\ckptcp"
output_dir = "ckptcp"
bert_config_file = "bert_config.json"
vocab_file = "vocab-nq.txt"
train_precomputed_file = main_dir + "\\traintestsamples\\nq-train.tfrecords"
test_precomputed_file = main_dir + "\\traintestsamples\\nq-test.tfrecords"


do_lower_case = True
max_seq_length = 512
doc_stride = 128
max_query_length = 64
do_train = True
do_predict = False
make_tf_records = False
train_batch_size = 1
predict_batch_size = 8
learning_rate = 5e-5
num_train_epochs = 3.0
warmup_proportion = 0.1
save_checkpoints_steps = 1000
iterations_per_loop = 1000
train_num_precomputed = 300000
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
BERT = False    ### If using BERT model, else do Albert model
check_ckpt=False
init_checkpoint = output_dir+"\\model_cpkt-1" if BERT else None


if BERT:
    import bert_tokenization as tokenization
else:
    import albert_tokenization as tokenization


if __name__ == '__main__':
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    tf.io.gfile.makedirs(output_dir)
    
    ### Get different sums now!
    if check_ckpt:
        check_ckpt_instantiation(bert_model(bert_config.to_dict()),init_checkpoint)
    
    
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    
    run_config = tf.estimator.RunConfig(
      model_dir=output_dir,
       save_checkpoints_steps=save_checkpoints_steps)
    
    num_train_steps = None
    num_warmup_steps = None
    if do_train:
      num_train_features = train_num_precomputed
      num_train_steps = int(num_train_features / train_batch_size *
                            num_train_epochs)
    
    num_warmup_steps = int(num_train_steps * warmup_proportion)
    
    #model_fn = model_fn_builder(
    #    config=bert_config,
    #    init_checkpoint=init_checkpoint,
    #    learning_rate=learning_rate,
    #    num_train_steps=num_train_steps,
    #    num_warmup_steps=num_warmup_steps,
    #    use_tpu=use_tpu,
    #    use_one_hot_embeddings=use_one_hot_embeddings,\
    #        bert_yn=BERT)
    
    #estimator = tf.estimator.Estimator(
    #    model_fn=model_fn,
    #    config=run_config,
    #    params={'batch_size':train_batch_size})
    
    

    
    if do_train:
      print("***** Running training on precomputed features *****")
      print("  Num split examples = %d", num_train_features)
      print("  Batch size = %d", train_batch_size)
      print("  Num steps = %d", num_train_steps)
      train_filenames = tf.io.gfile.glob(train_precomputed_file)
      train_input_fn = input_fn_builder(
          input_file=train_filenames,
          seq_length=max_seq_length,
          batch_size=train_batch_size,
          is_training=True,
          drop_remainder=True)
    #  estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    
    
    #dict_keys(['answer_types', 'end_positions', 'input_ids', 'input_mask', 'segment_ids', 'start_positions', 'unique_ids'])
    import time
    train_dataset = train_input_fn({})
    if BERT:
       model = bert_model(bert_config.to_dict())
    else:
       model = albert_model()
       
       
    optimizer = create_optimizer(learning_rate,
                        num_train_steps,
                        num_warmup_steps)   
    
    # Prepare the metrics.
    train_acc_metric = tf.keras.metrics.Mean()
    #val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    def train_loop(epochs,train_dataset):
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()
        
            # Iterate over the batches of the dataset.
            for step, example in enumerate(train_dataset):
                unique_ids = example["unique_ids"]
                input_ids = example["input_ids"]
                input_mask = example["input_mask"]
                segment_ids = example["segment_ids"]
                
                start_positions = example["start_positions"]
                end_positions = example["end_positions"]
                answer_types = example["answer_types"]
                #y_true = (unique_ids,start_positions,end_positions,answer_types)
                y_true = (start_positions,end_positions,answer_types)

                with tf.GradientTape() as tape:       
                    y_pred = model((unique_ids,input_ids,input_mask,segment_ids))
                    loss = total_loss(y_true,y_pred)
        
                var_list = model.trainable_variables
        
                grad = tape.gradient(loss, var_list)
                grads_and_vars = list(zip(grad, var_list))
                
                train_op = optimizer.apply_gradients(grads_and_vars)
        
                # Update training metric.
                train_acc_metric.update_state(loss)
        
                # Log every 200 batches.
                if step % 200 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss))
                    )
                    print("Seen so far: %d samples" % ((step + 1) * 64))
        
            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))
        
            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()
        
    def validation_loop(val_dataset):
        #### NOT DONE YET
        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))
    
    
    print("BEFORE TRAIN")
    train_loop(1,train_dataset)
    print("AFTER TRAIN")
    
    
    
    if do_predict:
      if not output_prediction_file:
        raise ValueError(
            "--output_prediction_file must be defined in predict mode.")
    
      #eval_examples = read_nq_examples(
      #    input_file=predict_file, is_training=False)
    
      #print("predict_file", predict_file)
    
      #eval_writer = FeatureWriter(
      #    filename=os.path.join(output_dir, "eval.tf_record"),
      #    is_training=False)
      #eval_features = []
    
      #def append_feature(feature):
      #  eval_features.append(feature)
      #  eval_writer.process_feature(feature)
    
      #num_spans_to_ids = convert_examples_to_features(
      #    examples=eval_examples,
      #    tokenizer=tokenizer,
      #    is_training=False,
      #    output_fn=append_feature)
      #eval_writer.close()
      #eval_filename = eval_writer.filename
    
      print("***** Running predictions *****")
      print(f"  Num orig examples = %d" % len(eval_examples))
      print(f"  Num split examples = %d" % len(eval_features))
      print(f"  Batch size = %d" % predict_batch_size)
      for spans, ids in num_spans_to_ids.items():
        print(f"  Num split into %d = %d" % (spans, len(ids)))
    
      predict_input_fn = input_fn_builder(
          input_file=test_precomputed_file,
          seq_length=max_seq_length,
          is_training=False,
          drop_remainder=False)
    
      print(eval_filename)
    
      # If running eval on the TPU, you will need to specify the number of steps.
      all_results = []
    
      for result in estimator.predict(
          predict_input_fn, yield_single_examples=True):
        if len(all_results) % 1000 == 0:
          print("Processing example: %d" % (len(all_results)))
    
        unique_id = int(result["unique_ids"])
        start_logits = [float(x) for x in result["start_logits"].flat]
        end_logits = [float(x) for x in result["end_logits"].flat]
        answer_type_logits = [float(x) for x in result["answer_type_logits"].flat]
    
        all_results.append(
            RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits,
                answer_type_logits=answer_type_logits))
    
      print ("Going to candidates file")
    
      candidates_dict = read_candidates(predict_file)
    
      print ("setting up eval features")
    
      raw_dataset = tf.data.TFRecordDataset(eval_filename)
      eval_features = []
      for raw_record in raw_dataset:
          eval_features.append(tf.train.Example.FromString(raw_record.numpy()))
    
      print ("compute_pred_dict")
    
      nq_pred_dict = compute_pred_dict(candidates_dict, eval_features,
                                       [r._asdict() for r in all_results])
      predictions_json = {"predictions": list(nq_pred_dict.values())}
    
      print ("writing json")
    
      with tf.io.gfile.GFile(output_prediction_file, "w") as f:
        json.dump(predictions_json, f, indent=4)
