# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 13:56:33 2020

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


reader = tf.train.load_checkpoint('./tf_ckpt/bert_joint.ckpt')
shape_from_key = reader.get_variable_to_shape_map()
dtype_from_key = reader.get_variable_to_dtype_map()



TextSpan = collections.namedtuple("TextSpan", "token_positions text")


class AnswerType(enum.IntEnum):
  """Type of NQ answer."""
  UNKNOWN = 0
  YES = 1
  NO = 2
  SHORT = 3
  LONG = 4


class Answer(collections.namedtuple("Answer", ["type", "text", "offset"])):
  """Answer record.

  An Answer contains the type of the answer and possibly the text (for
  long) as well as the offset (for extractive).
  """

  def __new__(cls, type_, text=None, offset=None):
    return super(Answer, cls).__new__(cls, type_, text, offset)


class NqExample(object):
  """A single training/test example."""

  def __init__(self,
               example_id,
               qas_id,
               questions,
               doc_tokens,
               doc_tokens_map=None,
               answer=None,
               start_position=None,
               end_position=None):
    self.example_id = example_id
    self.qas_id = qas_id
    self.questions = questions
    self.doc_tokens = doc_tokens
    self.doc_tokens_map = doc_tokens_map
    self.answer = answer
    self.start_position = start_position
    self.end_position = end_position


def has_long_answer(a):
  return (a["long_answer"]["start_token"] >= 0 and
          a["long_answer"]["end_token"] >= 0)


def should_skip_context(e, idx):
  if (skip_nested_contexts and
      not e["long_answer_candidates"][idx]["top_level"]):
    return True
  elif not get_candidate_text(e, idx).text.strip():
    # Skip empty contexts.
    return True
  else:
    return False


def get_first_annotation(e):
  """Returns the first short or long answer in the example.

  Args:
    e: (dict) annotated example.

  Returns:
    annotation: (dict) selected annotation
    annotated_idx: (int) index of the first annotated candidate.
    annotated_sa: (tuple) char offset of the start and end token
        of the short answer. The end token is exclusive.
  """

  if "annotations" not in e:
      return None, -1, (-1, -1)

  positive_annotations = sorted(
      [a for a in e["annotations"] if has_long_answer(a)],
      key=lambda a: a["long_answer"]["candidate_index"])

  for a in positive_annotations:
    if a["short_answers"]:
      idx = a["long_answer"]["candidate_index"]
      start_token = a["short_answers"][0]["start_token"]
      end_token = a["short_answers"][-1]["end_token"]
      return a, idx, (token_to_char_offset(e, idx, start_token),
                      token_to_char_offset(e, idx, end_token) - 1)

  for a in positive_annotations:
    idx = a["long_answer"]["candidate_index"]
    return a, idx, (-1, -1)

  return None, -1, (-1, -1)


def get_text_span(example, span):
  """Returns the text in the example's document in the given token span."""
  token_positions = []
  tokens = []
  for i in range(span["start_token"], span["end_token"]):
    t = example["document_tokens"][i]
    if not t["html_token"]:
      token_positions.append(i)
      token = t["token"].replace(" ", "")
      tokens.append(token)
  return TextSpan(token_positions, " ".join(tokens))


def token_to_char_offset(e, candidate_idx, token_idx):
  """Converts a token index to the char offset within the candidate."""
  c = e["long_answer_candidates"][candidate_idx]
  char_offset = 0
  for i in range(c["start_token"], token_idx):
    t = e["document_tokens"][i]
    if not t["html_token"]:
      token = t["token"].replace(" ", "")
      char_offset += len(token) + 1
  return char_offset


def get_candidate_type(e, idx):
  """Returns the candidate's type: Table, Paragraph, List or Other."""
  c = e["long_answer_candidates"][idx]
  first_token = e["document_tokens"][c["start_token"]]["token"]
  if first_token == "<Table>":
    return "Table"
  elif first_token == "<P>":
    return "Paragraph"
  elif first_token in ("<Ul>", "<Dl>", "<Ol>"):
    return "List"
  elif first_token in ("<Tr>", "<Li>", "<Dd>", "<Dt>"):
    return "Other"
  else:
    absl.logging.warning("Unknoww candidate type found: %s", first_token)
    return "Other"


def add_candidate_types_and_positions(e):
  """Adds type and position info to each candidate in the document."""
  counts = collections.defaultdict(int)
  for idx, c in candidates_iter(e):
    context_type = get_candidate_type(e, idx)
    if counts[context_type] < max_position:
      counts[context_type] += 1
    c["type_and_position"] = "[%s=%d]" % (context_type, counts[context_type])


def get_candidate_type_and_position(e, idx):
  """Returns type and position info for the candidate at the given index."""
  if idx == -1:
    return "[NoLongAnswer]"
  else:
    return e["long_answer_candidates"][idx]["type_and_position"]


def get_candidate_text(e, idx):
  """Returns a text representation of the candidate at the given index."""
  # No candidate at this index.
  if idx < 0 or idx >= len(e["long_answer_candidates"]):
    return TextSpan([], "")

  # This returns an actual candidate.
  return get_text_span(e, e["long_answer_candidates"][idx])


def candidates_iter(e):
  """Yield's the candidates that should not be skipped in an example."""
  for idx, c in enumerate(e["long_answer_candidates"]):
    if should_skip_context(e, idx):
      continue
    yield idx, c


def create_example_from_jsonl(line):
  """Creates an NQ example from a given line of JSON.
     Line contains: document_text
     annotation: has long and short answer candidates
     
  """
  print("INSIDE CREATE EXAMPLE")
  e = json.loads(line, object_pairs_hook=collections.OrderedDict)
  document_tokens = e["document_text"].split(" ")
  e["document_tokens"] = []
  for token in document_tokens:
      e["document_tokens"].append({"token":token, "start_byte":-1, "end_byte":-1, "html_token":"<" in token})

  add_candidate_types_and_positions(e)
  annotation, annotated_idx, annotated_sa = get_first_annotation(e)
  print("ANNOTATION",annotation)
  # annotated_idx: index of the first annotated context, -1 if null.
  # annotated_sa: short answer start and end char offsets, (-1, -1) if null.
  question = {"input_text": e["question_text"]}
  answer = {
      "candidate_id": annotated_idx,
      "span_text": "",
      "span_start": -1,
      "span_end": -1,
      "input_text": "long",
  }
  print("ANSWER",answer)
  # Yes/no answers are added in the input text.
  if annotation is not None:
    assert annotation["yes_no_answer"] in ("YES", "NO", "NONE")
    if annotation["yes_no_answer"] in ("YES", "NO"):
      answer["input_text"] = annotation["yes_no_answer"].lower()

  # Add a short answer if one was found.
  if annotated_sa != (-1, -1):
    answer["input_text"] = "short"
    span_text = get_candidate_text(e, annotated_idx).text
    answer["span_text"] = span_text[annotated_sa[0]:annotated_sa[1]]
    answer["span_start"] = annotated_sa[0]
    answer["span_end"] = annotated_sa[1]
    expected_answer_text = get_text_span(
        e, {
            "start_token": annotation["short_answers"][0]["start_token"],
            "end_token": annotation["short_answers"][-1]["end_token"],
        }).text
    assert expected_answer_text == answer["span_text"], (expected_answer_text,
                                                         answer["span_text"])

  # Add a long answer if one was found.
  elif annotation and annotation["long_answer"]["candidate_index"] >= 0:
    answer["span_text"] = get_candidate_text(e, annotated_idx).text
    answer["span_start"] = 0
    answer["span_end"] = len(answer["span_text"])

  context_idxs = [-1]
  context_list = [{"id": -1, "type": get_candidate_type_and_position(e, -1)}]
  context_list[-1]["text_map"], context_list[-1]["text"] = (
      get_candidate_text(e, -1))
  for idx, _ in candidates_iter(e):
    context = {"id": idx, "type": get_candidate_type_and_position(e, idx)}
    context["text_map"], context["text"] = get_candidate_text(e, idx)
    context_idxs.append(idx)
    context_list.append(context)
    if len(context_list) >= max_contexts:
      break

  if "document_title" not in e:
      e["document_title"] = e["example_id"]

  # Assemble example.
  example = {
      "name": e["document_title"],
      "id": str(e["example_id"]),
      "questions": [question],
      "answers": [answer],
      "has_correct_context": annotated_idx in context_idxs
  }

  single_map = []
  single_context = []
  offset = 0
  for context in context_list:
    single_map.extend([-1, -1])
    single_context.append("[ContextId=%d] %s" %
                          (context["id"], context["type"]))
    offset += len(single_context[-1]) + 1
    if context["id"] == annotated_idx:
      answer["span_start"] += offset
      answer["span_end"] += offset

    # Many contexts are empty once the HTML tags have been stripped, so we
    # want to skip those.
    if context["text"]:
      single_map.extend(context["text_map"])
      single_context.append(context["text"])
      offset += len(single_context[-1]) + 1

  example["contexts"] = " ".join(single_context)
  example["contexts_map"] = single_map
  if annotated_idx in context_idxs:
    expected = example["contexts"][answer["span_start"]:answer["span_end"]]

    # This is a sanity check to ensure that the calculated start and end
    # indices match the reported span text. If this assert fails, it is likely
    # a bug in the data preparation code above.
    assert expected == answer["span_text"], (expected, answer["span_text"])

  return example


def make_nq_answer(contexts, answer):
  """Makes an Answer object following NQ conventions.

  Args:
    contexts: string containing the context
    answer: dictionary with `span_start` and `input_text` fields

  Returns:
    an Answer object. If the Answer type is YES or NO or LONG, the text
    of the answer is the long answer. If the answer type is UNKNOWN, the text of
    the answer is empty.
  """
  start = answer["span_start"]
  end = answer["span_end"]
  input_text = answer["input_text"]

  if (answer["candidate_id"] == -1 or start >= len(contexts) or
      end > len(contexts)):
    answer_type = AnswerType.UNKNOWN
    start = 0
    end = 1
  elif input_text.lower() == "yes":
    answer_type = AnswerType.YES
  elif input_text.lower() == "no":
    answer_type = AnswerType.NO
  elif input_text.lower() == "long":
    answer_type = AnswerType.LONG
  else:
    answer_type = AnswerType.SHORT

  return Answer(answer_type, text=contexts[start:end], offset=start)


def read_nq_entry(entry, is_training):
  """Converts a NQ entry into a list of NqExamples."""

  def is_whitespace(c):
    return c in " \t\r\n" or ord(c) == 0x202F

  print("*****INSIDE NQ ENTRY*******")
  examples = []
  contexts_id = entry["id"]
  contexts = entry["contexts"]
  doc_tokens = []
  char_to_word_offset = []
  prev_is_whitespace = True
  print(contexts)
  for c in contexts:
    if is_whitespace(c):
      prev_is_whitespace = True
    else:
      if prev_is_whitespace:
        doc_tokens.append(c)
      else:
        doc_tokens[-1] += c
      prev_is_whitespace = False
    char_to_word_offset.append(len(doc_tokens) - 1)

  questions = []
  for i, question in enumerate(entry["questions"]):
    qas_id = "{}".format(contexts_id)
    question_text = question["input_text"]
    start_position = None
    end_position = None
    answer = None
    if is_training:
      answer_dict = entry["answers"][i]
      print("ANSWERDICT",answer_dict)
      answer = make_nq_answer(contexts, answer_dict)

      # For now, only handle extractive, yes, and no.
      if answer is None or answer.offset is None:
        continue
      start_position = char_to_word_offset[answer.offset]
      end_position = char_to_word_offset[answer.offset + len(answer.text) - 1]

      # Only add answers where the text can be exactly recovered from the
      # document. If this CAN'T happen it's likely due to weird Unicode
      # stuff so we will just skip the example.
      #
      # Note that this means for training mode, every example is NOT
      # guaranteed to be preserved.
      actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
      cleaned_answer_text = " ".join(
          tokenization.whitespace_tokenize(answer.text))
      if actual_text.find(cleaned_answer_text) == -1:
        absl.logging.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
        continue

    questions.append(question_text)
    example = NqExample(
        example_id=int(contexts_id),
        qas_id=qas_id,
        questions=questions[:],
        doc_tokens=doc_tokens,
        doc_tokens_map=entry.get("contexts_map", None),
        answer=answer,
        start_position=start_position,
        end_position=end_position)
    examples.append(example)
    print("EXAMPLES",examples)
  return examples


def convert_examples_to_features(examples, tokenizer, is_training, output_fn):
  """Converts a list of NqExamples into InputFeatures."""
  print("INSIDE CONVERT EXAMPLES TO FEATURES")
  num_spans_to_ids = collections.defaultdict(list)

  for example in examples:
    example_index = example.example_id
    features = convert_single_example(example, tokenizer, is_training)
    num_spans_to_ids[len(features)].append(example.qas_id)

    for feature in features:
      feature.example_index = example_index
      feature.unique_id = feature.example_index + feature.doc_span_index
      output_fn(feature)

  return num_spans_to_ids


def check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""
  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


def convert_single_example(example, tokenizer, is_training):
  """Converts a single NqExample into a list of InputFeatures."""
  tok_to_orig_index = []
  orig_to_tok_index = []
  all_doc_tokens = []
  features = []
  for (i, token) in enumerate(example.doc_tokens):
    orig_to_tok_index.append(len(all_doc_tokens))
    sub_tokens = tokenize(tokenizer, token)
    tok_to_orig_index.extend([i] * len(sub_tokens))
    all_doc_tokens.extend(sub_tokens)

  # `tok_to_orig_index` maps wordpiece indices to indices of whitespace
  # tokenized word tokens in the contexts. The word tokens might themselves
  # correspond to word tokens in a larger document, with the mapping given
  # by `doc_tokens_map`.
  if example.doc_tokens_map:
    tok_to_orig_index = [
        example.doc_tokens_map[index] for index in tok_to_orig_index
    ]

  # QUERY
  query_tokens = []
  query_tokens.append("[Q]")
  query_tokens.extend(tokenize(tokenizer, example.questions[-1]))
  if len(query_tokens) > max_query_length:
    query_tokens = query_tokens[-max_query_length:]

  # ANSWER
  tok_start_position = 0
  tok_end_position = 0
  if is_training:
    tok_start_position = orig_to_tok_index[example.start_position]
    if example.end_position < len(example.doc_tokens) - 1:
      tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
    else:
      tok_end_position = len(all_doc_tokens) - 1

  # The -3 accounts for [CLS], [SEP] and [SEP]
  max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

  # We can have documents that are longer than the maximum sequence length.
  # To deal with this we do a sliding window approach, where we take chunks
  # of up to our max length with a stride of `doc_stride`.
  _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
      "DocSpan", ["start", "length"])
  doc_spans = []
  start_offset = 0
  while start_offset < len(all_doc_tokens):
    length = len(all_doc_tokens) - start_offset
    length = min(length, max_tokens_for_doc)
    doc_spans.append(_DocSpan(start=start_offset, length=length))
    if start_offset + length == len(all_doc_tokens):
      break
    start_offset += min(length, doc_stride)

  for (doc_span_index, doc_span) in enumerate(doc_spans):
    tokens = []
    token_to_orig_map = {}
    token_is_max_context = {}
    segment_ids = []
    ### Add in the query at the front
    tokens.append("[CLS]")
    segment_ids.append(0)
    tokens.extend(query_tokens)
    segment_ids.extend([0] * len(query_tokens))
    tokens.append("[SEP]")
    segment_ids.append(0)

    for i in range(doc_span.length):
      split_token_index = doc_span.start + i
      token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

      is_max_context = check_is_max_context(doc_spans, doc_span_index,
                                            split_token_index)
      token_is_max_context[len(tokens)] = is_max_context
      tokens.append(all_doc_tokens[split_token_index])
      segment_ids.append(1)
    tokens.append("[SEP]")    #[SEP] gets added at the end of each stride???
    segment_ids.append(1)
    assert len(tokens) == len(segment_ids)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    start_position = None
    end_position = None
    answer_type = None
    answer_text = ""
    if is_training:
      doc_start = doc_span.start
      doc_end = doc_span.start + doc_span.length - 1
      # For training, if our document chunk does not contain an annotation
      # we throw it out, since there is nothing to predict.
      contains_an_annotation = (
          tok_start_position >= doc_start and tok_end_position <= doc_end)
      if ((not contains_an_annotation) or
          example.answer.type == AnswerType.UNKNOWN):
        # If an example has unknown answer type or does not contain the answer
        # span, then we only include it with probability --include_unknowns.
        # When we include an example with unknown answer type, we set the first
        # token of the passage to be the annotated short span.
        if (include_unknowns < 0 or
            random.random() > include_unknowns):
          continue
        start_position = 0
        end_position = 0
        answer_type = AnswerType.UNKNOWN
      else:
        doc_offset = len(query_tokens) + 2
        start_position = tok_start_position - doc_start + doc_offset
        end_position = tok_end_position - doc_start + doc_offset
        answer_type = example.answer.type

      answer_text = " ".join(tokens[start_position:(end_position + 1)])

    feature = InputFeatures(
        unique_id=-1,
        example_index=-1,
        doc_span_index=doc_span_index,
        tokens=tokens,
        token_to_orig_map=token_to_orig_map,
        token_is_max_context=token_is_max_context,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        start_position=start_position,
        end_position=end_position,
        answer_text=answer_text,
        answer_type=answer_type)

    features.append(feature)
    print("ONEFEATURE",feature)

  return features


# A special token in NQ is made of non-space chars enclosed in square brackets.
_SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)


def tokenize(tokenizer, text, apply_basic_tokenization=False):
  """Tokenizes text, optionally looking up special tokens separately.

  Args:
    tokenizer: a tokenizer from bert.tokenization.FullTokenizer
    text: text to tokenize
    apply_basic_tokenization: If True, apply the basic tokenization. If False,
      apply the full tokenization (basic + wordpiece).

  Returns:
    tokenized text.

  A special token is any text with no spaces enclosed in square brackets with no
  space, so we separate those out and look them up in the dictionary before
  doing actual tokenization.
  """
  tokenize_fn = tokenizer.tokenize
  if apply_basic_tokenization:
    tokenize_fn = tokenizer.basic_tokenizer.tokenize
  tokens = []
  for token in text.split(" "):
    if _SPECIAL_TOKENS_RE.match(token):
      if token in tokenizer.vocab:
        tokens.append(token)
      else:
        #tokens.append(tokenizer.wordpiece_tokenizer.unk_token)
        tokens.append('<unk>')
    else:
      tokens.extend(tokenize_fn(token))
  return tokens


class CreateTFExampleFn(object):
  """Functor for creating NQ tf.Examples."""

  def __init__(self, is_training, bert_yn):
    self.is_training = is_training
    self.tokenizer = instantiate_full_tokenizer(vocab_file=vocab_file,do_lower_case=do_lower_case,\
                                                bert_yn=bert_yn)

  def process(self, example):
    """Coverts an NQ example in a list of serialized tf examples."""
    nq_examples = read_nq_entry(example, self.is_training)
    input_features = []
    for nq_example in nq_examples:
      input_features.extend(
          convert_single_example(nq_example, self.tokenizer, self.is_training))

    for input_feature in input_features:
      input_feature.example_index = int(example["id"])
      input_feature.unique_id = (
          input_feature.example_index + input_feature.doc_span_index)

      def create_int_feature(values):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=list(values)))

      features = collections.OrderedDict()
      features["unique_ids"] = create_int_feature([input_feature.unique_id])
      features["input_ids"] = create_int_feature(input_feature.input_ids)
      features["input_mask"] = create_int_feature(input_feature.input_mask)
      features["segment_ids"] = create_int_feature(input_feature.segment_ids)

      if self.is_training:
        features["start_positions"] = create_int_feature(
            [input_feature.start_position])
        features["end_positions"] = create_int_feature(
            [input_feature.end_position])
        features["answer_types"] = create_int_feature(
            [input_feature.answer_type])
      else:
        token_map = [-1] * len(input_feature.input_ids)
        for k, v in input_feature.token_to_orig_map.items():
          token_map[k] = v
        features["token_map"] = create_int_feature(token_map)

      yield tf.train.Example(features=tf.train.Features(
          feature=features)).SerializeToString()


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tokens,
               token_to_orig_map,
               token_is_max_context,
               input_ids,
               input_mask,
               segment_ids,
               start_position=None,
               end_position=None,
               answer_text="",
               answer_type=AnswerType.SHORT):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tokens = tokens
    self.token_to_orig_map = token_to_orig_map
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.start_position = start_position
    self.end_position = end_position
    self.answer_text = answer_text
    self.answer_type = answer_type


def read_nq_examples(input_file, is_training):
  """Read a NQ json file into a list of NqExample."""
  input_paths = tf.io.gfile.glob(input_file)
  input_data = []

  def _open(path):
    if path.endswith(".gz"):
      return gzip.GzipFile(fileobj=tf.io.gfile.GFile(path, "rb"))
    else:
      return tf.io.gfile.GFile(path, "r")

  for path in input_paths:
    absl.logging.info("Reading: %s", path)
    with _open(path) as input_file:
      for index, line in enumerate(input_file):
        input_data.append(create_example_from_jsonl(line))
        # if index > 100:
        #     break

  examples = []
  for entry in input_data:
    examples.extend(read_nq_entry(entry, is_training))
  return examples


def create_model(bert_config, is_training, use_one_hot_embeddings, init_checkpoint = None):
  """Creates a classification model."""

  print("CONFIGDICT",bert_config.to_dict())
    
  model = modeling.mk_model(bert_config.to_dict())


  return model




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
      loss_value = loss()
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




RawResult = collections.namedtuple(
    "RawResult",
    ["unique_id", "start_logits", "end_logits", "answer_type_logits"])


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)

    if self.is_training:
      features["start_positions"] = create_int_feature([feature.start_position])
      features["end_positions"] = create_int_feature([feature.end_position])
      features["answer_types"] = create_int_feature([feature.answer_type])
    else:
      token_map = [-1] * len(feature.input_ids)
      for k, v in feature.token_to_orig_map.items():
        token_map[k] = v
      features["token_map"] = create_int_feature(token_map)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


Span = collections.namedtuple("Span", ["start_token_idx", "end_token_idx"])


class EvalExample(object):
  """Eval data available for a single example."""

  def __init__(self, example_id, candidates):
    self.example_id = example_id
    self.candidates = candidates
    self.results = {}
    self.features = {}


class ScoreSummary(object):

  def __init__(self):
    self.predicted_label = None
    self.short_span_score = None
    self.cls_token_score = None
    self.answer_type_logits = None


def read_candidates_from_one_split(input_path):
  """Read candidates from a single jsonl file."""
  candidates_dict = {}
  if input_path.endswith(".gz"):
    with gzip.GzipFile(fileobj=tf.io.gfile.GFile(input_path, "rb")) as input_file:
      absl.logging.info("Reading examples from: %s", input_path)
      for index, line in enumerate(input_file):
        e = json.loads(line)
        candidates_dict[e["example_id"]] = e["long_answer_candidates"]
        # if index > 100:
        #   break
  else:
    with tf.io.gfile.GFile(input_path, "r") as input_file:
      absl.logging.info("Reading examples from: %s", input_path)
      for index, line in enumerate(input_file):
        e = json.loads(line)
        candidates_dict[e["example_id"]] = e["long_answer_candidates"]
        # if index > 100:
        #   break


  return candidates_dict


def read_candidates(input_pattern):
  """Read candidates with real multiple processes."""
  input_paths = tf.io.gfile.glob(input_pattern)
  final_dict = {}
  for input_path in input_paths:
    final_dict.update(read_candidates_from_one_split(input_path))
  return final_dict


def get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(
      enumerate(logits[1:], 1), key=lambda x: x[1], reverse=True)
  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def compute_predictions(example):
  """Converts an example into an NQEval object for evaluation."""
  predictions = []
  n_best_size = 10
  max_answer_length = 30

  for unique_id, result in example.results.items():
    if unique_id not in example.features:
      raise ValueError("No feature found with unique_id:", unique_id)
    token_map = example.features[unique_id]["token_map"].int64_list.value
    start_indexes = get_best_indexes(result["start_logits"], n_best_size)
    end_indexes = get_best_indexes(result["end_logits"], n_best_size)
    for start_index in start_indexes:
      for end_index in end_indexes:
        if end_index < start_index:
          continue
        if token_map[start_index] == -1:
          continue
        if token_map[end_index] == -1:
          continue
        length = end_index - start_index + 1
        if length > max_answer_length:
          continue
        summary = ScoreSummary()
        summary.short_span_score = (
            result["start_logits"][start_index] +
            result["end_logits"][end_index])
        summary.cls_token_score = (
            result["start_logits"][0] + result["end_logits"][0])
        summary.answer_type_logits = result["answer_type_logits"]
        start_span = token_map[start_index]
        end_span = token_map[end_index] + 1

        # Span logits minus the cls logits seems to be close to the best.
        score = summary.short_span_score - summary.cls_token_score
        predictions.append((score, summary, start_span, end_span))

  short_span = Span(-1, -1)
  long_span = Span(-1, -1)
  score = 0
  summary = ScoreSummary()
  if predictions:
    score, summary, start_span, end_span = sorted(predictions, reverse=True)[0]
    short_span = Span(start_span, end_span)
    for c in example.candidates:
      start = short_span.start_token_idx
      end = short_span.end_token_idx
      if c["top_level"] and c["start_token"] <= start and c["end_token"] >= end:
        long_span = Span(c["start_token"], c["end_token"])
        break

  summary.predicted_label = {
      "example_id": example.example_id,
      "long_answer": {
          "start_token": long_span.start_token_idx,
          "end_token": long_span.end_token_idx,
          "start_byte": -1,
          "end_byte": -1
      },
      "long_answer_score": score,
      "short_answers": [{
          "start_token": short_span.start_token_idx,
          "end_token": short_span.end_token_idx,
          "start_byte": -1,
          "end_byte": -1
      }],
      "short_answers_score": score,
      "yes_no_answer": "NONE"
  }

  return summary


def compute_pred_dict(candidates_dict, dev_features, raw_results):
  """Computes official answer key from raw logits."""

  raw_results_by_id = {int(res["unique_id"]):res for res in raw_results}

  # Cast example id to int32 for each example, similarly to the raw results.
  all_candidates = candidates_dict.items()
  example_ids = tf.cast(np.array([int(k) for k, _ in all_candidates]), dtype=tf.int32).numpy()
  examples_by_id = dict(zip(example_ids, all_candidates))

  # Cast unique_id also to int32 for features.
  feature_ids = []
  features = []
  for f in dev_features:
    feature_ids.append(f.features.feature["unique_ids"].int64_list.value[0])
    features.append(f.features.feature)
  feature_ids = tf.cast(np.array(feature_ids), dtype=tf.int32).numpy()
  features_by_id = dict(zip(feature_ids, features))

  # Join examplew with features and raw results.
  examples = []
    
  for example_id in examples_by_id:
    example = examples_by_id[example_id]
    examples.append(EvalExample(example[0], example[1]))
    examples[-1].features[example_id] = features_by_id[example_id]
    examples[-1].results[example_id] = raw_results_by_id[example_id]
    
  # Construct prediction objects.
  summary_dict = {}
  nq_pred_dict = {}
  for e in examples:
    summary = compute_predictions(e)
    summary_dict[e.example_id] = summary
    nq_pred_dict[e.example_id] = summary.predicted_label
    if len(nq_pred_dict) % 100 == 0:
      print("Examples processed: %d" % len(nq_pred_dict))

  return nq_pred_dict

def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  if not do_train and not do_predict:
    raise ValueError("At least one of `{do_train,do_predict}` must be True.")

  if do_train:
    if not train_precomputed_file:
      raise ValueError("If `do_train` is True, then `train_precomputed_file` "
                       "must be specified.")
    if not train_num_precomputed:
      raise ValueError("If `do_train` is True, then `train_num_precomputed` "
                       "must be specified.")

  if do_predict:
    if not predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")

  if max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (max_seq_length, bert_config.max_position_embeddings))

  if max_seq_length <= max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (max_seq_length, max_query_length))





#jsonfile = 'simplified-nq-test.jsonl'
jsonfile = 'D:\\NLPDatasets\\QA\\simplified-nq-train.jsonl'
readjson = open(jsonfile)
line = readjson.readline()
x = create_example_from_jsonl(line)

"""
create_example_from_jsonl makes dictionary with name,id,questions,
answers,has_correct_context,contexts,contexts_map

*** the contexts come from the long_answer_candidates, which is a list
   of ordered dicts, giving start and end tokens for the contexts
   in contexts_map -1 seems to delineate a new context/paragraph
   ([ContextId][Paragraph])



**** IMPORTANT NOTE **** The long_answer_candidates are already broken out
     in the original json file! A more expanded work would need to generate these
"""

nqentry = read_nq_entry(x,False)

"""
read_nq_entry takes in the dictionary and converts it into a class

Properties include doc_tokens, which is just a single list of all tokens
 including the special context and paragraph indicators
 
 Includes additional properties for start and end position (for the answer?)
 The answer, the questions, and any ids

"""

albert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/2",
                                  trainable=False)
sp_model_file = albert_layer.resolved_object.sp_model_file.asset_path.numpy()

tokenizer = tokenization.FullTokenizer(
vocab_file=vocab_file, do_lower_case=do_lower_case,spm_model_file=sp_model_file)

tokenizer = instantiate_full_tokenizer()

example = nqentry[0]
num_spans_to_ids = collections.defaultdict(list)
example_index = example.example_id
features = convert_single_example(example, tokenizer, False)
num_spans_to_ids[len(features)].append(example.qas_id)

"""
Convert_examples_to_features is going to take these raw QA sets and tokenize them
and convert into a numerical form which TF can use
It returns a defaultdict: key is len of sequence, and value is a list of ids
                          that have that len
                 
In convert_single_example, tokenization happens and the question and answer texts
      are merged together and processed both for BERT (add [CLS] and [SEP] tags)
      and for marking the question and answer texts ([Q] for question). 
      The question and answer texts are merged together.
      "segment_ids" runs the length of the text and is 0 for question words, 1 for text words
                  
      *** NOTE: At this point from convert_single_examples,
      it's still not in proper tf format. The input_ids
             are still integer numbers corresponding to the tokens. 
             
The last step of convert_examples... is the FeatureWriter.
This converts the (integer) inputs into tf features...
          tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
          
All of these features are then fed into a tf.Example and written to file:
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())          
          

***NOTE: The creating of features is not by default done for the train files...
         these are actually already created and stored as TFRecord files
         
NOTE ON tf.train.Example:
   is of format {"string": tf.train.Feature}
   
example_proto = tf.train.Example.FromString(serialized_example)


            
"""

eval_writer = FeatureWriter(
        filename=os.path.join(output_dir, "eval.tf_record"),
        is_training=False)
eval_features = []

def append_feature(feature):
  eval_features.append(feature)
  eval_writer.process_feature(feature)

num_spans_to_ids = convert_examples_to_features(\
    examples=eval_examples,\
    tokenizer=tokenizer,\
    is_training=True,\
    output_fn=append_feature)
eval_writer.close()
eval_filename = eval_writer.filename

vocab = []
f = open(encoder.resolved_object.vocab_file.asset_path.numpy(),encoding='utf-8')
for line in f:
    vocab.append(line.strip())
    
#x = read_nq_examples(jsonfile,False)
raw = tf.data.TFRecordDataset("nq-train.tfrecords-00000-of-00001")
### There's about 500,000 examples in TFRecordDataset

eval_features = []
for raw_record in raw.take(5):
    print("raw_record",raw_record.numpy())
    eval_features.append(tf.train.Example.FromString(raw_record.numpy()))

poop = eval_features[0]
### TO GET FEATURES: poopers = poop.ListFields()[0][1]


"""
 NOTES ON DEALING WITH TFRecordDatasets...

 
 tf.io.parse_example(
    serialized, features, example_names=None, name=None
)
    --- This function parses a number of serialized Example protos given in 
    serialized. We refer to serialized as a batch with batch_size many entries
    of individual Example protos.


TO SERIALIZE:
example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

        
Comments...
1) tf.train.Example is just a wrapper for tf.train.Features
2) tf.train.Features is given a dictionar, where the keys are feature names,
    and the values are individual features, converted into tf feature formats,
    such as tf.int64, tf.float32, tf.string
    
    Example of feed into tf.train.Features...
feature = {
    'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
}


### How to get data into dataset format straight from arrays (features are all numpy)
features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))


"""


def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)



bert_config = modeling.BertConfig.from_json_file(bert_config_file)

#validate_flags_or_throw(bert_config)
#tf.io.gfile.makedirs(output_dir)

run_config = tf.estimator.RunConfig(
  model_dir=output_dir,
   save_checkpoints_steps=save_checkpoints_steps)

num_train_steps = None
num_warmup_steps = None
#if do_train:
#    num_train_features = train_num_precomputed
#    num_train_steps = int(num_train_features / train_batch_size *
#                          num_train_epochs)

#num_warmup_steps = int(num_train_steps * warmup_proportion)



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
        print("CONFIG",config)
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
    
    seq_len = config['max_position_embeddings']
    unique_id  = tf.keras.Input(shape=(1,),dtype=tf.int64,name='unique_id')
    input_ids   = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='input_ids')
    input_mask  = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='input_mask')
    segment_ids = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='segment_ids')
    BERT = modeling.BertModel(config=config,name='bert')
    pooled_output, sequence_output = BERT(input_word_ids=input_ids,
                                          input_mask=input_mask,
                                          input_type_ids=segment_ids)
    
    ### NOTE: output of sequence_output is [batch_size,max_seq_length,768]
    
    logits = TDense(2,name='logits')(sequence_output)   ### I think it's calculating this for each token in the sequence
    start_logits,end_logits = tf.split(logits,axis=-1,num_or_size_splits= 2,name='split')
    
    ### These will also return one value per seq length... but should they?
    start_logits = tf.squeeze(start_logits,axis=-1,name='start_squeeze')
    end_logits   = tf.squeeze(end_logits,  axis=-1,name='end_squeeze')
    ### Do I need to add another layer for each that goes over the seq_length, to condense to 1?
    
    ans_type = TDense(5,name='ans_type')(pooled_output)
    return tf.keras.Model([input_ for input_ in [unique_id,input_ids,input_mask,segment_ids] 
                           if input_ is not None],
                          [unique_id,start_logits,end_logits,ans_type],
                          name='bert_model') 



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
    
    seq_len = 128
    unique_id  = tf.keras.Input(shape=(1,),dtype=tf.int64,name='unique_id')
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
    
    ans_type = TDense(5,name='ans_type')(albert_outputs['pooled_output'])
    return tf.keras.Model([input_ for input_ in [unique_id] + list(encoder_inputs.keys()) \
                           if input_ is not None],
                          [unique_id,start_logits,end_logits,ans_type],
                          name='albert') 


# Computes the loss for positions.
def compute_loss(logits, positions):
    one_hot_positions = tf.one_hot(
        tf.cast(positions,tf.int32),\
            depth=128,dtype=tf.float32)
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
    pred_id,start_logits,end_logits,answer_type_logits = y_pred
    unique_id, start_positions, end_positions, answer_types = y_true
    start_loss = compute_loss(start_logits, start_positions)
    end_loss = compute_loss(end_logits, end_positions)
    answer_type_loss = compute_label_loss(answer_type_logits, answer_types)
    total_loss = tf.add_n([start_loss,end_loss,answer_type_loss])
    return total_loss



model = albert_model()

xid = tf.constant(1,dtype=tf.int64,shape=(3,1))
xinp = tf.constant(1,dtype=tf.int32,shape=(3,128))
xmask = tf.constant(1,dtype=tf.int32,shape=(3,128))
xseg = tf.constant(1,dtype=tf.int32,shape=(3,128))

poop = model((xid,xinp,xmask,xseg))

y_true_1 = (tf.constant(1,dtype = tf.int32,shape=(3,)),\
            tf.constant(1,dtype = tf.int32,shape=(3,)),\
            tf.constant(1,dtype = tf.int32,shape=(3,)),\
            tf.constant(1,dtype = tf.int32,shape=(3,)))

y_true = []
for i in range(len(poop)):
    print(poop[i])
    #y_true.append(tf.constant(1,dtype=tf.float32,shape=tuple(tf.shape(poop[i]).numpy())))
    y_true.append(y_true_1)

y_true = y_true_1

pooped = total_loss(y_true,poop)




model.compile(
    optimizer='adam',
    loss=total_loss,
    metrics=['accuracy'],run_eagerly=True)



estimator = tf.keras.estimator.model_to_estimator(keras_model=model,model_dir=output_dir)

encoder = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3",
    trainable=True)

sp_model_file = encoder.resolved_object.sp_model_file.asset_path.numpy()

tokenizer = tokenization.FullTokenizer(
vocab_file=vocab_file, do_lower_case=do_lower_case,spm_model_file=sp_model_file)

modeltokvf = b'C:\\Users\\cfavr\\AppData\\Local\\Temp\\tfhub_modules\\590d7d0ea1d4e227b197a3512d641a1af6b36db1\\assets\\vocab.txt'

modeltokenizer = tokenization.FullTokenizer(vocab_file=modeltokvf, do_lower_case=do_lower_case)
qatokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
modelvocab = list(modeltokenizer.vocab)
qavocab = list(qatokenizer.vocab)


f = open(encoder.resolved_object.vocab_file.asset_path.numpy(),encoding='utf-8')
vocab = []
for line in f:
    vocab.append(line.strip())

vars = [var.name.replace(':0','') for var in model.variables]
ckptvars = list(shape_from_key.keys())


def model_fn_builder(config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, bert_yn=BERT):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        absl.logging.info("*** Features ***")
        for name in sorted(features.keys()):
          absl.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
    
        unique_ids = features["unique_id"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
    
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    

        if bert_yn:
            model = bert_model(bert_config.to_dict())
        else:
            model = albert_model()
        

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            model_tf = tf.keras.Model()
            checkpoint_tf = tf.train.Checkpoint(model=model_tf)
            status = checkpoint_tf.restore(init_checkpoint)
            print("CHECKPOINT",checkpoint_tf)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            seq_length = modeling.get_shape_list(input_ids)[1]


            start_positions = features["start_positions"]
            end_positions = features["end_positions"]
            answer_types = features["answer_types"]
            y_true = (unique_ids,start_positions,end_positions,answer_types)

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
                total_loss = total_loss(y_true,y_pred)

            var_list = model.trainable_variables
            compgrad = compute_gradients(optimizer,total_loss, var_list)
            print("TRAIN_OP", compgrad)
            train_op = optimizer.apply_gradients(compgrad)
            print("REAL", train_op)
    

            output_spec = tf.estimator.EstimatorSpec(
              mode=mode,
              loss=total_loss,
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





def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    
    name_to_features = {
        "unique_id": tf.io.FixedLenFeature([], tf.int64),
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
    }
    
    if is_training:
        name_to_features["start_positions"] = tf.io.FixedLenFeature([], tf.int64)
        name_to_features["end_positions"] = tf.io.FixedLenFeature([], tf.int64)
        name_to_features["answer_types"] = tf.io.FixedLenFeature([], tf.int64)
    
    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        print("RECORD",record)
        example = tf.io.parse_single_example(serialized=record, features=name_to_features)
        #print("EXAMPLE",example)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, dtype=tf.int32)
            example[name] = t
      
        return example


    def input_fn(params):
        """The actual input function."""
        #batch_size = params["batch_size"]
        batch_size = train_batch_size
        
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
      
        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
      
        return d
    
    return input_fn



eval_filename = "nq-train.tfrecords-00000-of-00001"

predict_input_fn = input_fn_builder(
    input_file=eval_filename,
    seq_length=max_seq_length,
    is_training=False,
    drop_remainder=False)


train_input_fn = input_fn_builder(
    input_file=eval_filename,
    seq_length=max_seq_length,
    is_training=True,\
    drop_remainder=False)




params = {'batch_size':train_batch_size}

model_fn = model_fn_builder(params, init_checkpoint, learning_rate,\
                     num_train_steps, num_warmup_steps, use_tpu,\
                     use_one_hot_embeddings,bert_yn=BERT)
estimator = tf.estimator.Estimator(model_fn, './tf_estimator_example/')

x = estimator.train(train_input_fn,steps=100)


y_true = [[0, 1, 0], [0, 0, 1]]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
# Using 'auto'/'sum_over_batch_size' reduction type.
cce = tf.keras.losses.CategoricalCrossentropy()
cce(y_true, y_pred).numpy()




all_results = []

for result in estimator.predict(predict_input_fn, yield_single_examples=True):
    if len(all_results) % 1000 == 0:
      print("Processing example: %d" % (len(all_results)))
    
    unique_id = int(result["unique_ids"])
    start_logits = [float(x) for x in result["start_logits"].flat]
    end_logits = [float(x) for x in result["end_logits"].flat]
    answer_type_logits = [float(x) for x in result["answer_type_logits"].flat]
    
    all_results.append(\
        RawResult(\
            unique_id=unique_id,\
            start_logits=start_logits,\
            end_logits=end_logits,\
            answer_type_logits=answer_type_logits))




def mk_model(config):
    seq_len = config['max_position_embeddings']
    unique_id  = tf.keras.Input(shape=(1,),dtype=tf.int64,name='unique_id')
    input_ids   = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='input_ids')   # The tokens
    input_mask  = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='input_mask')    # The mask for bert prediction
    segment_ids = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='segment_ids')    # 
    BERT = BertModel(config=config,name='bert')
    pooled_output, sequence_output = BERT(input_word_ids=input_ids,
                                          input_mask=input_mask,
                                          input_type_ids=segment_ids)

    logits = TDense(2,name='logits')(sequence_output)
    start_logits,end_logits = tf.split(logits,axis=-1,num_or_size_splits= 2,name='split')
    start_logits = tf.squeeze(start_logits,axis=-1,name='start_squeeze')
    end_logits   = tf.squeeze(end_logits,  axis=-1,name='end_squeeze')

    ans_type = TDense(5,name='ans_type')(pooled_output)
    return tf.keras.Model([input_ for input_ in [unique_id,input_ids,input_mask,segment_ids] 
                           if input_ is not None],
                          [unique_id,start_logits,end_logits,ans_type],
                          name='bert-baseline') 

vocab_file = None
spm_model_file = albert_layer.resolved_object.sp_model_file.asset_path.numpy()
tokenizer = FullTokenizer(
    vocab_file=vocab_file, do_lower_case=do_lower_case, spm_model_file = \
        spm_model_file)


def run():
  print("RUNNING THE 'RUN' OPTION IN BERTBASELINE")
  bert_config = modeling.BertConfig.from_json_file(bert_config_file)

  validate_flags_or_throw(bert_config)
  tf.io.gfile.makedirs(output_dir)

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

  model_fn = tf2baseline.model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=init_checkpoint,
      learning_rate=learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=use_tpu,
      use_one_hot_embeddings=use_one_hot_embeddings)

  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params={'batch_size':train_batch_size})


  if do_train:
    print("***** Running training on precomputed features *****")
    print("  Num split examples = %d", num_train_features)
    print("  Batch size = %d", train_batch_size)
    print("  Num steps = %d", num_train_steps)
    train_filenames = tf.io.gfile.glob(train_precomputed_file)
    train_input_fn = input_fn_builder(
        input_file=train_filenames,
        seq_length=max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if do_predict:
    if not output_prediction_file:
      raise ValueError(
          "--output_prediction_file must be defined in predict mode.")

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

    print("***** Running predictions *****")
    print(f"  Num orig examples = %d" % len(eval_examples))
    print(f"  Num split examples = %d" % len(eval_features))
    print(f"  Batch size = %d" % predict_batch_size)
    for spans, ids in num_spans_to_ids.items():
      print(f"  Num split into %d = %d" % (spans, len(ids)))

    predict_input_fn = input_fn_builder(
        input_file=eval_filename,
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

