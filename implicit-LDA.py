'''Folked from https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/latent_dirichlet_allocation_edward2.py,
Edited by Wenmin Zhao'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

# Dependency imports
from absl import flags
import numpy as np
import scipy.sparse
from six.moves import cPickle as pickle
from six.moves import urllib
import tensorflow as tf

from tensorflow_probability import edward2 as ed
from IPython.core.debugger import set_trace

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

import sklearn as sk
import bottleneck as bn
from sklearn import metrics
from sklearn.metrics import accuracy_score
import csv
from sklearn.model_selection import train_test_split

flags.DEFINE_float(
    "learning_rate", default=3e-4, help="Learning rate.")
flags.DEFINE_float(
    "momentum", default=0.90, help="Learning rate.")
flags.DEFINE_integer(
    "max_steps",
#     default=180000, 
    default=75000,
    help="Number of training steps to run.")
flags.DEFINE_integer(
    "num_topics",
    default=50,
    help="The number of topics.")
flags.DEFINE_list(
    "layer_sizes",
    default=["300", "300",'300'],
    help="Comma-separated list denoting hidden units per layer in the encoder.")
flags.DEFINE_string(
    "activation",
    default="relu",
    help="Activation function for all hidden layers.")
flags.DEFINE_integer(
    "batch_size",
    default=200,
    help="Batch size.")
flags.DEFINE_float(
    "prior_initial_value", default=0.7, help="The initial value for prior.")
flags.DEFINE_integer(
    "prior_burn_in_steps",
    default=50000,
#     default=120000,
    help="The number of training steps with fixed prior.")
flags.DEFINE_string(
    "data_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "lda/data"),
    help="Directory where data is stored (if using real data).")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "lda/IMDB"),
    help="Directory to put the model's fit.")
flags.DEFINE_integer(
    "viz_steps", default=10000, help="Frequency at which save visualizations.")

flags.DEFINE_bool("newsgroup", default=False, 
                  help="If true, uses newsgroup data, otherwise use IMDB")
flags.DEFINE_bool(
    "delete_existing",
    default=False,
    help="If true, deletes existing directory.")

FLAGS = flags.FLAGS

def _clip_dirichlet_parameters(x):
  """Clips the Dirichlet parameters to the numerically stable KL region."""
  return tf.clip_by_value(x, 1e-3, 1e3)

def _softplus_inverse(x):
  """Returns inverse of softplus function."""
  return np.log(np.expm1(x))

def latent_dirichlet_allocation(concentration, topics_words):
  """Latent Dirichlet Allocation in terms of its generative process.
  The model posits a distribution over bags of words and is parameterized by
  a concentration and the topic-word probabilities. It collapses per-word
  topic assignments.
  Args:
    concentration: A Tensor of shape [1, num_topics], which parameterizes the
      Dirichlet prior over topics.
    topics_words: A Tensor of shape [num_topics, num_words], where each row
      (topic) denotes the probability of each word being in that topic.
  Returns:
    bag_of_words: A random variable capturing a sample from the model, of shape
      [1, num_words]. It represents one generated document as a bag of words.
  """
  topics = ed.Dirichlet(concentration=concentration, name="topics")
  word_probs = tf.matmul(topics, topics_words)
  
  # The observations are bags of words and therefore not one-hot. However,
  # log_prob of OneHotCategorical computes the probability correctly in
  # this case.
  bag_of_words = ed.OneHotCategorical(probs=word_probs, name="bag_of_words")

  return bag_of_words

 def make_lda_variational(activation, num_topics, layer_sizes):
  """Creates the variational distribution for LDA.
  Args:
    activation: Activation function to use.
    num_topics: The number of topics.
    layer_sizes: The number of hidden units per layer in the encoder.
  Returns:
    lda_variational: A function that takes a bag-of-words Tensor as
      input and returns a distribution over topics.
  """
  encoder_net = tf.keras.Sequential()
  for num_hidden_units in layer_sizes:
    encoder_net.add(
        tf.keras.layers.Dense(
            num_hidden_units,
            activation=activation,
            kernel_initializer=tf.compat.v1.glorot_normal_initializer()))
#     encoder_net.add(tf.keras.layers.Dropout(0.25))
    encoder_net.add(
        tf.keras.layers.Dense(
          num_topics,
          activation=tf.nn.softplus,
          kernel_initializer=tf.compat.v1.glorot_normal_initializer()))

  def lda_variational(bag_of_words):
#     concentration = _clip_dirichlet_parameters(tf.contrib.layers.batch_norm(encoder_net(bag_of_words)))
    concentration = _clip_dirichlet_parameters(encoder_net(bag_of_words))
    return ed.Dirichlet(concentration=concentration, name="topics_posterior")

  return lda_variational, encoder_net

def model_fn(features, labels, mode, params, config):
  """Builds the model function for use in an Estimator.
  Arguments:
    features: The input features for the Estimator.
    labels: The labels, unused here.
    mode: Signifies whether it is train or test or predict.
    params: Some hyperparameters as a dictionary.
    config: The RunConfig, unused here.
  Returns:
    EstimatorSpec: A tf.estimator.EstimatorSpec instance.
  """
  del labels, config

  # Set up the model's learnable parameters.
  logit_concentration = tf.compat.v1.get_variable(
      "logit_concentration",
      shape=[1, params["num_topics"]],
      initializer=tf.compat.v1.initializers.constant(
          _softplus_inverse(params["prior_initial_value"])))
  concentration = _clip_dirichlet_parameters(
      tf.nn.softplus(logit_concentration))

  num_words = features.shape[1]
  topics_words_logits = tf.compat.v1.get_variable(
      "topics_words_logits",
      shape=[params["num_topics"], num_words],
      initializer=tf.compat.v1.glorot_normal_initializer())
  topics_words = tf.nn.softmax(topics_words_logits, axis=-1)

  # Compute expected log-likelihood. First, sample from the variational
  # distribution; second, compute the log-likelihood given the sample.
  lda_variational, encoder_net  = make_lda_variational(
      params["activation"],
      params["num_topics"],
      params["layer_sizes"])
  with ed.tape() as variational_tape:
    _ = lda_variational(features)

  with ed.tape() as model_tape:
    with ed.interception(
        ed.make_value_setter(topics=variational_tape["topics_posterior"])):
      posterior_predictive = latent_dirichlet_allocation(concentration,
                                                         topics_words)

  log_likelihood = posterior_predictive.distribution.log_prob(features)
  tf.compat.v1.summary.scalar("log_likelihood",
                              tf.reduce_mean(input_tensor=log_likelihood))

  # Compute the KL-divergence between two Dirichlets analytically.
  # The sampled KL does not work well for "sparse" distributions
  # (see Appendix D of [2]).
  kl = variational_tape["topics_posterior"].distribution.kl_divergence(
      model_tape["topics"].distribution)
  tf.compat.v1.summary.scalar("kl", tf.reduce_mean(input_tensor=kl))

  # Ensure that the KL is non-negative (up to a very small slack).
  # Negative KL can happen due to numerical instability.
  with tf.control_dependencies(
      [tf.compat.v1.assert_greater(kl, -1e-3, message="kl")]):
    kl = tf.identity(kl)

  elbo = log_likelihood - kl
  avg_elbo = tf.reduce_mean(input_tensor=elbo)
  tf.compat.v1.summary.scalar("elbo", avg_elbo)
  loss = -avg_elbo

  # Perform variational inference by minimizing the -ELBO.
  global_step = tf.compat.v1.train.get_or_create_global_step()
  optimizer = tf.compat.v1.train.AdamOptimizer(params["learning_rate"])

  # This implements the "burn-in" for prior parameters (see Appendix D of [2]).
  # For the first prior_burn_in_steps steps they are fixed, and then trained
  # jointly with the other parameters.
  grads_and_vars = optimizer.compute_gradients(loss)
  grads_and_vars_except_prior = [
      x for x in grads_and_vars if x[1] != logit_concentration]

  def train_op_except_prior():
    return optimizer.apply_gradients(
        grads_and_vars_except_prior,
        global_step=global_step)

  def train_op_all():
    return optimizer.apply_gradients(
        grads_and_vars,
        global_step=global_step)

  train_op = tf.cond(
      pred=global_step < params["prior_burn_in_steps"],
      true_fn=train_op_except_prior,
      false_fn=train_op_all)

  # The perplexity is an exponent of the average negative ELBO per word.
#   words_per_document = tf.reduce_sum(input_tensor=features, axis=1)
  
  log_perplexity = -tf.reduce_sum(elbo) / tf.reduce_sum(features)
  
#   tf.compat.v1.summary.scalar(
#       "perplexity", tf.exp(tf.reduce_mean(input_tensor=log_perplexity)))
  (log_perplexity_tensor,
   log_perplexity_update) = tf.compat.v1.metrics.mean(log_perplexity)
  perplexity_tensor = tf.exp(log_perplexity_tensor)

  # Obtain the topics summary. Implemented as a py_func for simplicity.
  topics = tf.compat.v1.py_func(
      functools.partial(get_topics_strings, vocabulary=params["vocabulary"]),
      [topics_words, concentration],
      tf.string,
      stateful=False)
  tf.compat.v1.summary.text("topics", topics)
  
  var_concentration = _clip_dirichlet_parameters(encoder_net(features))

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops={
          "elbo": tf.compat.v1.metrics.mean(elbo),
          "log_likelihood": tf.compat.v1.metrics.mean(log_likelihood),
          "kl": tf.compat.v1.metrics.mean(kl),
          "perplexity": (perplexity_tensor, log_perplexity_update),
          "topics": (topics, tf.no_op()),
      },
      
      predictions={'topics_posterior_params': var_concentration}
      
      )


def get_topics_strings(topics_words, alpha, vocabulary,
                       topics_to_print=20, words_per_topic=10):
  """Returns the summary of the learned topics.
  Arguments:
    topics_words: KxV tensor with topics as rows and words as columns.
    alpha: 1xK tensor of prior Dirichlet concentrations for the
        topics.
    vocabulary: A mapping of word's integer index to the corresponding string.
    topics_to_print: The number of topics with highest prior weight to
        summarize.
    words_per_topic: Number of wodrs per topic to return.
  Returns:
    summary: A np.array with strings.
  """
  alpha = np.squeeze(alpha, axis=0)
  # Use a stable sorting algorithm so that when alpha is fixed
  # we always get the same topics.
  highest_weight_topics = np.argsort(-alpha, kind="mergesort")
  top_words = np.argsort(-topics_words, axis=1)

  res = []
  for topic_idx in highest_weight_topics[:topics_to_print]:
    l = ["index={} alpha={:.2f}".format(topic_idx, alpha[topic_idx])]
    l += [vocabulary[word] for word in top_words[topic_idx, :words_per_topic]]
    res.append(" ".join(l))

  return np.array(res)

def IMDB_dataset(split_name, num_words, words_to_idx, shuffle_and_repeat):
  path="/content/drive/My Drive/IMDB/IMDB_Dataset.csv"
  
  with open(path, newline='\n') as csvfile:
    data = list(csv.reader(csvfile, delimiter=','))[1:]
  
  label=[]
  documents=[]

  for i in range(len(data)):
    if data[i][-1] == 'positive':
      label.append(1)
    else: 
      label.append(0)
    documents.append(data[i][0])

  train_data, test_data, train_labels, test_labels = train_test_split(documents, np.array(label), test_size=0.4, random_state=0)
  vectorizer = CountVectorizer(vocabulary=words_to_idx, dtype=np.float32)
  
  if split_name=='train':
    data=train_data
  else:
    data=test_data
    
  sparse_matrix = vectorizer.fit_transform(data)
  
  num_documents=sparse_matrix.shape[0]
  dataset = tf.data.Dataset.range(num_documents)
  
  # For training, we shuffle each epoch and repenp.at the epochs.
  if shuffle_and_repeat:
    dataset = dataset.shuffle(num_documents).repeat()

  # Returns a single document as a dense TensorFlow tensor. The dataset is
  # stored as a sparse matrix outside of the graph.
  def get_row_py_func(idx):
    def get_row_python(idx_py):
      return np.squeeze(np.array(sparse_matrix[idx_py].todense()), axis=0)

    py_func = tf.compat.v1.py_func(
        get_row_python, [idx], tf.float32, stateful=False)
    py_func.set_shape((num_words,))
    return py_func

  dataset = dataset.map(get_row_py_func)
  return dataset

def IMDB_build_input_fns(data_dir, batch_size):
  """Builds iterators for train and evaluation data.
  Each object is represented as a bag-of-words vector.
  Arguments:
    data_dir: Folder in which to store the data.
    batch_size: Batch size for both train and evaluation.
  Returns:
    train_input_fn: A function that returns an iterator over the training data.
    eval_input_fn: A function that returns an iterator over the evaluation data.
    vocabulary: A mapping of word's integer index to the corresponding string.
  """

  vocab_path="/content/drive/My Drive/IMDB/IMDB_vocab.pkl"
  
  with open(vocab_path, "rb") as f:
    words_to_idx = pickle.load(f)
  num_words = len(words_to_idx)

  vocabulary = [None] * num_words
  for word, idx in words_to_idx.items():
    vocabulary[idx] = word

  # Build an iterator over training batches.
  def train_input_fn():
    dataset = IMDB_dataset(
        "train", num_words, words_to_idx, shuffle_and_repeat=True)
    # Prefetching makes training about 1.5x faster.
    dataset = dataset.batch(batch_size).prefetch(batch_size)
    return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

  # Build an iterator over the heldout set.
  def eval_input_fn():
    dataset = IMDB_dataset(
        "test", num_words, words_to_idx, shuffle_and_repeat=False)
    dataset = dataset.batch(7532)
    return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

  return train_input_fn, eval_input_fn, vocabulary


def newsgroups_dataset(split_name, num_words, words_to_idx, shuffle_and_repeat):
  news_data = fetch_20newsgroups(subset=split_name, remove=('headers', 'footers', 'quotes'))
  data=news_data.data
  vectorizer = CountVectorizer(vocabulary=words_to_idx, dtype=np.float32)
  sparse_matrix = vectorizer.fit_transform(data)
  num_documents=sparse_matrix.shape[0]
  dataset = tf.data.Dataset.range(num_documents)
  
  # For training, we shuffle each epoch and repenp.at the epochs.
  if shuffle_and_repeat:
    dataset = dataset.shuffle(num_documents).repeat()

  # Returns a single document as a dense TensorFlow tensor. The dataset is
  # stored as a sparse matrix outside of the graph.
  def get_row_py_func(idx):
    def get_row_python(idx_py):
      return np.squeeze(np.array(sparse_matrix[idx_py].todense()), axis=0)

    py_func = tf.compat.v1.py_func(
        get_row_python, [idx], tf.float32, stateful=False)
    py_func.set_shape((num_words,))
    return py_func

  dataset = dataset.map(get_row_py_func)
  return dataset

def newsgroup_build_input_fns(data_dir, batch_size):
  """Builds iterators for train and evaluation data.
  Each object is represented as a bag-of-words vector.
  Arguments:
    data_dir: Folder in which to store the data.
    batch_size: Batch size for both train and evaluation.
  Returns:
    train_input_fn: A function that returns an iterator over the training data.
    eval_input_fn: A function that returns an iterator over the evaluation data.
    vocabulary: A mapping of word's integer index to the corresponding string.
  """

  with open("/content/drive/My Drive/LN-lda/data/20news_clean/vocab.pkl", "rb") as f:
    words_to_idx = pickle.load(f)
  num_words = len(words_to_idx)

  vocabulary = [None] * num_words
  for word, idx in words_to_idx.items():
    vocabulary[idx] = word

  # Build an iterator over training batches.
  def train_input_fn():
    dataset = newsgroups_dataset(
        "train", num_words, words_to_idx, shuffle_and_repeat=True)
    # Prefetching makes training about 1.5x faster.
    dataset = dataset.batch(batch_size).prefetch(batch_size)
    return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

  # Build an iterator over the heldout set.
  def eval_input_fn():
    dataset = newsgroups_dataset(
        "test", num_words, words_to_idx, shuffle_and_repeat=False)
    dataset = dataset.batch(7532)
    return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

  return train_input_fn, eval_input_fn, vocabulary


def main(argv):
  del argv  # unused

  params = FLAGS.flag_values_dict()
  params["layer_sizes"] = [int(units) for units in params["layer_sizes"]]
  params["activation"] = getattr(tf.nn, params["activation"])
  
#   if FLAGS.delete_existing and tf.io.gfile.exists(FLAGS.model_dir):
#     tf.compat.v1.logging.warn("Deleting old log directory at {}".format(
#         FLAGS.model_dir))
#     tf.io.gfile.rmtree(FLAGS.model_dir)
#   tf.io.gfile.makedirs(FLAGS.model_dir)

  if FLAGS.newsgroup:
    train_input_fn, eval_input_fn, vocabulary = newsgroup_build_input_fns(
        FLAGS.data_dir,
        FLAGS.batch_size)
  else:
    train_input_fn, eval_input_fn, vocabulary = IMDB_build_input_fns(
        FLAGS.data_dir, FLAGS.batch_size)
    
  params["vocabulary"] = vocabulary

  estimator = tf.estimator.Estimator(
      model_fn,
      params=params,
      config=tf.estimator.RunConfig(
#           model_dir=FLAGS.model_dir,
          model_dir=path,
          save_checkpoints_steps=FLAGS.viz_steps,
      ),
  )
  
  eval_results = estimator.evaluate(eval_input_fn)

  for _ in range(FLAGS.max_steps // FLAGS.viz_steps):
    estimator.train(train_input_fn, steps=FLAGS.viz_steps)
    eval_results = estimator.evaluate(eval_input_fn)
    # Print the evaluation results. The keys are strings specified in
    # eval_metric_ops, and the values are NumPy scalars/arrays.
    for key, value in eval_results.items():
      print(key)
      if key == "topics":
        # Topics description is a np.array which prints better row-by-row.
        for s in value:
          print(s)
      else:
        print(str(value))
      print("")
    print("")

if __name__ == "__main__":
  tf.compat.v1.app.run()


