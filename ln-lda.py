'''Folked from https://github.com/akashgit/autoencoding_vi_for_topic_models, edited by Wenmin Zhao'''

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import itertools, time
import sys, os
from collections import OrderedDict
from copy import deepcopy
from time import time
import matplotlib.pyplot as plt
import pickle
import sys, getopt
import scipy.sparse

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import sklearn as sk
import bottleneck as bn
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.special import softmax
import csv

from gensim.matutils import hellinger, jaccard, jensen_shannon, kullback_leibler
from scipy.special import softmax

#Global Params
  
learning_rate = 0.005
batch_size = 200
training_epochs = 600
display_step = 20
prior_alpha = 1.0
transfer_fct = tf.nn.softplus
topics = 200
burn_in_epoch = 10
unit_size = 200
is_prod_lda = True
data_name ='IMDB'
model_path=''

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def xavier_init(fan_in, fan_out, constant=1):
  low = -constant*np.sqrt(6.0/(fan_in + fan_out))
  high = constant*np.sqrt(6.0/(fan_in + fan_out))

  return tf.random_uniform((fan_in, fan_out),
                           minval=low, maxval=high,
                           dtype=tf.float32)

def log_dir_init(fan_in, fan_out,topics=50):
    return tf.log((1.0/topics)*tf.ones([fan_in, fan_out]))


def load_data(data_type):
  if data_type=='20news':
    newsgroups_data_train = fetch_20newsgroups(subset='train')
    newsgroups_data_test = fetch_20newsgroups(subset='test')
    train_data=newsgroups_data_train.data
    test_data = newsgroups_data_test.data
    
    train_labels = fetch_20newsgroups(subset='train').target
    test_labels = fetch_20newsgroups(subset='test').target
    
    DIR='/content/drive/My Drive/LN-lda/'
    vocab_path = DIR + 'data/20news_clean/vocab'
    
  else: 
    path="/content/drive/My Drive/IMDB/IMDB_Dataset.csv"
    vocab_path="/content/drive/My Drive/IMDB/IMDB_vocab"
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
    
    
  vocab_dict=load_obj(vocab_path)
  vocab_size = len(vocab_dict)
  id2word = {}
  vocabulary = [None] * vocab_size

  for word, idx in vocab_dict.items():
    vocabulary[idx] = word
    id2word[idx] = word
  
  vectorizer = CountVectorizer(vocabulary=vocab_dict, dtype=np.float32)
#   vectorizer = CountVectorizer(dtype=np.float32)
  sparse_matrix_test = vectorizer.fit_transform(test_data)
  sparse_matrix_train = vectorizer.fit_transform(train_data)
  
  return sparse_matrix_train, sparse_matrix_test, train_labels, test_labels, vocab_dict, vocabulary

def create_minibatch(data):
    rng = np.random.RandomState(10)

    while True:
        # Return random data samples of a size 'minibatch_size' at each iteration
        ixs = rng.randint(data.shape[0], size=batch_size)
        yield data[ixs]

def print_top_words(beta, feature_names, n_top_words=10):
    print('---------------Printing the Topics------------------')
    for i in range(len(beta)):
        print(" ".join([feature_names[j]
            for j in beta[i].argsort()[:-n_top_words - 1:-1]]))
    print('---------------End of Topics------------------')

def make_network(layer1=unit_size,layer2=unit_size,num_topics=topics, size=vocab_size):
  
  network_architecture = \
      dict(n_hidden_recog_1 = layer1, # 1st layer encoder neurons
           n_hidden_recog_2 = layer2, # 2nd layer encoder neurons
           n_hidden_gener_1 = size, # 1st layer decoder neurons
           n_input = size, # MNIST data input (img shape: 28*28)
           n_z = num_topics)  # dimensionality of latent space

  return network_architecture

def initialize_weights(n_hidden_recog_1, 
                       n_hidden_recog_2,
                       n_hidden_gener_1,
                       n_input, 
                       n_z):
  
  all_weights = dict()
  
  all_weights['weights_recog'] = {
      'h1': tf.get_variable('h1',[n_input, n_hidden_recog_1]),
      'h2': tf.get_variable('h2',[n_hidden_recog_1, n_hidden_recog_2]),
      'out_mean': tf.get_variable('out_mean',[n_hidden_recog_2, n_z]),
      'out_log_sigma': tf.get_variable('out_log_sigma',[n_hidden_recog_2, n_z])}
  
  all_weights['biases_recog'] = {
      'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
      'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
      'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
      'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}

  all_weights['weights_gener'] = {
      'h2': tf.Variable(xavier_init(n_z, n_hidden_gener_1))}
  
  return all_weights


def recognition_network(weights, biases):
  
  layer_1 = transfer_fct(tf.add(tf.matmul(x, weights['h1']),
                                     biases['b1']))

  layer_2 = transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                     biases['b2']))
  
  layer_do = tf.nn.dropout(layer_2, keep_prob)

  z_mean = tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_do, weights['out_mean']),
                  biases['out_mean']))

  z_log_sigma_sq = tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_do, weights['out_log_sigma']),
             biases['out_log_sigma']))

  return (z_mean, z_log_sigma_sq)


def generator_network(z, weights):

  layer_do_0 = tf.nn.dropout(tf.nn.softmax(z), keep_prob)
  
  if is_prod_lda:
  	x_reconstr_mean = tf.nn.softmax(tf.contrib.layers.batch_norm(tf.add(
                tf.matmul(layer_do_0, weights['h2']),0.0)))

  else:
  	x_reconstr_mean = tf.add(tf.matmul(layer_do_0, tf.nn.softmax(tf.contrib.layers.batch_norm(weights['h2']))), 0.0)

  return x_reconstr_mean


'''-------Constructing Laplace Approximation to Dirichlet Prior--------------'''
def make_prior(alpha):
  
  a = alpha*np.ones((1 , int(h_dim))).astype(np.float32)
  
  mu2 = (np.log(a).T-np.mean(np.log(a),1)).T
  
  var2 = ( ( (1.0/a)*( 1 - (2.0/h_dim) ) ).T + ( 1.0/(h_dim*h_dim) )*np.sum(1.0/a,1) ).T

  dist = tfd.MultivariateNormalDiag(loc=mu2, scale_diag=var2)
  
  return dist
    

def main():

  sparse_matrix_train, sparse_matrix_test, train_labels, test_labels, vocab_dict, vocabulary = load_data(data_name)
  vocab_size=len(vocab_dict)

  data_te=sparse_matrix_test.toarray()
  data_tr=sparse_matrix_train.toarray()

  #--------------print the data dimentions--------------------------
  print('Data Loaded')
  print('Dim Training Data',data_tr.shape)
  print('Dim Test Data',data_te.shape)


  n_samples_tr = data_tr.shape[0]
  n_samples_te = data_te.shape[0]


  tf.reset_default_graph()
  tf.random.set_random_seed(0)

  network_architecture=make_network(num_topics=topics, size=vocab_size)

  h_dim = float(network_architecture["n_z"])

  n_z=network_architecture["n_z"]

  kl_coefficient =  tf.Variable(1., trainable = False, name ='kl_coefficient')

  '''----------------Inputs----------------'''
  x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
  keep_prob = tf.placeholder(tf.float32)
  words_per_document = tf.reduce_sum(input_tensor=x, axis=1)
  w_bar = x/words_per_document[:, None]

  # VAR_VARIABLE_SCOPE = "var"
  # with tf.variable_scope(VAR_VARIABLE_SCOPE, reuse=tf.AUTO_REUSE):
  network_weights = initialize_weights(**network_architecture)

  prior=make_prior(prior_alpha)

  z_mean, z_log_sigma_sq = recognition_network(network_weights["weights_recog"],
                                              network_weights["biases_recog"])

  n_z = network_architecture["n_z"]

  sigma=tf.exp(z_log_sigma_sq)

  posterior = tfd.MultivariateNormalDiag(loc=z_mean, scale_diag=tf.sqrt(sigma))

  z = posterior.sample()

  x_reconstr_mean = prodlda_generator_network(z, network_weights["weights_gener"])

  x_reconstr_mean += 1e-10
        
  neg_log_likelihood = - tf.reduce_sum(x * tf.log(x_reconstr_mean),1) #/tf.reduce_sum(x,1)

  kl = posterior.kl_divergence(prior)

  cost = kl + neg_log_likelihood # average over batch

  sum_neg_log_likelihood=tf.reduce_sum(cost)

  # log_perplexity =  tf.reduce_mean((latent_loss + log_likelihood) / words_per_document)

  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.99)

  main_op=optimizer.minimize(tf.reduce_mean(cost))

  # PRIOR_VARS=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=PRIOR_VAR_VARIABLE_SCOPE)

  # VARS=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=VAR_VARIABLE_SCOPE)

  # prior_op = optimizer.minimize(cost, var_list = PRIOR_VARS)

  test_perplexity=[]

  for epoch in range(training_epochs):
  
  #   kl_coefficient = tf.dtypes.cast(epoch/training_epochs, tf.float32)
  #   print(epoch/training_epochs)

    total_batch = int(n_samples_tr / batch_size)
    minibatches = create_minibatch(data_tr.astype('float32'))
  
    avg_cost = 0.
    avg_kl = 0.
    avg_log_likelihood = 0.
  
    for i in range(total_batch):
      batch_xs = next(minibatches)

      opt, kl_, log_likelihood_, cost_= sess.run((main_op, tf.reduce_mean(kl), tf.reduce_mean(neg_log_likelihood), 
                                                                       tf.reduce_mean(cost)),
                                                                       feed_dict={x: batch_xs, keep_prob: .75})
    
#     if epoch < burn_in_epoch:
    
#       # Fit training using batch data
#       opt, kl_, log_likelihood_, cost_= sess.run((main_op, tf.reduce_mean(kl), tf.reduce_mean(neg_log_likelihood), 
#                                                                              tf.reduce_mean(cost)),
#                                                                              feed_dict={x: batch_xs, keep_prob: .75})
#     else:
      
#       _,_, kl_, log_likelihood_, cost_= sess.run((main_op, prior_op, tf.reduce_mean(kl), tf.reduce_mean(neg_log_likelihood), 
#                                                                              tf.reduce_mean(cost)),
#                                                                              feed_dict={x: batch_xs, keep_prob: .75})

    # Compute average loss
    avg_cost += cost_ / n_samples_tr * batch_size
    avg_kl += kl_ / n_samples_tr * batch_size
    avg_log_likelihood += log_likelihood_ / n_samples_tr * batch_size

    neg_log_likelihood_ = sess.run(sum_neg_log_likelihood, feed_dict={x: docs_te, keep_prob:1.0})

    test_perplexity.append(np.exp(neg_log_likelihood_/docs_te.sum()))
    
    print("Epoch:", '%04d' % (epoch+1), \
          "cost=", "{:.9f}".format(avg_cost),\
          "kl=", "{:.9f}".format(avg_kl))

    if np.isnan(avg_cost):
        print(epoch,i,np.sum(batch_xs,1).astype(np.int),batch_xs.shape)
        print('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
        # return vae,emb
        sys.exit()

    # Display logs per epoch step
    if epoch % 50 == 0:
        saver.save(sess, model_path + "{}_model.ckpt".format(epoch))
        emb_, sum_neg_log_likelihood_= sess.run((network_weights["weights_gener"]['h2'], sum_neg_log_likelihood), feed_dict={x: docs_te, keep_prob:1.0})
      
        print("Epoch:", '%04d' % (epoch+1),
            "test perplexity =", "{:.9f}".format(np.exp(sum_neg_log_likelihood_/docs_te.sum())))

        print_top_words(emb_, next(zip(*sorted(vocab_dict.items(), key=lambda x: x[1]))))
      
  saver.save(sess, model_path + "model.ckpt")

if __name__ == "__main__":
	main()
