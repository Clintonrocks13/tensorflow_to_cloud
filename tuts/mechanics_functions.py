import argparse 
import sys 
import os 
import time 

import math 
import tensorflow as tf 

NUM_CLASSES = 10 
# 28x28 
IMAGE_SIZE = 28 
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE 

# function for creating Artificial neural network 

def inference(images, hidden_units1,hidden_units2):
    with tf.name_scope('H_layer_1'):
        # hidden layer 1 : 
        weights = tf.Variable(
            tf.truncated_normal(
                [IMAGE_PIXELS,hidden_units1],stddev=1.0 / math.sqrt(float(IMAGE_PIXELS)),
                
            ),dtype=tf.float32,name='weight_1'
        )
        b = tf.Variable(tf.zeros([hidden_units1]),name='biases') 
        # create first hidden layer 
        hidden1 = tf.nn.relu(tf.matmul(images,weights) + b,name='Hidden_1') 
    with tf.name_scope('H_layer_2'):
        # hidden layer 2 
        weights = tf.Variable(
            tf.truncated_normal([hidden_units1,hidden_units2],stddev=1.0/math.sqrt(float(hidden_units1))),
            dtype=tf.float32,
            name='weights_2'
        )
        b = tf.Variable(tf.zeros(hidden_units2),name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1,weights)+b,name = 'hidden_2')
    with tf.name_scope('H_layer_output'):
        # weights 
        weights = tf.Variable(tf.truncated_normal([hidden_units2,NUM_CLASSES],stddev=0.1/math.sqrt(float(hidden_units2))), name='weights_3') 
        b = tf.Variable(tf.zeros([NUM_CLASSES]),name='biases')
        # logits # don't use activation function here, as it's being taken care by the loss function 
        # cross entropy
        output_layer = tf.matmul(hidden2,weights)+b 
    return output_layer 


def loss(ac_labels,pred_logits):

    cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=ac_labels,
        logits=pred_logits,
        name='Cross_Entropy'
    )

    return tf.reduce_mean(cross_ent,name='cross_ent_mean') 

def training(loss,learning_rate):
    # add loss summary for visalization 
    tf.summary.scalar('loss',loss)
    # global step 
    global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step') 
    # tf.train.global_step()
    # training
    train = tf.train.GradientDescentOptimizer(learning_rate = 0.3).minimize(loss,global_step=global_step)
    return train 

def evaluation(logits,labels):
    return tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits,labels,1),tf.int32)) 


