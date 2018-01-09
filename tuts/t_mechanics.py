import tensorflow as tf 
import numpy as np 
import argparse 
import sys 
import os 
import time 
import math 
import mechanics_functions as nn_algo 
from six.moves import xrange
# mnist data 
from tensorflow.examples.tutorials.mnist import input_data 
FLAGS = None
def placeholders(batch_size,image_rows=28,image_columns=28,channels=1):
    # x,y
    return tf.placeholder(tf.float32,shape=(batch_size,nn_algo.IMAGE_PIXELS)),\
        tf.placeholder(tf.int32,shape=(batch_size))
def do_eval(sess,eval_correct,image_placeholder,label_placeholder,data_set):
    # print('Dataset size : ',(data_set))
    true_eval = 0 
    step_per_epoch = data_set.num_examples // FLAGS.batch_size 
    num_examples = data_set.num_examples
    for step in xrange(step_per_epoch):
        images , labels = data_set.next_batch(FLAGS.batch_size,FLAGS.fake_data)
        # print('data shape x : {0}, y : {1}'.format(tf.shape(images)[0],tf.shape(labels)[0]))
        
        feed_dict = {
            image_placeholder : images,
            label_placeholder : labels 
        }
        true_eval+=sess.run(eval_correct,feed_dict=feed_dict) 
    precision = float(true_eval) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_eval, precision))


def run_training():
    datasets = input_data.read_data_sets(FLAGS.input_data_dir,FLAGS.fake_data)
    with tf.Graph().as_default():
        image_placeholder,label_placeholder = placeholders(FLAGS.batch_size)
        logits = nn_algo.inference(
            images = image_placeholder,  
            hidden_units1 = FLAGS.hidden1,
            hidden_units2 = FLAGS.hidden2
        )

        loss = nn_algo.loss(label_placeholder,logits)
        train_op = nn_algo.training(loss,FLAGS.learning_rate)

        # evaluation  
        eval_correct = nn_algo.evaluation(logits,label_placeholder) 

        # merge summary of all tensors 
        summary = tf.summary.merge_all() 

        # ariable init 
        init = tf.global_variables_initializer() 
        # saver for writing training checkpoints 
        saver = tf.train.Saver() 

        # session 
        sess = tf.Session() 

        # summary writer 
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir,sess.graph) 

        sess.run(init) 
        data_set = datasets.train
        # training  
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            # get data  
            images,labels = data_set.next_batch(FLAGS.batch_size,FLAGS.fake_data)
            feed_dict = {
                                image_placeholder: images, 
                                label_placeholder: labels
                            }
            # run training step 
            _,loss_value = sess.run([train_op,loss],feed_dict = feed_dict)
            duration = time.time() - start_time 
            # write summary to 
            if step % 100 == 0 : 
                # merge summary 
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # update the event file 
                summary_str = sess.run(summary,feed_dict=feed_dict) 
                summary_writer.add_summary(summary_str,step) 
                summary_writer.flush() 

            if (step +1) % 1000 == 0 or (step+1) % FLAGS.max_steps == 0 :
                checkpoint_file = os.path.join(FLAGS.log_dir,'model.ckpt')
                saver.save(sess,checkpoint_file,global_step=step) 

                print('train set evaluation')
                do_eval(
                    sess,
                    eval_correct,
                    image_placeholder,
                    label_placeholder,
                    datasets.train
                )
                print('validation set ')
                do_eval(
                    sess,
                    eval_correct,
                    image_placeholder,
                    label_placeholder,
                    datasets.validation
                )
                print('test set ')
                do_eval(
                    sess,
                    eval_correct,
                    image_placeholder,
                    label_placeholder,
                    datasets.test
                )



def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir) 
    tf.gfile.MakeDirs(FLAGS.log_dir) 
    run_training()
if __name__ == '__main__':
    # arguments 
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate '

    )
    parser.add_argument(
        '--max_steps',
        type=float,
        default=2000,
        help='Max epochs'
    )
    parser.add_argument(
        '--hidden1',
        type=float,
        default=128,
        help='total neurons in first hidden layer'
    )
    parser.add_argument(
        '--hidden2',
        type=float,
        default=32,
        help='total neurons in second hidden layer'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                            'tensorflow/mnist/input_data'),
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/fully_connected_feed'),
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'
    )
    FLAGS,unparsed = parser.parse_known_args()
    print('Log directory : ',FLAGS.log_dir)
    tf.app.run(main=main,argv=[sys.argv[0]]+unparsed)
