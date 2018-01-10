import tensorflow as tf 
import numpy as np 
import pandas as pd 
import sys 
import os 
import argparse
import scipy.ndimage as s_image 
FLAGS = 0

train_images = np.array([],ndmin=2)
test_images = np.array([],ndmin=2)

train_csv = None 
test_csv = None 

tfrecord_train = 'train_tf.tfrecord' 
tfrecord_test = 'test_tf.tfrecord' 
def add_to_trrecord(file_name,tr_writer,offset=0):
    with tf.gfile.Open(file_name,'rb') as f :
        
def write_to_tf(n_ex,tf_file):
    # write train set to trecord  
    with tf.python.io.TFRecordWriter(tf_file):
        add_to_tfrecords()
        # for i in n_ex:


def load_csv():
    """
        load train and test csv files 
    """
    train_csv = pd.read_csv('train_csv',index_col='row_id')
    test_csv = pd.read_csv('test_csv',index_col='row_id') 
    
def load_images():
    """
    loading images as numpy arrays 

    """

    for image_file_name in train_csv.get('image_name'):
        # load the image in numpy array format 
        train_images.append(s_image.imread(image_file_name,flatten=True),axis=0)
    # test set 
    for image_file_name in test_csv.get('image_name'):
        # load the image in numpy array format 
        test_images.append(s_image.imread(image_file_name,flatten=True),axis=0)    


def main(_):
    pass 
if __name__ == '__main__':
    parser = aragparse.ArgumentParser() 
    parser.add_argument(
        '--data_dir',
        type=str,
        help='data directory'
    )
    FLAGS,unparsed = parser.parse_known_args()
    tf.app.run(main=main,argv=[sys.argv[0]+unparsed]) 