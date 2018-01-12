import matplotlib.image as npimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
from PIL import Image
from scipy.fftpack import fft
from skimage import color,io
from skimage.exposure import equalize_adapthist
from skimage.transform import resize
import sys 
import argparse 
import cv2
import os 
IMAGE_SIZE 
IMAGE_DATA = np.empty([10,IMAGE_SIZE,IMAGE_SIZE])
data_csv = pd.DataFrame()
def load_csv(file_name,index):
    """
        fille name : relative path or absolute path to csv file 
        index : data frame column to set the index 
        returns : pandas data frame 
    """
    # load csv data 
    data_csv = pd.read_csv(file_name,index=index) 
    return data_csv

def load_proc_image(data_csv,save_path):
    i=0
    for _,name in data_csv['photo_name'].iteritems():
        im = npimg.imread(name)
        if i < 5:
            IMAGE_DATA[i] = im 
        processed_image = clahe(image = im)
        if i<5:
            IMAGE_DATA[i+1] = processed_image
        # save image 
        io.imsave(os.path.join(save_path,processed_image)) 
        
        i+=2
def display_images():
    plt.figure(1) 
    for i in range(0,IMAGES_DATA.shape[0],2):
        plt.subplot(5,2,i+1)
        plt.imshow(IMAGES_DATA[i],cmap='gray')
        plt.imshow(IMAGE_DATA[i+1])
        plt.title('image {}'.format(i+1))
    plt.draw()
# don't use 
def fft_run():
    print('fourier transformation ')
    global fourier 
    fourier = fft(IMAGES_DATA[0,:,:,0])
def clahe(image):
    cl1 = equalize_adapthist(image)
    # IMAGES_DATA[1] = cl1
    return cl1 
def plot_hist(image):
    n,bins,patches = plt.hist(image,50,normed=1,facecolor = 'green',alpha=0.75)
    plt.draw()

if __name__=='__main__':

    # argument parsing 
    parser = argparse.ArgumentParser(description='data preprocessing') 
    parser.add_argument('--train_dir',dest='train_dir',type=str,action='store',help='train directory')
    parser.add_argument('--test_dir',dest='test_dir',action='store',type=str,help='test directory')
    parser.add_argument('--train_dir_csv',dest='train_csv',type=str,action='store',help='train directory')
    parser.add_argument('--test_dir_csv',dest='test_csv',action='store',type=str,help='test directory')
    
    FLAGS = parser.parse_args(sys.argv[1:])
    print(FLAGS)
    # create new folders for preprocessed data to be put into 
    if os.path.isdir(os.path.join(FLAGS.train_dir,'train_p') )!= True:
        os.mkdir(os.path.join(FLAGS.train_dir,'train_p'),755)
    else:
        # delete data from folder 
        os.system('rm'+os.path.join(FLAGS.train_dir,'train_p')+'*')

    if os.path.isdir(os.path.join(FLAGS.test_dir,'test_p') )!= True:
        os.mkdir(os.path.join(FLAGS.test_dir,'test_p'),755)
    else:
        os.system('rm'+os.path.join(FLAGS.test_dir,'test_p')+'*')
        
    # load csv train 
    train_csv = load_csv(FLAGS.train_csv) 
    # load image and preprocess train data and save 
    load_proc_image(data_csv = train_csv,save_path = os.path.join(FLAGS.train_dir,'train_p'))
    # load csv test 
    test_csv = load_csv(FLAGS.test_csv) 
    # load image and preprocess test data and save 
    load_proc_image(data_dir = test_csv,save_path = os.path.join(FLAGS.test_dir,'test_p'))
    
    
