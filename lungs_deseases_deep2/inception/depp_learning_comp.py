import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os
from tqdm import tqdm
import pandas as pd
import cv2
import scipy.ndimage as ndimg
# Functions and classes for loading and using the Inception model.
import inception
from sklearn import preprocessing
from sklearn.decomposition import PCA

TEST_DATA = "./my_data/test_img/"
TRAIN_DATA = "./my_data/train_img/"
# data root directory 
DATA_DIR = "./my_data/"
CHECKPOINT_PATH = DATA_DIR+"model_checkpoints/"
train_writer = tf.summary.FileWriter(DATA_DIR+'/graph/')    
# load train data into dataframe 

## load files
train = pd.read_csv(DATA_DIR+'train.csv')
test = pd.read_csv(DATA_DIR+'test.csv')
# encode train labels to numbers 
le = preprocessing.LabelEncoder()
# test 
# train = train.iloc[0:300]
# test = test.iloc[0:300]
y_labels = le.fit_transform(train['label'])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
# test-point :
# y_labels = y_labels[0:300]
# #############################################################
NUM_LABELS = train['label'].nunique()

print(train.head())
# download inception model 
inception.maybe_download()

# load inception model 
model = inception.Inception()
# calculate transfer values 
from inception import transfer_values_cache

# read images into numpy arrays (Train and test )
# function to read images as arrays
def read_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = tf.image.decode_png(img,channels=3)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = tf.image.resize_images(img,[128,128])
    # img = cv2.resize(img, (128,128)) # you can resize to  (128,128) or (256,256)
    return resized_img

train_data = []
test_data = []
train_labels = train['label'].values
N_RECORDS = train_labels.size
print("n records : {}".format(N_RECORDS))
train_file_names = []
test_file_names = []
# ------------------------------------------------------------
# train = train[0:300]
# test = test[0:300]
train_image_matrix = ([ndimg.imread(TRAIN_DATA + '{}.png'.format(img)) for img in tqdm(train['image_id'].values)])
test_image_matrix = ([ndimg.imread(TEST_DATA + '{}.png'.format(img)) for img in tqdm(test['image_id'].values)])
train_image_matrix = np.array(train_image_matrix)
test_image_matrix = np.array(test_image_matrix)

print(type(train_image_matrix))
# -------------------------------------------------------------
# show some images 
def plot_images(images, cls_true, cls_pred=None, smooth=True):

    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i],
                      interpolation=interpolation)

            # Name of the true class.
            cls_true_name = le.inverse_transform(cls_true[i])

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# display images 
# Get the first images from the test-set.
# images = train_image_matrix[0:9]

# Get the true classes for those images.
# cls_true = y_labels[0:9]

# Plot the images and labels using our helper-function above.
# plot_images(images=images, cls_true=cls_true, smooth=False)



# train and test data cache files 
file_path_cache_train = os.path.join(DATA_DIR, 'inception_grocery_train.pkl')
file_path_cache_test = os.path.join(DATA_DIR, 'inception_grocery_test.pkl')
print(file_path_cache_train)
print(file_path_cache_test)

print("Processing Inception transfer-values for training-images ...")

# train_shape = train_image_matrix.shape
print("dataset : ")
# print(train_shape)
# x_train = tf.get_variable("x_train",shape = [-1,train_shape[1],train_shape[2],train_shape[3]])
with tf.name_scope(name="x_train_y_train_x_test"):
    x_train = tf.placeholder(dtype=tf.int32,name="Train_X") # input x train
    y_train = tf.placeholder(dtype=tf.int32,name="Train_Y") # input y train
    x_test = tf.placeholder(dtype=tf.int32,name="Test_X") # input x test

size_tensor = tf.Variable([160,160],dtype=tf.int32)

sess = tf.Session()

# print(train_image_matrix.shape)
# print("Size :  ")
# print(train_image_matrix.nbytes)
# with tf.name_scope(name="matrix_tensors"):
#     tf_train_images = tf.Variable(train_image_matrix,name="x_matrix")
#     tf_test_images = tf.Variable(test_image_matrix,name="y_matrix")

# print("Shape : {0}".format(tf_train_images.get_shape()))
# resize images 

# -------------------------------------------------------------------------------
# print("Debug : "+train_image_matrix.shape)
# get batches 
IM_BATCH_SIZE = 50 # 5 images at a time 
# train_image_matrix_tmp = np.empty((len(train_image_matrix),160,160,3))
# test_image_matrix_tmp = np.empty((len(test_image_matrix),160,160,3))
# for i in range(0,len(train_image_matrix),IM_BATCH_SIZE):
#     print("Converting : "+str(i)+" end : "+str(i+IM_BATCH_SIZE)) 
    

#     tf_train_images = tf.image.resize_images(images=train_image_matrix[i:(i+IM_BATCH_SIZE)],
#                                              size=[160,160])
#     train_image_matrix_tmp[i:(i+IM_BATCH_SIZE)] = sess.run(tf_train_images)

# for i in range(0,len(test_image_matrix),IM_BATCH_SIZE):
#     print("Converting : "+str(i)+" end : "+str(i+IM_BATCH_SIZE)) 
    
#     tf_test_images = tf.image.resize_images(images=test_image_matrix[i:(i+IM_BATCH_SIZE)],
#                                              size=[160,160])
#     test_image_matrix_tmp[i:(i+IM_BATCH_SIZE)] = sess.run(tf_test_images)

#     # print("X test : original : "+str(train_image_matrix_tmp.shape))
#     # print("Y original : "+str(test_image_matrix_tmp.shape))
# # load resized matrix

# remaining_items = (len(train_image_matrix)-1) % IM_BATCH_SIZE
# if(remaining_items > 0):
#     tf_train_images = tf.image.resize_images(images=train_image_matrix[-remaining_items:],
#                                              size=[160,160])
#     train_image_matrix_tmp[-remaining_items:] = sess.run(tf_train_images)
    
# remaining_items = (len(test_image_matrix)-1) % IM_BATCH_SIZE
# if(remaining_items > 0):
#     tf_test_images = tf.image.resize_images(images=test_image_matrix[-remaining_items:],
#                                              size=[160,160])
#     test_image_matrix_tmp[-remaining_items:] = sess.run(tf_test_images)

    

# # lets find dimens 
# print("X test : original : "+str(train_image_matrix.shape))
# print("Y original : "+str(test_image_matrix.shape))

# print("X test : original : "+str(train_image_matrix_tmp.shape))
# print("Y original : "+str(test_image_matrix_tmp.shape))

# train_image_matrix = train_image_matrix_tmp
# test_image_matrix = test_image_matrix_tmp

# train_image_matrix_tmp,test_image_matrix_temp = sess.run([tf_train_images,tf_test_images])   
# -------------------------------------------------------------------------------
# print(tf_train_images)
# print("Rank : "+str(tf.rank(tf_train_images)))

print("_________________________________________________________________")

transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,images=train_image_matrix,model=model)
    # test images 
transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,images=test_image_matrix,model=model)

# test-point : 
# transfer_values_train = transfer_values_train[0:300]
# transfer_values_test = transfer_values_test[0:300]
print("Completed matrix to transfer values ")
# ________________________________________________________________________________________

print("Completed transfer values cals for TRAIN and test ")
# print(transfer_values_train_op)

# print(transfer_values_test.shape)

def plot_transfer_values(i):
    print("Input image:")
    
    # Plot the i'th image from the test-set.
    plt.imshow(tf_test_images[i], interpolation='nearest')
    plt.show()

    print("Transfer-values for the image using Inception model:")
    
    # Transform the transfer-values into an image.
    img = transfer_values_test[i]
    img = img.reshape((32, 64))

    # Plot the image for the transfer-values.
    plt.imshow(img, interpolation='nearest', cmap='Reds')
    plt.show()

# dense layer for the fully connected layer of Inveption model 

pca = PCA(n_components=2)

transfer_values_reduces = pca.fit_transform(transfer_values_train)
num_classes = NUM_LABELS
def plot_scatter(values, cls):
    # Create a color-map with a different color for each class.
    import matplotlib.cm as cm
    cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))

    # Get the color for each sample.
    colors = cmap[cls]

    # Extract the x- and y-values.
    x = values[:, 0]
    y = values[:, 1]

    # Plot it.
    plt.scatter(x, y, color=colors)
    plt.show()
# print("Y shape : ")    
# print(str(y_labels.shape))
plot_scatter(transfer_values_reduces,y_labels)

# TNSE not covered 

transfer_len = model.transfer_len
# with tf.name_scope(name="inputs_from_transfer_values"):
x = tf.placeholder(tf.float32,shape=[None , transfer_len],name='X')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

y_true_cls = tf.argmax(y_true, dimension=1,name="y_true_classes")
# logits = tf.layers.dense(inputs=x, units=NUM_LABELS,name='Dense_layer')
y_pred = tf.contrib.layers.fully_connected(
    inputs = x,
    num_outputs = NUM_LABELS,
    activation_fn = tf.nn.softmax
        
)
y_pred_cls = tf.argmax(y_pred, dimension=1,name="predicted_y_values")

with tf.name_scope(name="logits_from_dense_to_loss_calc"):

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred,name='Cross_entropy_loss_function')
    loss = tf.reduce_mean(cross_entropy,name='Reduced_loss')
with tf.name_scope(name="Optimizer"):
    global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,name="Adam").minimize(loss,global_step=global_step)

# y_pred_cls = tf.argmax(y_pred, dimension=1)
with tf.name_scope(name="Accuracy_calculation"):
    correct_prediction = tf.equal(tf.cast(y_pred_cls,dtype=tf.float32),tf.cast(y_true_cls,dtype=tf.float32),name='Accuracy_test')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name = "accuracy")

# create a summary for our cost and accuracy
tf.summary.scalar("cost", loss)
tf.summary.scalar("accuracy", accuracy)

# initialise all variabeles

train_batch_size = 60

# total examples in training 
num_images = 100                                                                                                                                                                                                                                                                                                                                          
# [random examples 
# from tf.contrib.data import Dataset, Iterator
print('Get batches ')
# image_batch, label_batch =  tf.train.shuffle_batch(
#         [transfer_values_train ,y_labels],
#         batch_size = train_batch_size,
#         capacity = num_images,                                                                                                                                                                                                                                                                          
#         num_threads=1,
#         allow_smaller_final_batch=True,
#         name="random_batch",
#         min_after_dequeue=50
#     )





# print("batch finished {0} {1}".format(image_batch,label_batch))
# # image_batch_matrix,label_batch_matrix = sess.run([image_batch,label_batch])
# print('Batch lengths : x : {0} y : {1}'.format(image_batch.shape,label_batch.shape))
# with tf.Session() as se: 

#     image_batch, label_batch = tf.train.batch(
#         tensors=[transfer_values_train ,y_labels],
#         batch_size=train_batch_size,
#         num_threads=1,
#         capacity=32,
#         enqueue_many=False,
#         shapes=None,
#         dynamic_pad=False,
#         allow_smaller_final_batch=True,
#         shared_name=None,
#         name=None
#     )


#     # initialize the queue threads to start to shovel data
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     image_batch_mat = sess.run(image_batch) # array of image batches 
#     label_batch_mat = sess.run(label_batch) # array of label batches

#         # stop our queue threads and properly close the session
#     coord.request_stop()
#     coord.join(threads)
#     print("batch done")

# def random_batch():
    
#     high_limit =len(image_batch_mat)
#     index_ = np.random.randint(low=0,high = high_limit-1)
#     return image_batch_mat[index_] , label_batch_mat[index_]

def random_batch():
    # Number of images (transfer-values) in the training-set.
    num_images = len(transfer_values_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random x and y-values.
    # We use the transfer-values instead of images as x-values.
    x_batch = transfer_values_train[idx]
    y_batch = y_labels[idx]
    return x_batch, y_batch


init = tf.global_variables_initializer()
sess.run(init) 
# ------------------Load saved session -----------------------------
ckpt = tf.train.get_checkpoint_state(CHECKPOINT_PATH)
saver = tf.train.Saver()
if ckpt and ckpt.model_checkpoint_path:
 saver.restore(sess, ckpt.model_checkpoint_path)

# -----------------------------------------------------------------
summary_op = tf.summary.merge_all()
# model saver 

def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):
        print("Iteration : {}".format(i))
        # Get a batch of training examples.
        # x_batch now holds a batch of images (transfer-values) and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = random_batch()

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        y_true_batch = tf.one_hot(indices=y_true_batch,depth=num_classes,axis=1)
        y_true_batch = sess.run(y_true_batch)
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        # We also want to retrieve the global_step counter.
        i_global, _ ,summary_final = sess.run([global_step, optimizer,summary_op],
                                  feed_dict=feed_dict_train)
        train_writer.add_summary(summary_final,i)
        # Print status to screen every 100 iterations (and last).
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-batch.
            with tf.name_scope("batch_accuacy"):
                batch_acc = sess.run(accuracy,
                                        feed_dict=feed_dict_train)
                # tf.summary.scalar("Accuracy_of_batch",batch_acc)
                print("(B)::Accuracy : "+str(batch_acc))
            # Print status.
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            # print(msg.format(i_global, batch_acc))
        if (i_global % 1000 == 0 ):
            saver.save(sess,CHECKPOINT_PATH,global_step=i_global)
    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

optimize(num_iterations=20000)
# predict 
print("Predictions")

predicted_values = sess.run(y_pred_cls,
    feed_dict={x:transfer_values_test}
    )
print(predicted_values)
# store in test data Frame 
test['label']  = le.inverse_transform(predicted_values)
# write output 

test.to_csv(DATA_DIR+'solution.csv')

train_writer.add_graph(sess.graph)

