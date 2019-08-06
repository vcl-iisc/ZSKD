import tensorflow as tf
tf.set_random_seed(777)
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
from math import floor, ceil, pi
import matplotlib.image as mpimg
from include.model_alex_full import model
from keras.utils import to_categorical

NO_OF_CLASSES = 10
IMAGE_SIZE = 32
NO_OF_CHANNELS = 3
tf.set_random_seed(777)

#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
config = tf.ConfigProto()
#config.gpu_options.allow_growth = False

def create_more_samples_using_augmentation(train_x, train_y, train_y_softmax, dataset):
    original_train_x = train_x
    original_train_y = train_y
    original_train_y_softmax = train_y_softmax
    train_x=[]; 
    train_x.extend(original_train_x)

    """scaling"""
    scales = [0.90, 0.75, 0.60]
    scaled_x = central_scale_images(original_train_x, scales)
    train_x.extend(scaled_x) 

    """translation""" 
    translate_x = translate_images(original_train_x)
    train_x.extend(translate_x) 
    
    rotate_x = rotate_images_with_finer_angles(original_train_x, -90, 90, 10)
    train_x.extend(rotate_x) 
    
    #flipping
    flip_x = flip_images(original_train_x)
    train_x.extend(flip_x) 
    
    #pepper noise
    if dataset=='cifar10':
       pepper_x = add_salt_pepper_noise(original_train_x)
       train_x.extend(pepper_x)
    
       gauss_x = add_gaussian_noise(original_train_x)
       train_x.extend(gauss_x)
       
    #scaling and translation
    translate_x1 = translate_images(scaled_x)
    train_x.extend(translate_x1) 
    
    #translation and rotation
    rotate_x1 = rotate_images_with_finer_angles(translate_x, -90, 90, 10)
    train_x.extend(rotate_x1) 

    #scaling and rotation
    rotate_x1 = rotate_images_with_finer_angles(scaled_x, -90, 90, 10)
    train_x.extend(rotate_x1) 
    
    #pepper and gaussian
    if dataset=='cifar10':
       gauss_x1 = add_gaussian_noise(pepper_x)
       train_x.extend(gauss_x1)

    train_x = np.array(train_x, dtype = np.float32)
    if dataset == 'cifar10':
        train_y_softmax = cifar_10_soft_logits_from_alexnet_teacher(train_x)

    train_y = np.argmax(train_y_softmax, axis=1)  
    train_y = to_categorical(train_y, NO_OF_CLASSES)
    
    #print(train_x.shape)
    #print(train_y.shape)
    #print(train_y_softmax.shape)
    return train_x, train_y, train_y_softmax

def cifar_10_soft_logits_from_alexnet_teacher(train_x):
    x, y, z, logits, y_pred_cls, global_step, is_training = model() 
    logits = logits/20.0
    softmax = tf.nn.softmax(logits=logits)
    _BATCH_SIZE = 1000
    _CLASS_SIZE = 10
    _SAVE_PATH = "./checkpoints/teacher/using_cifar10_(original_data)/lr_0.001/"
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    sess = tf.Session(config=config)
    try:
        #print("\nTrying to restore last checkpoint ...")
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
        saver.restore(sess, save_path=last_chk_path)
        #print("Restored checkpoint from:", last_chk_path)
    except ValueError:
        print("\nFailed to restore checkpoint. Initializing variables instead.")
    i = 0
    softmax_values = np.zeros(shape=(len(train_x), NO_OF_CLASSES), dtype=np.float)
    while i < len(train_x):
        j = min(i + _BATCH_SIZE, len(train_x))
        batch_xs = train_x[i:j, :]
        softmax_values[i:j] = sess.run(softmax, feed_dict={x: batch_xs, is_training:False})
        i = j

    #print("Done extracting softmax from teacher network.")
    return softmax_values

def central_scale_images(X_imgs, scales):
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([IMAGE_SIZE, IMAGE_SIZE], dtype = np.int32)
    
    X_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (1, IMAGE_SIZE, IMAGE_SIZE, NO_OF_CHANNELS))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        for index, img_data in enumerate(X_imgs):
            batch_img = np.expand_dims(img_data, axis = 0)
            scaled_imgs = sess.run(tf_img, feed_dict = {X: batch_img})
            X_scale_data.extend(scaled_imgs)
    
    X_scale_data = np.array(X_scale_data, dtype = np.float32)
    return X_scale_data

def get_translate_parameters(index):
    if index == 0: # Translate left 20 percent
        offset = np.array([0.0, 0.2], dtype = np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype = np.int32)
        w_start = 0
        w_end = int(ceil(0.8 * IMAGE_SIZE))
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 1: # Translate right 20 percent
        offset = np.array([0.0, -0.2], dtype = np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype = np.int32)
        w_start = int(floor((1 - 0.8) * IMAGE_SIZE))
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 2: # Translate top 20 percent
        offset = np.array([0.2, 0.0], dtype = np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype = np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = int(ceil(0.8 * IMAGE_SIZE)) 
    else: # Translate bottom 20 percent
        offset = np.array([-0.2, 0.0], dtype = np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype = np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = int(floor((1 - 0.8) * IMAGE_SIZE))
        h_end = IMAGE_SIZE 
        
    return offset, size, w_start, w_end, h_start, h_end

def translate_images(X_imgs):
    offsets = np.zeros((len(X_imgs), 2), dtype = np.float32)
    n_translations = 4
    X_translated_arr = []
    
    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_translations):
            X_translated = np.zeros((len(X_imgs), IMAGE_SIZE, IMAGE_SIZE, NO_OF_CHANNELS), dtype = np.float32)
            X_translated.fill(1.0) # Filling background color
            base_offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(i)
            offsets[:, :] = base_offset 
            glimpses = tf.image.extract_glimpse(X_imgs, size, offsets)
            
            glimpses = sess.run(glimpses)
            X_translated[:, h_start: h_start + size[0], w_start: w_start + size[1], :] = glimpses
            X_translated_arr.extend(X_translated)
    X_translated_arr = np.array(X_translated_arr, dtype = np.float32)
    return X_translated_arr

def rotate_images(X_imgs):
    X_rotate = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (IMAGE_SIZE, IMAGE_SIZE, NO_OF_CHANNELS))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k = k)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            for i in range(3):  # Rotation at 90, 180 and 270 degrees
                rotated_img = sess.run(tf_img, feed_dict = {X: img, k: i + 1})
                X_rotate.append(rotated_img)
        
    X_rotate = np.array(X_rotate, dtype = np.float32)
    return X_rotate

def rotate_images_with_finer_angles(X_imgs, start_angle, end_angle, n_images):
    X_rotate = []
    iterate_at = (end_angle - start_angle) / (n_images - 1)
    
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (None, IMAGE_SIZE, IMAGE_SIZE, NO_OF_CHANNELS))
    radian = tf.placeholder(tf.float32, shape = (len(X_imgs)))
    tf_img = tf.contrib.image.rotate(X, radian)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
    
        for index in range(n_images):
            degrees_angle = start_angle + index * iterate_at
            radian_value = degrees_angle * pi / 180  # Convert to radian
            radian_arr = [radian_value] * len(X_imgs)
            rotated_imgs = sess.run(tf_img, feed_dict = {X: X_imgs, radian: radian_arr})
            X_rotate.extend(rotated_imgs)

    X_rotate = np.array(X_rotate, dtype = np.float32)
    return X_rotate

def flip_images(X_imgs):
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (IMAGE_SIZE, IMAGE_SIZE, NO_OF_CHANNELS))
    tf_img1 = tf.image.flip_left_right(X)
    tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict = {X: img})
            X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype = np.float32)
    return X_flip

def add_salt_pepper_noise(X_imgs):
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = X_imgs.copy()
    row, col, _ = X_imgs_copy[0].shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))
    
    for X_img in X_imgs_copy:
        # Add Salt noise
        np.random.seed(777)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 1

        # Add Pepper noise
        np.random.seed(777)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 0
    return X_imgs_copy

def add_gaussian_noise(X_imgs):
    X_imgs = X_imgs.astype(np.float32)
    gaussian_noise_imgs = []
    row, col, _ = X_imgs[0].shape
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    
    for X_img in X_imgs:
        np.random.seed(777)
        gaussian = np.random.random((row, col, 1)).astype(np.float32)
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2).astype(np.float32)
        gaussian_img = cv2.addWeighted(X_img, 0.75, 0.25 * gaussian, 0.25, 0)
        gaussian_noise_imgs.append(gaussian_img)
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype = np.float32)
    return gaussian_noise_imgs

def tf_resize_images(X_img):
    X_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, NO_OF_CHANNELS))
    tf_img = tf.image.resize_images(X, (IMAGE_SIZE, IMAGE_SIZE), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for index, image in enumerate(X_img):
            img = image
            resized_img = sess.run(tf_img, feed_dict = {X: img})
            X_data.append(resized_img)

    X_data = np.array(X_data, dtype = np.float32) # Convert to numpy
    return X_data
