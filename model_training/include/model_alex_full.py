import tensorflow as tf
tf.set_random_seed(777)
import numpy as np
from tensorflow.contrib.layers import flatten
from tensorflow.python.tools import inspect_checkpoint
import tensorflow.contrib as tf_contrib

def maxpool(x, ksize, strides, padding = "SAME"):
    """max-pooling layer"""
    return tf.nn.max_pool(x, 
                          ksize = [1, ksize, ksize, 1], 
                          strides = [1, strides, strides, 1], 
                          padding = padding, 
                          name='maxpooling')

def dropout(x, rate, is_training):
    """drop out layer"""
    return tf.layers.dropout(x, rate, name='dropout', training = is_training)

def lrn(x, depth_r=2, alpha=0.0001, beta=0.75, bias=1.0):
    """local response normalization"""
    return tf.nn.local_response_normalization(x, 
                                              depth_radius = depth_r, 
                                              alpha = alpha, 
                                              beta = beta, 
                                              bias = bias, 
                                              name='lrn')

def fc(x, output_size, name, activation_func=tf.nn.relu):
    """fully connected layer"""
    with tf.variable_scope(name):
        input_size = x.get_shape().as_list()[-1]
        w = tf.Variable(tf.random_normal([input_size, output_size], 
                        dtype=tf.float32, 
                        stddev=0.01), 
                        name='weights')
        b = tf.Variable(tf.constant(value=0.0, 
                        dtype = tf.float32, 
                        shape=[output_size]), 
                        name='bias')
    
        out = tf.nn.xw_plus_b(x, w, b)
        if activation_func:
            return activation_func(out)
        
        else:
            return out

def conv(x, ksize, strides, output_size, name, activation_func=tf.nn.relu, padding = "SAME", bias=0.0):
    """conv layer"""
    with tf.variable_scope(name):
        input_size = x.get_shape().as_list()[-1]
        w = tf.Variable(tf.random_normal([ksize, ksize, input_size, output_size], 
                        dtype=tf.float32, 
                        stddev=0.01), 
                        name='weights')
        b = tf.Variable(tf.constant(value=bias, 
                        dtype=tf.float32, 
                        shape=[output_size]), 
                        name='bias')
        
        conv = tf.nn.conv2d(x, w, [1, strides, strides, 1], padding=padding)
        conv = tf.nn.bias_add(conv, b)
        
        if activation_func:
            conv = activation_func(conv)
            
        return conv

def batch_norm(x, name, is_training):
    with tf.variable_scope(name):
        return tf_contrib.layers.batch_norm(x, center=True, scale=True, is_training=is_training, updates_collections=None)

def model():
    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10

    with tf.variable_scope('teacher_alex'):
        keepPro = 0.5
        #temperature = 1.0
        x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE , _IMAGE_SIZE , _IMAGE_CHANNELS], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        z = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')

        is_training = tf.placeholder_with_default(True, shape=())

        x_image = x

        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')

        with tf.name_scope('conv1layer'):
            conv1 = conv(x=x_image, ksize=5, strides=1, output_size=48, name='conv1') # output[32,32,48]
            conv1 = lrn(conv1)
            #print('conv1:{}'.format(conv1.get_shape().as_list()))
            conv1 = maxpool(conv1, ksize=3, strides=2, padding='VALID') # output[15,15,48]
            #print('maxpool1:{}'.format(conv1.get_shape().as_list()))
            conv1 = batch_norm(conv1, 'bn1', is_training)
            #print(conv1)
        
        # layer 2
        with tf.name_scope('conv2layer'):
            conv2 = conv(x=conv1, ksize=5, strides=1, output_size=128, bias=1.0, name='conv2') # output[15,15,128]
            conv2 = lrn(conv2)
            #print('conv2:{}'.format(conv2.get_shape().as_list()))
            conv2 = maxpool(conv2, ksize=3, strides=2, padding='VALID') # output[7,7,128]
            #print('maxpool2:{}'.format(conv2.get_shape().as_list()))
            conv2 = batch_norm(conv2, 'bn2', is_training)
        
        # layer 3
        with tf.name_scope('conv3layer'):
            conv3 = conv(x=conv2, ksize=3, strides=1, output_size=192, name='conv3') # output[7,7,192]
            #print('conv3:{}'.format(conv3.get_shape().as_list()))
            conv3 = batch_norm(conv3, 'bn3', is_training)
        
        # layer 4
        with tf.name_scope('conv4layer'):
            conv4 = conv(x=conv3, ksize=3, strides=1, output_size=192, bias=1.0, name='conv4') # output[7,7,192]
            #print('conv4:{}'.format(conv4.get_shape().as_list()))
            conv4 = batch_norm(conv4, 'bn4', is_training)
        
        # layer 5
        with tf.name_scope('conv5layer'):
            conv5 = conv(x=conv4, ksize=3, strides=1, output_size=128, bias=1.0, name='conv5') # output[7,7,128]
            #print('conv5:{}'.format(conv5.get_shape().as_list()))
            conv5 = maxpool(conv5, ksize=3, strides=2, padding='VALID') #output[3,3,128]
            #print('maxpool5:{}'.format(conv5.get_shape().as_list()))
            conv5 = batch_norm(conv5, 'bn5', is_training)
        
        # flatten
        conv5size = conv5.get_shape().as_list()
        conv5 = tf.reshape(conv5, [-1, conv5size[1] * conv5size[2] * conv5size[3]])
        #print('flatten:{}'.format(conv5.get_shape().as_list()))

        # layer 6
        with tf.name_scope('fc1layer'):
            fc1 = fc(x=conv5, output_size=512, name='fc1')
            #print('fc1:{}'.format(fc1.get_shape().as_list()))
            fc1 = dropout(fc1, keepPro, is_training)
            fc1 = batch_norm(fc1, 'bn6', is_training)

        # layer 7
        with tf.name_scope('fc2layer'):
            fc2 = fc(x=fc1, output_size=256, name='fc2')
            #print('fc2:{}'.format(fc2.get_shape().as_list()))
            fc2 = dropout(fc2, keepPro, is_training)
            fc2 = batch_norm(fc2, 'bn7', is_training)

        # layer 8 - output
        with tf.name_scope('fc3layer'):
            logits = fc(x=fc2, output_size=10, activation_func=None, name='fc3')

        with tf.variable_scope('softmax'):
            softmax = tf.nn.softmax(logits=logits)

    y_pred_cls = tf.argmax(softmax, axis=1)

    return x, y, z, logits, y_pred_cls, global_step, is_training


