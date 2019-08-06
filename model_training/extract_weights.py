import argparse
import tensorflow as tf
import os
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

checkpoint_dir = 'checkpoints/teacher/using_cifar10_(original_data)/lr_0.001/'

def main():
    data={}
    
    with tf.Session(config=config) as sess:

        new_saver = tf.train.import_meta_graph('checkpoints/teacher/using_cifar10_(original_data)/lr_0.001/-27257.meta')
        new_saver.restore(sess,tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir))
        
        weights = tf.get_default_graph().get_tensor_by_name('teacher_alex/fc3layer/fc3/weights:0')
        bias = tf.get_default_graph().get_tensor_by_name('teacher_alex/fc3layer/fc3/bias:0')

        data['weight']= sess.run(weights)
        data['bias'] = sess.run(bias)
        savefilename = "../di_generation/weight_and_bias_of_last_layer_alexnet_teacher_cifar10.pickle"

        pickle_out = open(savefilename,"wb")
        pickle.dump(data, pickle_out)
        pickle_out.close()
        

if __name__ == '__main__':
    main()
