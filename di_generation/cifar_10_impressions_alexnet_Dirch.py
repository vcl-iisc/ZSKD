import sys
import matplotlib.pyplot as plt
import pylab
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.datasets import cifar10
from keras.utils import to_categorical
import argparse
from model_alex_full import *
import tensorflow as tf
tf.set_random_seed(777)
import numpy as np
import time
import math
import pickle
import os
import sys
import argparse
from tqdm import tqdm

desc = "Generate Data Impressions"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--iterations', type=int, default=1500, help='The number of epochs to run')
parser.add_argument('--batch_size', type=int, default=100, help='The size of batch')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

args = parser.parse_args()
if args is None:
   exit()


def generateDirch(batch_size,c,scale, simMat):
	x=[]
	sim=simMat[c,:] 
	for b in range(batch_size):
		#temp=sim*scale
		temp=(sim-np.min(sim))/(np.max(sim)-np.min(sim))
		temp=temp*scale + 0.0001
		x.append(np.random.dirichlet(temp))
	x=np.array(x)

	return x


## Setting the configurations
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
batch_size = args.batch_size
learning_rate = args.lr
size=32

# noise create and augment
noise_l = tf.placeholder(tf.float32, shape=[batch_size,1])
noise_l_strip = tf.cast(tf.squeeze(noise_l, axis = 1), tf.int32)
scale_f  = tf.placeholder(tf.float32, shape = [1])
rotate_f = tf.placeholder(tf.float32, shape = [batch_size])


noise_image = tf.Variable(tf.random_uniform([batch_size,size,size,3],minval=0,maxval=1), name='noise_image', dtype='float32')

noise_image_clip  = noise_image.assign((noise_image-tf.reduce_min(noise_image))/(tf.reduce_max(noise_image)-tf.reduce_min(noise_image)))

noise_image_cropped = noise_image 

sess = tf.Session(graph = tf.get_default_graph(), config=config)

x, y, z, net_logits, y_pred_cls, global_step, is_training = model(noise_image_cropped)
print("#################################")

net_logits = net_logits/20.0    #using temp 20
net_softmax = tf.nn.softmax(net_logits)

################################## Generate Dirch ###################################################################

net_softmax_dirich = tf.placeholder(tf.float32, shape=[batch_size,10])

##################################Loss Optimizer and Update Steps ###################################################

sess.run(noise_image.initializer)

tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'noise_image') 

noise_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= net_logits, labels = net_softmax_dirich ))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

grads = optimizer.compute_gradients(noise_cost,tvars)
update = optimizer.apply_gradients(grads)
sess.run(tf.global_variables_initializer())


############################################Integrate CIFAR MODEL IN HERE##################################################
res_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "teacher_alex") 

saver = tf.train.Saver(res_vars)


_SAVE_PATH = '../model_training/checkpoints/teacher/using_cifar10_(original_data)/lr_0.001/'

try:
    print("\nTrying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except ValueError:
    print("\nFailed to restore checkpoint. Initializing variables instead.")


def createClassImpressions(num_samples,batch_size,max_iter,dType, rows, simMat):

	noise_image_all = list()
	label_image_all = list()
	softmax_image_all = list()
	for row_no in range(rows):
                print("#################################")
                if row_no < 10:
                    print("Creating samples for Class : " +str(row_no+1)+" with scaling factor (beta)=1.0")
                else:
                    print("Creating samples for Class : " +str(row_no+1)+" with scaling factor (beta)=0.1")
                sess.run(noise_image.initializer)
		train_confid = 0
		train_pred = row_no+1
		
		noise_l_train = row_no*np.ones((batch_size,1 ))
		for sam_no in tqdm(range(num_samples)):
                      
			softmax_generated = generateDirch(batch_size,row_no,1, simMat) 
			for j in range(max_iter):
				
				rand_rotate_value = 0.0174533* np.random.randint(low =-5, high =6, size= batch_size)
				rand_scale_value = np.random.choice([ 1, 0.975, 1.025, 0.95, 1.05], size = 1 )
				if j<max_iter-1:
					_, _, cost, train_logits, train_softmax = sess.run([noise_image_clip, update, noise_cost, net_logits, net_softmax], feed_dict={ net_softmax_dirich: softmax_generated, scale_f: rand_scale_value, rotate_f: rand_rotate_value})
                                        
				else: 
					train_noise_image, _, train_noise_loss, train_softmax= sess.run([noise_image, update, noise_cost, net_softmax], feed_dict={net_softmax_dirich: softmax_generated, scale_f: rand_scale_value, rotate_f: rand_rotate_value})
					noise_image_all.append(train_noise_image)
					softmax_image_all.append(train_softmax)
					print 'row', row_no, 'iter', j, 'cost', train_noise_loss
					break
	x=np.array(noise_image_all)
	x=np.reshape(x,[x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4]])

	yS=np.array(softmax_image_all)

	yS=np.reshape(yS,[yS.shape[0]*yS.shape[1],yS.shape[2]])
	print(x.shape,yS.shape)
	
	np.save('alex_di/cifar_10/dirichlet/40000_di/X_'+ dType+'.npy', x)
	np.save('alex_di/cifar_10/dirichlet/40000_di/ySoft_'+ dType+'.npy', yS)


def main():

	e=pickle.load(open('visualMat_alexnet_cifar10_scale_1.pickle','rb'))
	f=pickle.load(open('visualMat_alexnet_cifar10_scale_0.1.pickle','rb'))

	simMat = np.concatenate([e,f]); rows=20; 

	createClassImpressions(20,args.batch_size,args.iterations,'T20_40000_lr_'+str(args.lr)+"_batch"+str(args.batch_size)+"_"+str(args.iterations)+"_iterations", rows, simMat)        

if __name__ == "__main__":
    main()



