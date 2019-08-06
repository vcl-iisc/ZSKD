import numpy as np
import tensorflow as tf
tf.set_random_seed(777)
from time import time
import math
import os
import argparse
from include.data_new import get_data_set, shuffle_training_data, original_data_set, sampling
from data_augmentation import create_more_samples_using_augmentation, tf_resize_images,  cifar_10_soft_logits_from_alexnet_teacher
from tqdm import tqdm
from keras.utils import to_categorical

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = False


global_accuracy = 0
epoch_start = 0

desc = "Train teacher/student network"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--network', type=str, default='student', help='[teacher, student]')
parser.add_argument('--dataset', type=str, default='data_impressions', help='[cifar10, data_impressions]')
parser.add_argument('--resize', action='store_true', default =False, help="resizing the input image, model takes 32x32")
parser.add_argument('--suffix', type=str, default='T20_40000_lr_0.001_batch100_1500_iterations', help='(contain DI info like temp, sample count etc. or original data)')
parser.add_argument('--epoch', type=int, default=500, help='The number of epochs to run')
parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
parser.add_argument('--sampling', action='store_true', default =False, help="To sample the di's")
parser.add_argument('--data_augmentation', action='store_true', default =False, help="create more samples using data_augmentation")
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/', help='Directory name to save the checkpoints')

args = parser.parse_args()
if args is None:
   exit()

# PARAMS setting
_BATCH_SIZE = args.batch_size
_EPOCH = args.epoch


_SAVE_PATH = args.checkpoint_dir +str(args.network)+"/using_"+str(args.dataset)+"_("+str(args.suffix)+")/lr_"+str(args.lr)+"/"

initial_lr = args.lr
suffix = args.suffix

#default params
teacher_loss = True # for training with distillation
student_loss = False # for training with cross entropy
lambda_value = 0.30


folder_name = "../di_generation/alex_di/cifar_10/dirichlet/40000_di"
end_to_end_training = True

if args.network == 'teacher':
  from include.model_alex_full import model
  student_loss = True
  teacher_loss = False
elif args.network == 'student':
  from include.model_alex_half import model
else:
  print("Invalid Network")
  exit(1)

#data impressions data
if (args.dataset=='data_impressions' and args.network=='student'):
  train_x, train_y = get_data_set("train", args.suffix, folder_name)
  print("Training with DI's has shape: " +str(train_x.shape))
  print("###########################################")
  test_x, test_y = get_data_set("test", args.suffix, folder_name)
elif (args.dataset=='data_impressions' and args.network=='teacher'):
  print("Use Data Impressions to train the Student network")
  exit(1)

#original_data_set
if args.dataset=='cifar10':
  train_x, train_y = original_data_set("train", '','')
  print("Training with original data has shape: " + str(train_x.shape))
  test_x, test_y = original_data_set("test",'','')

if teacher_loss and args.dataset=='data_impressions':
  
  softmax_file_name = 'ySoft_'+str(suffix)+'.npy'	
  train_y_softmax = np.load(folder_name + '/'+ softmax_file_name)
  softmax_temperature = float(suffix.split('_')[0].split('T')[1])

else:
  train_y_softmax = np.zeros_like(train_y)
  softmax_temperature = 1.0

if args.dataset!='data_impressions' and teacher_loss:
  if args.dataset == 'cifar10':
    train_y_softmax = cifar_10_soft_logits_from_alexnet_teacher(train_x)

if (not np.any(train_y[:,1:])) and (args.dataset=='data_impressions'):
      train_y = np.argmax(train_y_softmax, axis=1)  
      train_y = to_categorical(train_y, 10)

if args.sampling:
      sampling_per_class = 2000
      train_x, train_y, train_y_softmax = sampling(train_x, train_y, train_y_softmax, sampling_per_class)
      train_x, train_y, train_y_softmax = shuffle_training_data(train_x, train_y, train_y_softmax)

if args.resize:
  train_x = tf_resize_images(train_x)
  test_x = tf_resize_images(test_x)

tf.reset_default_graph()

tf.set_random_seed(777)

x, y, z, logits, y_pred_cls, global_step, is_training = model() 

# LOSS
finalloss = 0.0

if student_loss:    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))   
    finalloss = loss  

if end_to_end_training:
    fc = None    

if teacher_loss:
     softmax_temperature = 20.0            
     teacher_loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits/softmax_temperature, labels=z))
     finalloss = tf.reduce_mean(lambda_value * finalloss + tf.square(softmax_temperature)*teacher_loss_value) 

#OPTIMIZER
optimizer = tf.train.AdamOptimizer(learning_rate=initial_lr).minimize(finalloss, var_list=fc, global_step=global_step)

# PREDICTION AND ACCURACY CALCULATION
correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# SAVER
with tf.name_scope('summaries'): 
	tf.summary.scalar('Training loss', finalloss)
	
merged = tf.summary.merge_all()
saver = tf.train.Saver()

sess = tf.Session(config=config)

train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)

saver_model = tf.train.Saver()

sess.run(tf.global_variables_initializer())
    

def train(epoch):
    global epoch_start
    epoch_start = time()
    batch_size = int(math.ceil(len(train_x) / _BATCH_SIZE))
    
    i_global = 0
    train_x1, train_y1, train_y_softmax1 = shuffle_training_data(train_x, train_y, train_y_softmax)
    print("Epoch : " + str(epoch+1))

    for s in tqdm(range(batch_size)):
        
        batch_xs = train_x1[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
        batch_ys = train_y1[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
        batch_ysoft = train_y_softmax1[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]

        if args.data_augmentation:
             batch_xs, batch_ys, batch_ysoft = create_more_samples_using_augmentation(batch_xs, batch_ys, batch_ysoft, 'cifar10')

        start_time = time()
        summary, i_global, _, batch_loss, batch_acc = sess.run(
            [merged, global_step, optimizer, finalloss, accuracy],
            feed_dict={x: batch_xs, y: batch_ys, z: batch_ysoft})
        #print("the training accuracy for epoch" + str(epoch)+" : "+ str(batch_acc*100))
        train_writer.add_summary(summary, i_global)
        duration = time() - start_time

        if s % 10 == 0:
            percentage = int(round((s*1.0/batch_size)*100))

            bar_len = 29
            filled_len = int((bar_len*int(percentage))/100)
            bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len)

            msg = "Global step: {:>5} - [{}] {:>3}% - acc: {:.4f} - loss: {:.4f} - {:.1f} sample/sec"
            #print(msg.format(i_global, bar, percentage, batch_acc, batch_loss, _BATCH_SIZE / duration))
    
    test_and_save(i_global, epoch)


def test_and_save(_global_step, epoch):
    global global_accuracy
    global epoch_start

    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j] = sess.run(
            y_pred_cls,
            feed_dict={x: batch_xs, y: batch_ys, is_training:False}
        )
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean()*100
    correct_numbers = correct.sum()

    hours, rem = divmod(time() - epoch_start, 3600)
    minutes, seconds = divmod(rem, 60)
    mes = "\nEpoch {} - accuracy: {:.2f}% ({}/{}) - time: {:0>2}:{:0>2}:{:05.2f}"
    #print(mes.format((epoch+1), acc, correct_numbers, len(test_x), int(hours), int(minutes), seconds))
    
    
    if global_accuracy != 0 and global_accuracy < acc:

        saver.save(sess, save_path=_SAVE_PATH, global_step=_global_step)

        mes = "The epoch {} receive better accuracy: {:.2f} > {:.2f}. Saving session..."
        print(mes.format((epoch+1), acc, global_accuracy))
        global_accuracy = acc

    elif global_accuracy == 0:
        global_accuracy = acc

    #print("###########################################################################################################")


def main():

    train_start = time()
    print("Learning rate:"+str(args.lr))
    print("Suffix:"+str(args.suffix))
    print("##############################################")
    for i in range(_EPOCH):
        train(i)
        

    hours, rem = divmod(time() - train_start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("###########################################################################################################")
    print("Learning rate:"+str(args.lr))
    print("Suffix:"+str(args.suffix))
    mes = "Best accuracy : {:.2f}, time: {:0>2}:{:0>2}:{:05.2f}"
    print(mes.format(global_accuracy, int(hours), int(minutes), seconds))
    print("###########################################################################################################")
        

if __name__ == "__main__":
    main()


sess.close()
