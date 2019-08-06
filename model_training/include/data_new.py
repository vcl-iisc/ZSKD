import pickle
import numpy as np
import os
import tarfile
import zipfile
import sys
from keras.utils import to_categorical
import sys

def rescale(values, new_min = 0, new_max = 255):
    output = []
    old_min, old_max = np.min(values), np.max(values)

    for v in values:
        new_v = (new_max - new_min) / (old_max - old_min) * (v - old_min) + new_min
        output.append(new_v)

    return output

def get_data_set(name="train", suffix='', folder_name=''):

    data_file_name = 'X_'+ suffix +'.npy'
    label_file_name = 'y_'+ suffix +'.npy'
    
    x = None
    y = None

    if name is "train":

            _X = np.load(folder_name + '/'+ data_file_name)
            _X = np.array(_X, dtype=float)
            
            if not os.path.exists(folder_name + '/'+ label_file_name):
                _Y = np.zeros((_X.shape[0],1))
            else:
                _Y = np.load(folder_name + '/'+ label_file_name)

            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)

    elif name is "test":
        f = open('./data_set/cifar_10/test_batch', 'rb')
        datadict = pickle.load(f)
        f.close()
   
        x = datadict["data"]
        x = np.array(x, dtype=float)          
        y = np.array(datadict['labels'])
        
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = np.array(x, dtype=float) / 255.0

    return x, to_categorical(y, 10)


def original_data_set(name="train", suffix='', folder_name=''):
    
    x = None
    y = None

    folder_name = "cifar_10"

    if name is "train":
        for i in range(5):
            f = open('./data_set/'+folder_name+'/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f)
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']

            _X = np.array(_X, dtype=float)
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])
            
            _X = np.array(_X, dtype=float) / 255.0

            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)

    elif name is "test":
        f = open('./data_set/'+folder_name+'/test_batch', 'rb')
        datadict = pickle.load(f)
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        x = np.array(x, dtype=float) 
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = np.array(x, dtype=float) / 255.0 

    return x, to_categorical(y, 10)

def shuffle_training_data(train_data, train_labels, train_softmax_values):
    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)
    np.random.seed(seed)
    np.random.shuffle(train_softmax_values)
    return train_data, train_labels, train_softmax_values


def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot

def sampling(train_x, train_y, train_y_softmax, sample_per_class, num_classes=10):
    train_y = np.argmax(train_y, axis=1)
    #seed = 777
    #np.random.seed(seed)
    data = list(zip(train_x, train_y, train_y_softmax))
    data = sorted(data, key=lambda x: x[1])
    data = np.array(data)
    unique, counts = np.unique(data[:,1], return_counts=True)
    #print np.asarray((unique, counts)).T
    num_classes = len(unique)
    data_per_class = counts
    end = 0
    sampled_data =[]
    for index, class_no in enumerate(range(num_classes)):
        start = end
        end = start + data_per_class[index]
        class_data = data[start:end]
        
        if data_per_class[index] >= sample_per_class:
            replace = False
        else:
            replace = True 
        seed = 777
        np.random.seed(seed)
        sampled_class_data = np.random.choice(data_per_class[index], size=sample_per_class, replace=replace)
        sampled_data.append(class_data[sampled_class_data])

    sampled_data = np.concatenate(sampled_data, axis=0)
    train_x_subset = np.array(sampled_data[:,0].tolist())
    train_y_subset = to_categorical(np.array(sampled_data[:,1].tolist()), 10)
    train_y_softmax_subset = np.array(sampled_data[:,2].tolist())

    return train_x_subset, train_y_subset, train_y_softmax_subset





