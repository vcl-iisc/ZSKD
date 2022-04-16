# ZSKD
Zero-Shot Knowledge Distillation in Deep Networks

**Paper link** : [http://proceedings.mlr.press/v97/nayak19a/nayak19a.pdf](http://proceedings.mlr.press/v97/nayak19a/nayak19a.pdf)

**Presentation slides link** : [https://icml.cc/media/Slides/icml/2019/grandball(13-11-00)-13-11-30-4371-zero-shot_knowl.pdf](https://icml.cc/media/Slides/icml/2019/grandball(13-11-00)-13-11-30-4371-zero-shot_knowl.pdf)

**Poster link** : [https://drive.google.com/file/d/1ZMCUPnJ3epCtLov26mVttmJT5OQB2HwK/view?usp=sharing](https://drive.google.com/file/d/1ZMCUPnJ3epCtLov26mVttmJT5OQB2HwK/view?usp=sharing)

## Dependencies
- Python 2.7
- tensorflow-gpu 1.10.0
- tensorboard 1.10.0
- cudatoolkit 9.0
- cudnn 7.3.1
- tqdm 4.32.2
- keras-gpu 2.2.4
- numpy 1.15.4


## How to use this code:

*The cifar 10 dataset is available at*:

[Google Drive](https://drive.google.com/drive/folders/12mTAIrxSEGQthor3eFO4aBm6-bS4X-sI?usp=sharing)

Copy the cifar 10 folder from the above link and put it in the `model_training/dataset/` folder

Go to the folder "model_training"

* Step 1 : Train the Teacher network with cifar 10

```
 CUDA_VISIBLE_DEVICES=0 python train.py --network teacher --dataset cifar10 --suffix original_data --epoch 1000 --batch_size 512 
```
The pretrained teacher model weights are also kept in `checkpoints/teacher/` folder.

* Step 2 : Extract final layer weights from the Pretrained Teacher Network

Make sure the checkpoint and meta graph path is correct in the extract_weights.py script.

```
 python extract_weights.py
```

* Step 3 : Compute and save the Class Similarity for scales of 1.0 and 0.1

Go to the folder `di_generation/`

```
 python dirichmat.py
```

Two files with name "visualMat_alexnet_cifar10_scale_1.pickle" and "visualMat_alexnet_cifar10_scale_0.1.pickle" will get saved in the same directory

* Step 4 : Generate the Data Impressions (DI's)

```
 python cifar_10_impressions_alexnet_Dirch.py
```

40000 Di's will be saved in the folder `alex_di/cifar_10/dirichlet/40000_di/`

*The sample generated DI's are also available at* : 

https://drive.google.com/drive/folders/1nsQfzQQh6GTZU5XHd0YztFkw_JDuXO7x?usp=sharing 


* Step 5 : Train the Student network with generated DI's

```
 CUDA_VISIBLE_DEVICES=0 python train.py --network student --dataset data_impressions --data_augmentation
```

## Citing
If you use this code, please cite our work:

```text
@inproceedings{
nayak2019zero,
title={Zero-Shot Knowledge Distillation in Deep Networks},
author={Nayak, G. K., Mopuri, K. R., Shaj, V., Babu, R. V., and Chakraborty, A.},
booktitle={International Conference on Machine Learning},
pages={4743--4751},
year={2019}
}
```
