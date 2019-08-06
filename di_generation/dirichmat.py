import pickle
import sklearn.preprocessing as pp
import numpy as np
import sklearn.metrics.pairwise as pair

fc_weight=pickle.load(open('weight_and_bias_of_last_layer_alexnet_teacher_cifar10.pickle','rb'))
fc_weight=fc_weight['weight']
fc_weight_norm =  pp.normalize(fc_weight,axis=0)
im_sim_mat = np.matmul(np.transpose(fc_weight_norm),fc_weight_norm)

import pickle
import sklearn.preprocessing as pp
import numpy as np

norm1=(im_sim_mat-np.min(im_sim_mat))/(np.max(im_sim_mat)-np.min(im_sim_mat))

pickle.dump(norm1,open('visualMat_alexnet_cifar10_scale_1.pickle','wb'),protocol=2)
pickle.dump(norm1/10,open('visualMat_alexnet_cifar10_scale_0.1.pickle','wb'),protocol=2)

