import configparser
import numpy as np
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
import scipy.io as sio
import keras
import argparse
from keras import backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.layers import Input, Flatten, Dense,ReLU, Dropout, Lambda, Layer,Conv2D,MaxPooling2D,UpSampling2D,Add,BatchNormalization,Multiply,Reshape
from keras.models import Model
from Utils import *


parser = argparse.ArgumentParser()
parser.add_argument("-c", type = str, help = "configuration file", required = True)
parser.add_argument("-g", type = str, help = "GPU number to use, set '-1' to use CPU", required = True)
args = parser.parse_args()
Path, cfgTrain, cfgModel = ReadConfig(args.c)

# set GPU number or use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = args.g
if args.g != "-1":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    KTF.set_session(sess)
    print("Use GPU #"+args.g)
else:
    print("Use CPU only")

# ## 1.2. Analytic parameters

# [train] parameters
channels   = int(cfgTrain["channels"])
fold       = int(cfgTrain["fold"])
context    = int(cfgTrain["context"])
num_epochs = int(cfgTrain["epoch"])
batch_size = int(cfgTrain["batch_size"])
optimizer  = cfgTrain["optimizer"]
learn_rate = float(cfgTrain["learn_rate"])
lr_decay   = float(cfgTrain["lr_decay"])
# trail      = cfgTrain["trail"]

# [model] parameters
# dense_size            = np.array(str.split(cfgModel["Globaldense"],','),dtype=int)
num_EEG_channel       = int(cfgModel["num_EEG_channel"])
num_class             = int(cfgModel["num_class"])

l1                    = float(cfgModel["l1"])
l2                    = float(cfgModel["l2"])
dropout               = float(cfgModel["dropout"])

# ## 1.3. Parameter check and enable

# check optimizer（opt）
if optimizer=="adam":
    opt = keras.optimizers.Adam(lr=learn_rate,decay=lr_decay)
elif optimizer=="RMSprop":
    opt = keras.optimizers.RMSprop(lr=learn_rate,decay=lr_decay)
elif optimizer=="SGD":
    opt = keras.optimizers.SGD(lr=learn_rate,decay=lr_decay)
else:
    assert False,'Config: check optimizer'

# set l1、l2（regularizer）
if l1!=0 and l2!=0:
    regularizer = keras.regularizers.l1_l2(l1=l1, l2=l2)
elif l1!=0 and l2==0:
    regularizer = keras.regularizers.l1(l1)
elif l1==0 and l2!=0:
    regularizer = keras.regularizers.l2(l2)
else:
    regularizer = None


def ReadConfig(configfile):
    config = configparser.ConfigParser()
    print('Config: ', configfile)
    config.read(configfile)
    cfgPath = config['path']
    cfgTrain = config['train']
    cfgModel = config['model']
    return cfgPath, cfgTrain, cfgModel


def get_vector_deviation(vector1, vector2):
    return vector1 - vector2

#remove base
def get_dataset_deviation(trial_data, base_data):
    new_dataset = np.empty([0, 128])
    for i in range(0, 4800):
        base_index = i // 120
        base_index = 39 if base_index == 40 else base_index
        new_record = get_vector_deviation(trial_data[i], base_data[base_index]).reshape(1, 128)
        new_dataset = np.vstack([new_dataset, new_record])
    return new_dataset

def norm(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean)/std


def cross_entropy_loss(y_true, y_pred):
    loss = tf.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    return loss



#DEAP experimental setup
def prepare_dataset(FileName, data_dir, norm=True):

    Train_Data = np.empty([0, 62, 5])
    Train_Label = np.array([])
    Test_Data = np.empty([0, 62, 5])
    Test_Label = np.array([])
    All_Data = np.empty([0, 62, 5])

    label= []
    Data_mat = sio.loadmat(data_dir + str(FileName) + '.mat')
    for trail in range(9):
        Data = Data_mat['de_LDS' + str(trail+1)]

        Data = Data.transpose([1,0,2])
        All_Data = np.vstack([All_Data, Data])
        Train_Data = np.vstack([Train_Data, Data])
        Train_Label = np.append(Train_Label, [label[trail]] * Data.shape[0])

    for trail in range(9,15):
        Data = Data_mat['de_LDS' + str(trail+1)]
        Data = Data.transpose([1,0,2])
        Test_Data = np.vstack([Test_Data, Data])
        Test_Label = np.append(Test_Label, [label[trail]]* Data.shape[0])

def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return contrastive_loss


def my_categorical_crossentropy(pair_label):
    def categorical_crossentropy(y_true, y_pred):
        log_pred = K.log(K.softmax(y_pred))
        my_loss = Reshape((-1, 1))(-K.sum(y_true * log_pred, axis=1))
        loss = K.dot(my_loss, pair_label)
        return K.mean(loss)

    return categorical_crossentropy



def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.mean(K.square(x - y), axis=1, keepdims=False)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

euclidean_dis_layer_lambda_eeg_eog = Lambda(euclidean_distance, name="output_dist",
                                            output_shape=eucl_dist_output_shape)


class fusion_block(Layer):
    def __init__(self, **kwargs):
        super(fusion_block, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w1 = self.add_weight(name='ratio_eeg',
                                          shape=(1, 1),
                                          initializer='random_uniform',
                                          trainable=True)
        self.w2 = self.add_weight(name='ratio_eog',
                                          shape=(1, 1),
                                          initializer='random_uniform',
                                          trainable=True)
        super(fusion_block, self).build(input_shape)

    def call(self, x):
        eeg, eog = x
        fusion = K.dot(eeg, self.w1) + K.dot(eog, self.w2)
        return fusion

    def compute_output_shape(self, input_shape):
        eeg_shape, eog_shape = input_shape
        return eeg_shape


def Fusion_Model():
    # eeg_date,eye_data = fusion_data
    data_eeg = Input(shape=(16, 1), name='fusion_eeg')
    data_eog = Input(shape=(16, 1), name='fusion_eog')
    block = fusion_block()([data_eeg, data_eog])
    fusion_model = Model(inputs=[data_eeg, data_eog], outputs=block)
    return fusion_model