import configparser
import numpy as np
import tensorflow as tf
import scipy.io as sio
from keras import backend as K
from keras.layers import Input, Flatten, Dense,ReLU, Dropout, Lambda, Layer,Conv2D,MaxPooling2D,UpSampling2D,Add,BatchNormalization,Multiply,Reshape
from keras.models import Model

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

def get_dataset_deviation(trial_data, base_data):
    new_dataset = np.empty([0, 128])
    for i in range(0, 4800):
        base_index = i // 120
        # print(base_index)
        base_index = 39 if base_index == 40 else base_index
        new_record = get_vector_deviation(trial_data[i], base_data[base_index]).reshape(1, 128)
        # print(new_record.shape)
        new_dataset = np.vstack([new_dataset, new_record])
    # print("new shape:",new_dataset.shape)
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

    # 读数据
    label= [1, 0, 2, 2, 0, 1, 2, 0, 1, 1, 0, 2, 0, 1, 2]
    Data_mat = sio.loadmat(data_dir + str(FileName) + '.mat')
    # print(FileName[file_num])
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