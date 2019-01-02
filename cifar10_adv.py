from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.loss import CrossEntropy
from cleverhans.dataset import CIFAR10
from cleverhans.utils_tf import model_eval
from cleverhans.train import train
from cleverhans.attacks import FastGradientMethod
from cleverhans.model_zoo.all_convolutional import ModelAllConvolutional
from cleverhans.augmentation import random_horizontal_flip, random_shift

from skimage import color
from skimage import io
from skimage import transform

path = "cifar10_image_new" #你所希望读的目录, 图片必须目前必须是28 * 28 * 1, rgb 0 ~ 1的黑白图片
NB_EPOCHS = 1
BATCH_SIZE = 128
LEARNING_RATE = 0.001
CLEAN_TRAIN = True
BACKPROP_THROUGH_ATTACK = False
NB_FILTERS = 64
config_args = {}
train_start=0
train_end=60000
test_start=0
test_end=10000
fgsm_params = {
  'eps': 0.3,
  'clip_min': 0.,
  'clip_max': 1.
}
train_params = {
    'nb_epochs': NB_EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE
}
rng = np.random.RandomState([2017, 8, 30])

def evaluate():
    print("end")

print("STEP 1: Get training data...")
data = CIFAR10(train_start=train_start, train_end=train_end,
              test_start=test_start, test_end=test_end)
dataset_size = data.x_train.shape[0]
dataset_train = data.to_tensorflow()[0]
dataset_train = dataset_train.map(
     lambda x, y: (random_shift(random_horizontal_flip(x)), y), 4)
dataset_train = dataset_train.batch(BATCH_SIZE)
dataset_train = dataset_train.prefetch(16)
x_train, y_train = data.get_set('train')
x_test, y_test = data.get_set('test')

img_rows, img_cols, nchannels = x_train.shape[1:4]
nb_classes = y_train.shape[1]
my_data = []

#读取图片
print("STEP 2: Get local data...")
all_filename = os.listdir(path)
for filename in all_filename:
    pic = io.imread(path + "/" + filename, plugin='matplotlib')
    pic = transform.resize(pic, (img_rows, img_cols) , preserve_range=True)
    my_data.append(pic)
my_data = np.array(my_data)
#要改变的图片格式入口
my_data = my_data.reshape((my_data.shape[0],
                           my_data.shape[1],
                           my_data.shape[2],
                           3))

#训练图片格式入口
print("STEP 3: Start training model...")
x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
sess = tf.Session(config=tf.ConfigProto(**config_args))
model = ModelAllConvolutional('model1', nb_classes, NB_FILTERS,
                              input_shape=[32, 32, 3])
preds = model.get_logits(x)
loss = CrossEntropy(model, smoothing=0.1)

train(sess, loss, x_train, y_train, evaluate=None,
      args=train_params, rng=rng, var_list=model.get_params())

fgsm = FastGradientMethod(model, sess=sess)
adv_x = fgsm.generate(x, **fgsm_params)
preds_adv = model.get_logits(adv_x)
adv_image = adv_x.eval(session=sess, feed_dict={x: my_data})
# 快速显示，debug用
# plt.imshow(adv_image[0,:,:,0])
# plt.show()

# 是否生成对抗图片
print("STEP 4: Build melicious data...")
printimage = True
name_num = 0
directory = "image_adv_" + str(datetime.datetime.now())
if printimage is True:
    if not os.path.exists(directory):
        os.makedirs(directory)
    for pic in adv_image:
        if pic.shape[2] == 1:
            io.imsave(directory +"/" + all_filename[name_num], pic[:,:,0])
        else:
            io.imsave(directory +"/" + all_filename[name_num], pic)
        name_num = name_num + 1
