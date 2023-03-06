###########################################
####  Script used to run the training  ####
###########################################

import numpy as np
import tensorflow as tf
from IPython.display import clear_output
import datetime
import os
import time

import utils
import segnet
import psrlog as psg
from callbacks import DisplayCallback
from metrics_cm import ConfusionMatrixMetric


# simple way to add weight to class
def add_sample_w(img,label):
    cw = tf.constant([1.0, 3.0])
    cw = cw/tf.reduce_sum(cw)
    sw = tf.gather(cw,indices=tf.cast(label,tf.int32))
    return img,label,sw

n_classes = 2
batch_size = 4
max_dimension = None
max_tsub = False
batch_nb = 4700

epochs = 30

log_dir = "./dataset_2/last_net" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

#filelist containing the files used for each dataset
Img1 = np.loadtxt("./dataset_2/x_train", dtype="str")
Img2 = np.loadtxt("./dataset_2/x_val", dtype="str")
Mask1 = np.loadtxt("./dataset_2/y_train", dtype="str")
Mask2 = np.loadtxt("./dataset_2/y_val", dtype="str")


#train and validation generator 
tg = psg.ImageGenerator(imgs=Img1, masks=Mask1, batch_size=batch_size, batch_nb=batch_nb,shuffle=True, max_dimension=max_dimension, check_file=False)
vg = psg.ImageGenerator(imgs=Img2, masks=Mask2, batch_size=batch_size, batch_nb=batch_nb,shuffle=True, max_dimension=max_dimension, check_file=False)
train = tf.data.Dataset.from_generator(tg,(tf.float64, tf.float64), (tf.TensorShape([None,None,None,3]), tf.TensorShape([None,None,None,1])))
val = tf.data.Dataset.from_generator(vg,(tf.float64, tf.float64), (tf.TensorShape([None,None,None,3]), tf.TensorShape([None,None,None,1])))


metrics = [ConfusionMatrixMetric(2)]
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model = segnet.build_model(n_classes=n_classes, padding='same', metrics=metrics)
start = time.time()

train_batches = train.map(add_sample_w)
validation_batches = val.map(add_sample_w)
model.fit(train_batches, epochs=epochs, callbacks=[DisplayCallback(), tensorboard_callback], validation_steps=4, validation_data=validation_batches)
print("It tooks %f seconds to train the network" % (time.time() - start))
model.save("network")

