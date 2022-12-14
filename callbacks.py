from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, Callback
from IPython.display import clear_output

class DisplayCallback(tf.keras.callbacks.Callback):
    """
        Callback class to print the epoch at the end of each one.
    """
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
