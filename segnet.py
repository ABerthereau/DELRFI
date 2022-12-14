import tensorflow as tf
from typing import Optional, Union, Callable, List
#import tensorflow_datasets as tfds
#tfds.disable_progress_bar()
from callbacks import DisplayCallback
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input
from tensorflow.keras.layers import Input,GlobalMaxPooling2D,Dense,Conv2D, Conv2DTranspose, MaxPool2D
from utils import to_rgb
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
from os import listdir, path

def _downstack(inputs_shape: Optional[tuple] = (None,None,3),
               alpha: Optional[float] = 0.35,
               n_classes: int = 2):
    """
    Download the pre-trained MobileNetV2 layers to be used as downstack for the U-net model.
    Inputs:
        -inputs_shape: tuple - shape of the input image.
        -alpha: float - between 0 and 1. controls the width of the network. alpha < 1.0 proportionnaly decreases the number
        of filters, alpha > 1.0 increases the number of filters in each layer.
        -n_classes: integer - number of classe to classify
    Ouputs:
        -down_stack: list - contains the layer of MobileNetV2 used for downsampling.
    """

    inputs_ = Input(shape=inputs_shape)
    base_model = MobileNetV2(include_top=False, alpha=alpha, weights='imagenet', input_tensor=inputs_, classes=n_classes,classifier_activation='sigmoid')
    layer_names = [
      'block_1_expand_relu',   # 64x64
      'block_3_expand_relu',   # 32x32
      'block_6_expand_relu',   # 16x16
      'block_13_expand_relu',  # 8x8
      'block_16_project',      # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False

    return down_stack



def _upstack():
    """
        Download and use the pix2pix pre-trained model as upstack for U-net model.
        Outputs:
            -up_stack: list - contains the layer used to upsample by 2D transpose convolution.
    """
    up_stack = [
       upsample(512, 3),  # 4x4 -> 8x8
       upsample(256, 3),  # 8x8 -> 16x16
       upsample(128, 3),  # 16x16 -> 32x32
       upsample(64, 3),   # 32x32 -> 64x64
    ]

    return up_stack

def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
  """Upsamples an input.
  Conv2DTranspose => Batchnorm => Dropout => Relu
  Inputs:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer
  Ouputs:
    Upsample Sequential Model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def build_model(inputs_shape : Optional[tuple] = (None,None,3),
                alpha: Optional[float] = 0.35,
                n_classes: int = 2,
                padding: str = "same",
                optimizer: str = "adam",
                metrics: List[Union[str, Callable]] = "accuracy",
                loss: List[Union[str,Callable]]=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)):
    """
        Build the U-net model based on pre-trained MobileNetV2 downsample layers and pix2pix upsample layers.
        Inputs:
            -inputs_shape: tuple - shape of the input image.
            -alpha: float - between 0 and 1. controls the width of the network. alpha < 1.0 proportionnaly decreases the number
            of filters, alpha > 1.0 increases the number of filters in each layer.
            -n_classes: integer - number of classe to classify
            -padding: str - either "same" or "valid" (case insensitive)
            -optimizer: str - String (name of optimizer) or optimizer instance. See tf.keras.optimizers.
            -metrics: list - contains metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance. See tf.keras.metrics.
        Ouputs:
            -model : Tensorflow model.
    """

    inputs_ = Input(shape=inputs_shape)
    down_stack = _downstack(inputs_shape=inputs_shape, alpha=alpha, n_classes=n_classes)
    up_stack = _upstack()

    x = inputs_
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    last = tf.keras.layers.Conv2DTranspose(
                                n_classes , 3, strides=2,
                                padding=padding,
                                activation='sigmoid')
    x = last(x)

    model = tf.keras.Model(inputs=inputs_, outputs=x)
    model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics)

    return model
