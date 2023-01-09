import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from typing import Optional, Union, Callable, List, Dict, Tuple
from os import listdir
from scipy.stats import median_absolute_deviation as mad
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import statistics as sts
from sklearn.preprocessing import MinMaxScaler
import pandas as ps
import os


def output_shape(img,
                 shapes: Tuple[int,...]):
         """
         Put the input image into the shape of [Nx,Ny,:].
         
         Parameters
         ----------
         
         img: array
          the image to process
         Nx: integer
          dimension along the X-axis
         Ny: integer
          dimension along the Y-axis
          
          
          Returns
          -------
          
          img : array
            the processed array of shape [:Nx,:Ny,:]
         """
         padded_shape = (*[((img.shape[i]-shapes[i])//2, ((img.shape[i]-shapes[i])//2) + ((img.shape[i]-shapes[i])%2)) for i in range(2)], (0,0))
         Nx =  padded_shape[0]
         Ny =  padded_shape[1]
         print(Nx, Ny)
         return img[:,Ny[0]:img.shape[1]-Ny[1],:]

def create_mask(pred_mask):
    """
    Create an mask array using the two labels image.
    
    Parameters
    ----------
    
    pred_mask: arrays
      n_classes arrays predicted by the model
      
      
    Returns
    -------
    
    pred_mask[0]: array
      array containing all the predicted label in the shape of the image
    """
    
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

def display(display_list):
    """
    Display the image, the true label and the predicted one.
    
    Parameters
    ----------
    
    display_list: list
      contains the true image, the label and the predicted mask to display.
    """
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list[0])):
        fig, ax = plt.subplots(1, len(display_list))
        ax[0].imshow((display_list[0])[i,:,:,0], aspect='auto', origin='lower')
        ax[0].set_title(title[0])
        ax[1].imshow((display_list[1])[i,:,:,0], aspect='auto', origin='lower')
        ax[1].set_title(title[1])
        ax[2].imshow((display_list[2])[i,:,:,0], aspect='auto', origin='lower')
        ax[2].set_title(title[2])
        ax[1].set_xlabel('subints')
        ax[0].set_ylabel('frequency channels')


    plt.show()

def show_predictions(model: Model,
                     dataset: tf.data.Dataset,
                     num: int = 1):
    """
    Show the predicted mask for a dataset.
    
    Parameters
    ----------
    
    model: Tensorflow model
      trained model to make the prediction
    dataset: Tensorflow dataset
      dataset containing the image to predict mask
    num: integer, optional
      number of desired prediction from the dataset
    """
    i = 0
    for image, mask in dataset.take(1):
        pred_mask = model.predict(image)
        if num < len(image):
            display([image[:num], mask[:num], create_mask(pred_mask[:num])])
        else:
            display([image, mask, create_mask(pred_mask)])

def to_rgb(img: np.array):
    """
    Converts the given array into a RGB image and normalizes the values to [0, 1).
    If the number of channels is less than 3, the array is tiled such that it has 3 channels.
    If the number of channels is greater than 3, only the first 3 channels are used

    Parameters
    ----------
    
    param img : numpy array
      the array to convert [bs, nx, ny, channels]

    Returns
    -------
    img: array
      the rgb image [bs, nx, ny, 3]
    """
    img = img.astype(np.float32)
    img = np.atleast_3d(img)

    channels = img.shape[-1]
    if channels == 1:
        img = np.tile(img, 3)

    elif channels == 2:
        img = np.concatenate((img, img[..., :1]), axis=-1)

    elif channels > 3:
        img = img[..., :3]

    img[np.isnan(img)] = 0
    img -= np.amin(img)
    if np.amax(img) != 0:
        img /= np.amax(img)
    return img


def predict_from_files(filelist, model):
    """
    Predict masks from a list of input images.
    
    Parameters
    ----------
            
    filelist:   list - contains the list of image to use for prediction
    model:      keras model - model which predict the masks
    
    Returns
    -------
    
    prediction: list
      contains all the prediction with the correct shape
    """
    prediction = []
    tmp = []
    for img_ in filelist:
        if os.path.isfile(img_):
            img = np.load(img_)
            if len(img.shape) <3 :
                prediction.append([])
                print("error in img shape %s\n" % img_)
                continue
            nsub = img.shape[1]
            nchan = img.shape[0]
            tmp = np.zeros((nchan,nsub,3))       
            tmp[:,:,0] = np.ma.log10(np.median(img, axis=2))
            tmp[:,:,1] = np.ma.log10(mad(img, axis=2))
            tmp[:,:,2] = np.ma.log10(np.ptp(img, axis=2))
            tmp = _input_shape(tmp)
      
            tmp =  preprocess_input(tmp)
            tmp = np.expand_dims(tmp, axis=0)

        pred = model.predict(tmp)
        prediction.append(create_mask(pred[0,:nchan,:nsub])[:,:,0])
    return prediction


def predict_from_dir(dir_, model, new_net=False):
    """
    Predict masks for file in the given directory
    
    Parameters
    ----------
     
    dir_: str
      the path to the directory
    model: keras model
      model which predict the masks
    
    Returns
    -------
    
    prediction: list
      contains all the prediction with the correct shape
    """
    tmp_img = []
    for img in listdir(dir_):
        tmp_img.append(dir_+img)
    return predict_from_files(tmp_img,model,new_net)



def _input_shape(img):
    """
    Reshape the arrays to make them divisible by 32. This shape is required by
    the UNET architecture.
    
    Parameters
    ----------
    
    img : array
      the image or label array to be reshape.
    
    Returns
    -------
    
    new_img : array
      the reshaped array.
    """
    #this value is required
    value = 32
    height = img.shape[0]
    width = img.shape[1]
    if height % value:
        new_h = height + value - (height%value)
    else:
        new_h = height
    if width % value :
        new_w = width + value - (width%value)
    else :
        new_w = width
    if len(img.shape)<3:
        new_img = np.zeros((new_h, new_w))
        new_img[:height,:width] = img
    else :
        new_img = np.zeros((new_h,new_w,img.shape[2]))
        new_img[:height, :width, :] = img
    return new_img

def _normalize(filename, pftsub):
        """
        Normalize the image sub integration time.
            Inputs:
                - filename : str - path to the array
                - pftsub : pandas datafram - contains tsub value for all obs.
            Output:
                - img : array - float normalized numpy array
        """
        arr = np.load(filename)
        name = filename.split('/')[-1]
        name = name.split('.npy')[0]
        #print(name)
        tsub = pftsub.loc[pftsub['filename'] == name+'.ar']['tsub'].values
        if tsub.size != 0:
            img = arr / float(tsub)
        #input_image = tf.cast(input_image, tf.float64) / 255.0
        return img


