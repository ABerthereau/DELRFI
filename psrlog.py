import numpy as np
from os import path, listdir
from utils import to_rgb
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input
from scipy.stats import median_absolute_deviation as mad
from pathlib import PurePosixPath
import pandas as ps
import statistics as sts
from sklearn.preprocessing import MinMaxScaler


class ImageGenerator():
    def __init__(self, imgs, masks,
                 batch_size=2,
                 shuffle=False,
                 max_dimension=None,
                 batch_nb=None,
                 max_tsub=False,
                 check_file=True):
        """
            Initialize the generator with the directory where to find masks
            and spectrograms.
            
            Parameters
            ----------
            
                -imgs : List or Str
                    Either a list of string containing files to use as images or the path to the full directory
                -masks : List or Str
                    Same as the imgs, but for the mask labels
                -batch_size : int, optional
                    number of example per batch.
                -shuffle : Boolean, optional
                    to shuffle the batch at each epoch.
                -max_dimension: int or tuple, optional
                    the maximal dimension allowed to your arrays. If None, max_dimension take the biggest size inside the batch.
        """


        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_dimension = max_dimension
        self.image_paths = []
        self.input_shapes = []
        self.check_file = check_file

        if (isinstance(imgs, list) or isinstance(imgs, np.ndarray)) and \
           (isinstance(masks, list) or isinstance(masks, np.ndarray)):
            self._load_from_list(imgs, masks)
        elif isinstance(imgs, str) and isinstance(masks, str):
            self._load_from_dir(imgs, masks)
        else:
            raise(TypeError("Error : imgs and masks are not list nor directories"))
            
        self.idx = np.arange(len(self.image_paths)).astype(int)
        self.batch_nb = batch_nb
        self.max_tsub = max_tsub
        print("Initialization of the generator")

    def _prediction_preprocess(self,directory):
         for _,class_dir in enumerate(listdir(directory)):
            if ".npy" in class_dir:
                self.image_paths += [path.join(directory,class_dir)]
                self.input_shapes.append((np.load(directory+class_dir)).shape)


    def _training_preprocess(self,directory):
        tmp_img = []
        tmp_labels = []
        self.label_paths = []

        for _,class_dir in enumerate(listdir(directory+'img/')):
            self.image_paths += [path.join(directory+'img/',class_dir)]
            tmp_img.append(class_dir.split('.npy')[0])
        for _,class_dir in enumerate(listdir(directory+'masks/')):
            self.label_paths += [path.join(directory+'masks/',class_dir)]
            tmp_labels.append(class_dir.split('_cgmask.npy')[0])

        if self.check_file:
            self.check_files(tmp_labels, tmp_img)
        else:
            self.image_paths = np.array(self.image_paths)
            self.label_paths = np.array(self.label_paths)

        self.label_paths.sort()
        self.image_paths.sort()
        if self.check_file:
            f = open('./filerrors.list', 'ab')
            self.check_size()
            np.savetxt(f, self.filerrors, delimiter=';', fmt='%s')

    def __len__(self):
        """
        Return the number of batches.
        """
        if self.batch_nb:
            return self.batch_nb
        return int(np.ceil(len(self.image_paths)/float(self.batch_size)))

    def check_files(self, labels, imgs):
        idx_lab = []
        idx_img = []
        self.filerrors = []
        for i in range(len(imgs)):
            if imgs[i] in labels:
                idx_img.append(i)
            else:
                (self.filerrors).append([self.image_paths[i], (1,0)])
        for i in range(len(labels)):
            if labels[i] in imgs:
                idx_lab.append(i)
            else:
                (self.filerrors).append([self.label_paths[i], (1,0)])
        self.image_paths = np.array(self.image_paths)[idx_img]
        self.label_paths = np.array(self.label_paths)[idx_lab]


    def check_size(self):
        """
            Check the size of the images and masks list to ensure they are the same. If there is any difference or missed img/mask, the filename is stored into an errorfile.
        """
        idx_to_save = []
        for i in range(len(self.image_paths)):
            img = np.load(self.image_paths[i])
            lab = np.load(self.label_paths[i])
            if img.shape[0] == lab.shape[0] and img.shape[1] == lab.shape[1]:
                idx_to_save.append(i)
                (self.input_shapes).append(img.shape)
            else:
                (self.filerrors).append([(self.label_paths[i]), (0,1)])
        self.image_paths = self.image_paths[idx_to_save]
        self.label_paths = self.label_paths[idx_to_save]


    def _load_image_(self,img_path):
        """
            Load and preprocess the image array saved into numpy format. Arrays should
            be in the format [nchan, nsub, 1]. The output array has shape [nchan, nsub, 3]
            to be used by the MobileNetV2 as a transfert learning method.
            
            
            Parameters
            ----------
                - img_path: string,
                    path to the numpy array.
            
            Returns
            -------
                - img : numpy array
                    new array with scaled values.
        """
        img = np.load(img_path)
        nchan = img.shape[0]
        nsub = img.shape[1]
        max_dim = max(img.shape)
        #med, mad, p2p
        #include the generated rfi <3
        img_ = np.zeros((nchan,nsub,3))
        if img.shape[2] ==3:
        #since it's generated data it's already have the correct 3 channels input
            img_[:,:,0] = np.ma.log10(img[...,0])
            img_[:,:,1] = np.ma.log10(img[...,1])
            img_[:,:,2] = np.ma.log10(img[...,2])
        else:
            img_[:,:,0] = np.ma.log10(np.median(img, axis=2))
            img_[:,:,1] = np.ma.log10(mad(img, axis=2))
            img_[:,:,2] = np.ma.log10(np.ptp(img, axis=2))
        img = self._input_shape(img_)
        img =  preprocess_input(img)
        return img

    def _load_label_(self, label_path):
        """
            Load and preprocess the label array saved into numpy format. Arrays should
            be in the format [nchan, nsub]. The output array has shape [nchan, nsub, 1]
            to be used by the MobileNetV2 as a transfert learning method.
            
            Parameters
            ----------
            
                - label_path: string
                    path to the numpy array.
            
            Returns
            -------
            
                - label : numpy array
                    new array with integer values between 0 and n_classes.
        """
        img = np.load(label_path)
        img = self._input_shape(img)
        label = np.zeros((img.shape[0],img.shape[1],1))

        label[:,:,0] = img
        return label




    def _input_shape(self,img):
        """
            Reshape the arrays to make them divisible by 32. This shape is required by
            the UNET architecture.
            
            Parameters
            ----------
            
                - img : array 
                the image or label array to be reshape.
            
            Returns
            -------
            
                - new_img : array
                    the reshaped array.
        """
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



    def _pad_images(self, img, shape):
        """
            Pad the image with zeros into the input shape.
            
            Parameters
            ----------
                - img: array
                    numpy array of image or label to padding
                - shape: tuple
                    The wanted shape
        """
        if self.max_tsub :
            if shape[1]> self.max_tsub:
                start = np.random.randint(0, shape[1]-self.max_tsub)
                end = start + self.max_tsub
                img = img[:,start:end,:]
            else:
                shape = tuple((shape[1], self.max_tsub,1))
        try :
            img = np.pad(img,(*[((shape[i]-img.shape[i])//2, ((shape[i]-img.shape[i])//2) + ((shape[i]-img.shape[i])%2)) for i in range(2)], (0,0)), mode='constant', constant_values=0.)
        except ValueError:
            print(img.shape, shape)
        return img


    def _load_from_list(self, listimg, listlabel):
        tmp_img = listimg
        tmp_label = listlabel
        if self.check_file:
            self.image_paths = listimg
            self.label_paths = listlabel
            print(len(self.image_paths), len(self.label_paths))

            self.check_files(tmp_label, tmp_img)
            f = open('./filerrors2.list', 'ab')
            self.check_size()
            #save into logfile all the files that have not the same size or
            #does not have any counterpart.
            np.savetxt(f, self.filerrors, delimiter=';', fmt='%s')
            print(len(self.image_paths), len(self.label_paths))
        else:
            self.label_paths = np.array(tmp_label)
        self.image_paths = np.array(tmp_img)



    def _load_from_dir(self, dirimg, dirlab):
        tmp_img = []
        tmp_lab = []

        for img in listdir(dirimg):
            tmp_img.append(dirimg+img)
        for lab in listdir(dirlab):
            tmp_lab.append(dirlab+lab)
        self._load_from_list(tmp_img, listlabel=tmp_lab)


    def check_files(self, labels, imgs):
        """
            Check if the images and labels are matching.
            
            Parameters
            ----------
            
                - labels : list(str)
                    path of the labels 
                - imgs : list(str)
                    path of the images
        """
        idx_lab = []
        idx_img = []
        self.filerrors = []

        tmp_img = []
        tmp_lab = []
        for img in imgs:
            tmp_img.append((PurePosixPath(img).name).split(".npy")[0])
        for lab in labels:
            tmp_lab.append((PurePosixPath(lab).name).split("_cgmask.npy")[0])
        for i in range(len(tmp_img)):

            if tmp_img[i] in tmp_lab:
                idx_img.append(i)
            else:
                (self.filerrors).append([self.image_paths[i], (1,0)])
        for i in range(len(labels)):
            if tmp_lab[i] in tmp_img:
                idx_lab.append(i)
            else:
                (self.filerrors).append([self.label_paths[i], (1,0)])
        self.image_paths = np.array(self.image_paths)[idx_img]
        self.label_paths = np.array(self.label_paths)[idx_lab]

    

    def __call__(self):
        if self.shuffle:
            np.random.shuffle(self.idx)
        for batch in range(len(self)):

            batch_img_paths = self.image_paths[self.idx[batch*self.batch_size:(batch+1)*self.batch_size]]
            batch_images = [self._load_image_(image_path) for image_path in batch_img_paths]


            max_resolution = tuple((max(img.shape[i] for img in batch_images) for i in range(2)))
            batch_images = np.array([self._pad_images(image, max_resolution) for image in batch_images])

            batch_label_paths = self.label_paths[self.idx[batch*self.batch_size:(batch+1)*self.batch_size]]

            batch_labels_ = [self._load_label_(label_path) for label_path in batch_label_paths]
            batch_labels = np.array([self._pad_images(label, max_resolution) for label in batch_labels_])

            yield batch_images, batch_labels
