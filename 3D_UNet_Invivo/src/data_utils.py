"""
Classes for loading and visualising MRA scans with their corresponding
segmentation file for all patients in a dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from copy import deepcopy

#import re
import pickle
import logging
#import os

import skimage
import pydicom
#import nrrd
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.utils import to_categorical

import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from IPython.display import display, clear_output

from pydicom.pixel_data_handlers import gdcm_handler, pillow_handler

logger = logging.getLogger('tensorflow')


class Dataset(object):
    """
    Object to collect all patients within the train/test set.
    """
    def __init__(self, scan_files, seg_files):
        """
        Class initialiser.

        Args:
            scan_files (list <str>): List of paths to MRI scan files.
            seg_files (list <str>): List of paths to segmentation files.
        """
        self.scan_files = scan_files
        self.seg_files = seg_files
        self.patients = self.build_dataset()
        self.patient_ids = list(self.patients.keys())

    def build_dataset(self):
        """
        Adds all patients that can be found in the provided path folders.
        Reads in all the MRI scans (multiple per patient) and their
        corresponding segmentation files (multiple per patient). Finally, it orders
        the scans and segmentation files of each patient.
        
        # Dicom in python:https://pydicom.github.io/pydicom/0.9/pydicom_user_guide.html

        Returns:
            dict: a dict of :class:`src.data_utils.Patient` objects.
        """

        patients = dict()
        
        # read in all scans as a list - one file for each scan
        # ds = pydicom.dcmread(s.pixel_array) --> to get array from dicom
        scans = []
        
        for s in self.scan_files:
            ds = pydicom.dcmread(s)
            ds.pixel_data_handler = gdcm_handler
            #print('scan_files: '+str(ds))
            #print('-----------------------------------------------------------')
            scans.append(ds)
            
        
        
        # read in all the segmentation files as a list - one file for each seg
        # ds = nrrd.read(s)[0] to get array of from nrrd file
        segs = []
        
        for t in self.seg_files:
            dt = pydicom.dcmread(t)
            #print('seg_files: '+str(ds))
            segs.append(dt)
        
        
        for i, (scan, seg) in enumerate(zip(scans, segs)):
            for j in range(len(scan.pixel_array)):
                if scan.PatientSize and scan.PatientWeight not in patients: 
                    
                    patients[scan.PatientWeight] = Patient(scan=np.array(scan.pixel_array[j]), seg=seg.pixel_array, instance_nums=int(scan.InstanceNumber), instance_nums_seg=int(seg.InstanceNumber))
                else:
                    patients[scan.PatientWeight].add_scan(scan=np.array(scan.pixel_array[j]), instance_nums=int(scan.InstanceNumber))
                    patients[scan.PatientWeight].add_seg(seg=seg.pixel_array,instance_nums_seg=int(seg.InstanceNumber))
            logger.info('Reading in and parsing scan %d/%d' % (i, len(scans)))
        
        
#        # sort scans within each patient
#        for patient in patients.values():
#            patient.order_scans()
#            
#        
#        # TODO
#        # sort segmentation files within each patient
#        for patient in patients.values():
#            patient.order_segs()
#        # patient.order_segs()
#        # create a function o<rder_segs in class Patient
        
        return patients
       
        

    def preprocess_dataset(self, resize=True, width=128, height=128,
                           max_scans=32):
        """
        Scales each scan in the dataset to be between zero and one, then
        resizes all scans and targets to have the same width and instance_nums_segheight.

        It also turns the target into a one-hot encoded tensor of shape:
        [depth, width, height, num_classes].

        Finally it ensures that all patients have at maximum `max_scans` scans.
        Patients with fewer scans will be padded with zeros. Extra scans of
        patients who have more (around 5% of the dataset) will be discarded.
        This is to ensure that a 3D U-Net with depth 4 can be built, i.e. the
        downsampled layer of the maxpooling and the upsampled layer of the
        transponsed convolution are guaranteed to have the same depth and
        shortcut connections can be made between them.

        Args:
            resize (bool): Whether to resize or not the scans.
            width (int): Width to resize all scans and targets in dataset.
            height (int): Height to resize all scans and targets in dataset.
            max_scans (tuple <int>): Maximum number of scans to keep.
        """
        for i, patient in enumerate(self.patients.values()):
            patient.normalise_scans()
            #patient.normalise_segs()
            patient.resize_reshape(resize=resize, width=width, height=height)
            patient.adjust_depth(max_scans=max_scans)
            patient.preprocessed = True

            logger.info('Preprocessing data of patient %d/%d' % (
                i, len(self.patients.values())))

    def save_dataset(self, path):
        """
        Saves the dataset as a pickled object.

        Args:
            path (str): Full path and object name used for saving the dataset.
        """
        pickle.dump(self, open(path, "wb"))

    @staticmethod
    def load_dataset(path):
        """
        Loads the dataset from a pickled file.

        Args:
            path (str): Full path to the pickled dataset.

        Returns:
            dict: a dict of :class:`src.data_utils.Patient` objects.
        """
        return pickle.load(open(path, 'rb'))

    def create_tf_dataset(self, resized=True, num_classes=3):
        """
        Creates a TensorFlow DataSet from the DataSet. Note, this has to
        be run after all the MRI scans have been rescaled to the same size.
        They can have different depths, i.e. number of scans however.

        Args:
            resized (bool): Whether dealing with a uniformly resized scans.
            num_classes (int): Number of classes in segmentation files.

        Returns:seg
            :class:`tf.data.Dataset`: TensorFlow dataset.
        """

        # extract all scans and segmentation images from every patient
        scans_segs = [(p.scans, p.segs) for p in self.patients.values()]
        if resized:
            _, width, height = scans_segs[0][0].shape
            #print(width,height)
            x_shape = [None, width, height, 1]
            y_shape = [None, width, height, num_classes]
        else:
            x_shape = [None, None, None, 1]
            y_shape = [None, None, None, num_classes]
        
        
        # define an iterator which gives us ability to iterate through the dataset and to retrieve the real values of the data
        def gen_scans_segs():
            """seg
            Generator function for dataset creation.
            An iterator which gives us ability to iterate through the dataset and to retrieve the real values of the data
            """
            for s in scans_segs:
                # if each image is different sized, we need to get w, h here
                _, width, height = s[0].shape

                # add channel dimension to scans
                x = s[0].reshape((-1, width, height, 1))
                yield (x, s[1])

        return tf.data.Dataset.from_generator(
            generator=gen_scans_segs,
            output_types=(tf.float32, tf.int32),
            output_shapes=(x_shape, y_shape)
        )


class Patient(object):
    """
    Basic object to store all slices of a patient in the study.
    """
    def __init__(self, scan, seg, instance_nums, instance_nums_seg):
        """
        Class initialiser.

        Args:
            scan (:class:`pydicom.dataset.FileDataset'): A loaded MRI scan.
            seg (:class:`numpy.ndarray`): A loaded segmentation file.
        """
        
        #
#        print('scan values class patient: ' +sscanstr(np.unique(scan)))
#        print('seg values class patien: ' +str(np.unique(seg)))
        
        # store all scans of a patient in the study
        self.scans = list()       
        self._instance_nums = list()
        self.thicknesses = set()
        self.manufacturers = set()
        self.add_scan(scan, instance_nums)
        self.preprocessed = True
        
            
        # store all segmentation files of a patient in the study  
        self.segs = list()
        self._instance_nums_seg = list()
        self.thicknesses_seg = set()
        self.manufacturers_seg = set()
        self.add_seg(seg, instance_nums_seg)
        self.preprocessed = True
        
        # TODO
        # add up segmentation files in dictionary
        #self.add_seg(seg)

              
        # TODO
        # how to store all segmentation files of a patient
        # define functions to: add_seg, order_segs, normalise_segs, resize_reshape_scans_and_segs
        # 
        # adding key to dict: https://stackoverflow.com/questions/1024847/add-new-keys-to-a-dictionary 
        # iterate through dict: https://realpython.com/iterate-through-dictionary-python/
        # https://www.saltycrane.com/blog/2007/09/how-to-sort-python-dictionary-by-keys/
        
        instance_nums_seg
      
        # TODO
        # add up segmentation file in list
    def add_seg(self, seg, instance_nums_seg):
        """
        Adds single segmentation file to a patient's list of segmentation files
        
        Args:
            seg (:class:`nrrd.dataset.FileDataset'): A loaded segmentation files.
        
        """
        #
        #print('seg values add_seg: ' +str(np.unique(seg)))
        #
        #Instance number is an image ID or key use to order segs
        self._instance_nums_seg.append(instance_nums_seg)
        
        
        
        # make depth the first dim as with the scans
#        seg = np.moveaxis(seg, -1, 0) ----not used in this solution
        #print('seg values after moving axes add_seg: ' +str(np.unique(seg)))
        
# #                     /////////////////////
#        z = 0
#        o = 0
#        t = 0
#        for j in seg.flatten():
#            if(j == 0):
#                z+=1
#            if(j == 1):
#                o+=1
#            if(j == 2):
#                t+=1
#        print('add_seg: ' + 'zero: ' + str(z) + ' one: ' + str(o) + ' two: ' + str(t))
##                     /////////////////////
        
        self.segs.append(seg)
       
        #TODO
        # find way to add up segs in dictionay
        #segs = self.segs
        #self.segs = segs.update(seg)
        
        
        
    def add_scan(self, scan, instance_nums):
        """
        Adds one more scan to a patient's list of scans. It also saves it's
        manufacturer, slice thickness and instance number, i.e. the index of
        the scan in the sequence.

        Args:
            scan (:class:`pydicom.dataset.FileDataset'): A loaded MRI scan.
        """
        self.scans.append(np.array(scan))
        self._instance_nums.append(int(instance_nums))  
        #self.thicknesses.add(int(scan.SliceThickness))
        #self.manufacturers.add(scan.Manufacturer)
        
    def order_scans(self):
        """
        Orders the scans of a patient according to the imaging sequence.
        """
        order = np.argsort(self._instance_nums)
        self.scans = np.array(self.scans)
        self.scans = self.scans[order, :, :]
        
        
        
        # TODO: order list of segmentation files
    def order_segs(self):
        """
        Orders the segmentation files of a patients according to the image sequence.
        """
        order = np.argsort(self._instance_nums_seg)
        self.segs = np.array(self.segs)
        self.segs = self.segs[order, :, :]
        
#         #                     /////////////////////
#        for i, seg in enumerate(self.segs):
#            z = 0
#            o = 0
#            t = 0
#            for j in seg.flatten():
#                if(j == 0):
#                    z+=1
#                if(j == 1):
#                    o+=1
#                if(j == 2):
#                    t+=1
#            print('order_segs: ' + 'zero: ' + str(z) + ' one: ' + str(o) + ' two: ' + str(t))
##                     /////////////////////
        
        # TODO: order dictionary of segmentation files
#    def order_segs(self):
#        """
#        Orders the segmentation files of a patients according to the image sequence.
#        """
        

    def normalise_scans(self):
        """
        Scales each scan in the dataset to be between zero and one.
        """
        scanList = []
        for s, scan in enumerate(self.scans):
            scan = scan.astype(np.float32)
            scanList.append(scan)
        
        self.scans = np.array(scanList)
        
        scanList2 = []
        for s, scan in enumerate(self.scans):
            scan_min = np.min(scan)
            scan_max = np.max(scan)

            # avoid dividing by zero
            if scan_max != scan_min:
                scan = (scan - scan_min) / (scan_max - scan_min)
            else:
                scan = 0
            
            scanList2.append(scan)
        self.scans = np.array(scanList2)
    
    def normalise_segs(self):
        """
        Scales each seg in the dataset to be between zero and one.
        """
        segs = self.segs.astype(np.float32)

        for i, seg in enumerate(segs):
            seg_min = np.min(seg)
            seg_max = np.max(seg)

            # avoid dividing by zero
            if seg_max != seg_min:
                segs[i] = (seg - seg_min) / (seg_max - seg_min)
            else:
                segs[i] = seg * 0
        self.segs = segs
    
    def fix_labels_segs(self, segs, labels):
        """       
        Keep only a given sets of indexes classes from each image in list of seg images
        """
        encoded = []
        #numLabels = len(labels)
        for img in segs:
            img2 = np.zeros_like(img)
            for i,label in enumerate(labels):
                img2 = np.where(img==label, i+1, img2)
            encoded.append(img2)
        return encoded
    
    
    def one_hot_encoding_segs(self,segs, numLabels):
        """
        One hot encode each images in a list of segmentation images
        """
        encoded = []
        for img in segs:
            original_shape = img.shape
            img2 = to_categorical(img.flatten(), numLabels)
            img2 = np.reshape(img2, (original_shape[0], original_shape[1], numLabels))
            encoded.append(img2)
        return encoded
        
        
    def resize_reshape(self, resize, width, height):
        """
        Resizes each scan and target segmentation image of a patient to a
        given width and height. It also turns the target into a one-hot
        encoded tensor of shape: [depth, width, height, num_classes].

        Args:
            resize (bool): Whether to resize or not the scans.
            width (int): Width to resize all scans and targets in dataset.
            height (int): Height to resize all scans and targets in dataset.
            
        """
        # resize scans
        if resize:
            depth = len(self.scans)
            
            ScanList = []
            for i in range(len(self.scans)):
                scan = skimage.transform.resize(
                    image=self.scans[i],
                    output_shape=(width, height)
                )
                ScanList.append(scan)
            self.scans = np.array(ScanList)
        
#        # resize scans
#        if resize:
#            depth = self.scans.shape[0]
#                
#            scans = skimage.transform.resize(
#                image=self.scans,
#                output_shape=(depth, width, height)
#            )
#            self.scans = scans


        ####### one hot encoding: Working!!!
        # https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy/36960495#36960495
        # take a seg image: 
        #                   - compute the number of columns of the target
        #                   - one hot encoding the target image
        #                   - format .cso.dcm
        #                   -
        #                   - NB: resize function is working well for any given input
                        
        if resize:
            # https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy/36960495#36960495
            # https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
            # inputs
            depth = self.scans.shape[0]
            ncols = 3
            seg_preprocess = []
            
            for i, seg in enumerate(self.segs):
                
                
                # transform [width, height] into [width, height, number_classes]
                labels_one_hot = (seg.ravel()[np.newaxis] == np.arange(ncols)[:, np.newaxis]).T
                labels_one_hot.shape = seg.shape + (ncols, )
                
                # resize targets while preserving their boolean nature: [width, height, ncols]
                seg_resize_reshape = skimage.img_as_bool(
                                skimage.transform.resize(
                                    image=labels_one_hot,
                                    output_shape=(width, height, ncols)
                                ))
    
                seg_preprocess.append(seg_resize_reshape.astype(int))
            self.segs = np.array(seg_preprocess)
            

            
#        ####### one hot encoding appraoch based on Lilli advise
#        # https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy/36960495#36960495
#        # take a seg image: 
#        #                   - compute the number of columns of the target
#        #                   - one hot encoding the target image
#        #                   - format .cso.dcm
#        #                   -
#        #                   - NB: resize function is not working well for any given image input except the 512*512
#        if resize:
#             depth  = self.scans.shape[0]
#             numLabels  = 3
#             labels = [1,14]
#             segs = np.array(self.segs)
#             
#             #
#             fixLabels = []
#             oneHot = []
#             
#             # fix labels of target images
#             for img in segs:
#                 img2 = np.zeros_like(img)
#                 for i,label in enumerate(labels):
#                     
#                     #print('i: ' + str(i) + ' label: ' + str(label))
#                     img2 = np.where(img==label, i+1, img2)
#
#                 fixLabels.append(img2)
#             
#                 
#             # one hot encoding target images
#             for img in fixLabels:
#                 
#                 #img2 = np.reshape(img, (width, height))
#                 original_shape = img.shape
#                 print('original_shape: ' + str(original_shape))
#                 img2 = to_categorical(img.flatten(), numLabels)
#                 
#                 # resize encoded image
#                 img2 = np.reshape(img2, (width, height, numLabels))
#                 oneHot.append(img2)
#                 
#             self.segs = np.array(oneHot)
#                         
#             
                     
             

#        ####### one hot encoding: approach0
#        # https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy/36960495#36960495
#        # take a seg image: 
#        #                   - compute the number of columns of the target
#        #                   - one hot encoding the target image
#        #                   - format .cso.dcm
#        #                   -
#        #                   - NB: resize function is not working well               
#                 
#        if resize:
#            # https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy/36960495#36960495
#            # https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
#            # inputs
#            depth = self.scans.shape[0]
#            ncols = 3
#            seg_preprocess = []
#            
#            for i, seg in enumerate(self.segs):
#                               
#                # One hot encoding 
#                # transform [width, height] into [width, height, number_classes]
#                # working approach as well
#                labels_one_hot = (np.arange(ncols) == seg[...,None]).astype(int)
#                labels_one_hot.shape = seg.shape + (ncols, )
#                
#                # resize targets while preserving their boolean nature: [width, height, ncols]
#                seg_resize_reshape = skimage.img_as_bool(
#                                skimage.transform.resize(
#                                    image=labels_one_hot,
#                                    output_shape=(width, height, ncols)
#                                ))
#    
#                seg_preprocess.append(seg_resize_reshape.astype(int))
#            self.segs = np.array(seg_preprocess)
#              
#



#        ####### One hot encoding: approach 4
#        # https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy/36960495#36960495
#        # take a seg image: 
#        #                   - compute the number of columns of the target
#        #                   - one hot encoding the target image
#        #                   - format .nrrd
#        #                   -
#        #                   - NB: working for a single .nrrd file
#        if resize:
#            
#            depth = self.scans.shape[0]
#            seg = deepcopy(self.seg)
#            
#            # One hot encoding 
#            # transform [width, height] into [width, height, number_classes]
#            ncols = seg.max() + 1
#            labels_one_hot = (seg.ravel()[np.newaxis] == np.arange(ncols)[:, np.newaxis]).T
#            labels_one_hot.shape = seg.shape + (ncols, )
#            
#            # resize targets while preserving their boolean nature: [width, height, ncols]
#            seg_resize_reshape = skimage.img_as_bool(
#                            skimage.transform.resize(
#                                image=labels_one_hot,
#                                output_shape=(width, height, ncols)
#                            ))
#            seg_preprocess = seg_resize_reshape.astype(int)
#           
#            # resize and reshape target image to: [depth, width, height, ncols]
#            list_img = []
#            for i in range(depth):
#                list_img.append(seg_preprocess)
#            
#            
#            self.seg = np.array(list_img)  
            
        
        
    def adjust_depth(self, max_scans):
        """
        There's a wide range of scan numbers across the patients. We need to
        unify these so they can be fed into the network. We discard extra
        scans (i.e. more than the `max_scans`) and patients with less will be
        padded with zeros by TensorFlow.

        This is to ensure that a 3D U-Net with depth 4 can be built, i.e. the
        downsampled layer of the maxpooling and the upsampled layer of the
        transponsed convolution are guaranteed to have the same depth.

        Note this function can only be run once the target tensor has been
        converted to a one hot encoded version with `resize_and_reshape`.

        Args:
            max_scans (tuple <int>): Maximum number of scans to keep.
        """
        self.scans = self.scans[:max_scans]
        self.segs = self.segs[:max_scans]

    def patient_tile_scans(self):
        """
        Generates a tiled image, visualising all the scans of a patient.
        """
        Patient.tile_scans(self.scans, self.segs, self.preprocessed)
    
   
    def patient_tile_scans2(self):
        """
        Generates a tiled image, visualising all the scans of a patient.
        """
        Patient.tile_scans2(self.scans, self.segs, self.preprocessed)
    
    
    def patient_anim_scans(self):
        """
        Generates an animation for IPython, visualising all the scans of a
        patient.
        """
        Patient.anim_scans(self.scans, self.segs, self.preprocessed)
        
    
    def patient_anim_scans2(self):
        """
        Generates an animation for IPython, visualising all the scans of a
        patient.
        """
        Patient.anim_scans2(self.scans, self.segs, self.preprocessed)
        
    
    def patient_plot_scans_segs(self):
        Patient.plot_data([self.scans,self.segs[:,:,:,2]],['gray','gnuplot'])    # with resizing
#        Patient.plot_data([self.scans,self.segs],['gray','gnuplot'])              # without resizing
        #Patient.plot_data([self.scans,np.array(self.segs)],['gray','gnuplot'])
        #Patient.plot_data([self.scans,np.array(self.segs)[:,:,:,2]*100000000],['gray','gnuplot'])
        #Patient.plot_data([self.scans],['gray'])
        #Patient.plot_data([self.segs],['gray'])
        #Patient.plot_data([self.segs[:,:,:,1]],['gray'])
        # plot_data([np.array(ohlabels)[:,:,:,2]],['gnuplot'])

    
    @staticmethod
    def plot_data(data,cmaps):
        data_shape = np.array(data).shape
        numImageTypes = data_shape[0]
        numImageSets = data_shape[1]
        print(numImageTypes,numImageSets)
        fig = plt.figure(figsize=(3*numImageTypes,3*numImageSets))
        for i in range(numImageSets):
            for j in range(numImageTypes):
                plt.subplot(numImageSets, numImageTypes,i*numImageTypes + j+1)
                plt.axis('off')
                img = plt.imshow(data[j][i])
                img.set_cmap(cmaps[j])
        plt.show()
        
  
    @staticmethod
    def anim_scans(scans, segs, preprocessed):
        """
        Generates an animation for IPython, visualising all the scans of a
        patient.

        Args:
            scans (:class:`numpy.array`): MRI image with shape
                [depth, width, height]
            seg (:class:`numpy.array`): MRI image segmentation with shape
                [depth, width, height]
            preprocessed (bool): Whether the scans been preprocessed.
        """
#        print('--------------------------------------------------------------')
#        print('anim_scans_scans_shape: ' +str(scans.shape))
#        print('anim_scan_seg_shape: ' +str(segs.shape))
#        print('--------------------------------------------------------------')
        
        #anim_scans_scans_shape: (8, 32, 32)
        #anim_scan_seg_shape: (8, 32, 32, 15)
        
        fig, ax = plt.subplots()
        for i, scan in enumerate( scans):
           img = Patient.concat_scan_seg(scans, segs[:,:,:,2], i, preprocessed=True)   # with resizing
#           img = Patient.concat_scan_seg(scans, segs, i, preprocessed=True)             # without resizing
           plt.imshow(img, cmap='gray')
           clear_output(wait=True)
           display(fig)
        plt.axis('off')
        plt.show()
        

    @staticmethod
    def tile_scans(scans, segs, preprocessed):
        """
        Generates a tiled image, visualising all the scans of a patient.

        Args:
            scans (:class:`numpy.array`): MRI image with shape
                [depth, width, height]
            seg (:class:`numpy.array`): MRI image segmentation with shape
                [depth, width, height]
            preprocessed (bool): Whether the scans been preprocessed.
        """
#        print('--------------------------------------------------------------')
#        print('tile_scans_scans_shape: ' +str(scans.shape))
#        print('tile_scans_seg_shape: ' +str(segs.shape))
      
        #tile_scans_scans_shape: (8, 32, 32)
        #tile_scans_seg_shape: (8, 32, 32, 15)        
        
        n_scans = scans.shape[0]
        cols = int(np.ceil(np.power(n_scans, 1/3)))
        rows = cols * 2
        if cols * rows < n_scans:
            rows += 1
        fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
            
        for i in range(cols * rows):
            row_ind = int(i / cols)
            col_ind = int(i % cols)
            if i < n_scans:
            
                img = Patient.concat_scan_seg(scans, segs[:,:,:,2], i, preprocessed=True)         # with resizing
#                img = Patient.concat_scan_seg(scans, segs, i, preprocessed=True)                 # without resizing
                ax[row_ind, col_ind].set_title('slice %d' % (i + 1))
                ax[row_ind, col_ind].imshow(img, cmap=plt.cm.bone)
                ax[row_ind, col_ind].axis('off')
        plt.show()
        
    
    @staticmethod
    def concat_scan_seg(scans, segs, i, preprocessed):
        """
        Helper function, that concatenates the MRI image with its corresponding
        segmentation and rescales the latter so their colours are comparable.

        If we have preprocessed data, i.e. the target is a one hot encoded
        tensor, we use the 2nd class for visualisation.

        Args:
            scans (:class:`numpy.array`): MRI image with shape
                [depth, width, height]
            seg (:class:`numpy.array`): MRI image segmentation with shape
                [depth, width, height]
            i (int): Index of scan in the patient's list of scans.
            preprocessed (bool): Whether the scans been preprocessed.
        """
#        print('--------------------------------------------------------------')
#        print('concat_scan_seg:_scans_shape: ' +str(scans.shape))
#        print('concat_scan_seg:_seg_shape: ' +str(segs.shape))
#        print('--------------------------------------------------------------')
       
        #concat_scan_seg:_scans_shape: (8, 32, 32)
        #concat_scan_seg:_seg_shape: (8, 32, 32)
        
        scan = scans[i]
        seg = segs[i]
        # if we have preprocessed data use class 2 from target/segmentation
        # class indexes: 0, 1, 2, ..., 14
        # 2nd class is : 1
        if preprocessed:
            seg = seg * scan.max()
        else:
            seg = seg * (scan.max() / 2)
        return np.hstack([scan, seg])
        
    
        
        
        
        
############################////////////////// Prediction#####################################################  
    
    @staticmethod
    def anim_scans2(scans, segs, preprocessed):
        """
        Generates an animation for IPython, visualising all the scans of a
        patient.

        Args:
            scans (:class:`numpy.array`): MRI image with shape
                [depth, width, height]
            seg (:class:`numpy.array`): MRI image segmentation with shape
                [depth, width, height]
            preprocessed (bool): Whether the scans been preprocessed.
        """
#        print('--------------------------------------------------------------')
#        print('anim_scans_scans2_shape: ' +str(scans.shape))
#        print('anim_scan2_seg_shape: ' +str(segs.shape))
#        print('--------------------------------------------------------------')
        
        #anim_scans_scans_shape: (8, 32, 32)
        #anim_scan_seg_shape: (8, 32, 32, 15)
        
        fig, ax = plt.subplots()
        for i, scan in enumerate( scans):
#           img = Patient.concat_scan_seg2(scans, segs[:,:,:,2], i, preprocessed=True)   # with resizing
           img = Patient.concat_scan_seg2(scans, segs, i, preprocessed=True)             # without resizing
           plt.imshow(img, cmap='gray')
           clear_output(wait=True)
           display(fig)
        plt.axis('off')
        plt.show()  
            
    
    @staticmethod
    def tile_scans2(scans, segs, preprocessed):
        """
        Generates a tiled image, visualising all the scans of a patient.

        Args:
            scans (:class:`numpy.array`): MRI image with shape
                [depth, width, height]
            seg (:class:`numpy.array`): MRI image segmentation with shape
                [depth, width, height]
            preprocessed (bool): Whether the scans been preprocessed.
        """
#        print('--------------------------------------------------------------')
#        print('tile_scans_scans2_shape: ' +str(scans.shape))
#        print('tile_scans_seg2_shape: ' +str(segs.shape))
      
        #tile_scans_scans_shape: (8, 32, 32)
        #tile_scans_seg_shape: (8, 32, 32, 15)        
        
        n_scans = scans.shape[0]
        cols = int(np.ceil(np.power(n_scans, 1/3)))
        rows = cols * 2
        if cols * rows < n_scans:
            rows += 1
        fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
            
        for i in range(cols * rows):
            row_ind = int(i / cols)
            col_ind = int(i % cols)
            if i < n_scans:
                #print('segs[:,:,:,2] shape: ' + str(segs.shape))
                #print('segs[:,:,:,2]: ' + str(segs[:,:,:,2]))
                img = Patient.concat_scan_seg2(scans, segs, i, preprocessed=True)         # with resizing
#                img = Patient.concat_scan_seg(scans, segs, i, preprocessed=True)                 # without resizing
                ax[row_ind, col_ind].set_title('slice %d' % (i + 1))
                ax[row_ind, col_ind].imshow(img, cmap=plt.cm.bone)
                ax[row_ind, col_ind].axis('off')
        plt.show()
    
    
    @staticmethod
    def concat_scan_seg2(scans, segs, i, preprocessed):
        """
        Helper function, that concatenates the MRI image with its corresponding
        segmentation and rescales the latter so their colours are comparable.

        If we have preprocessed data, i.e. the target is a one hot encoded
        tensor, we use the 2nd class for visualisation.

        Args:
            scans (:class:`numpy.array`): MRI image with shape
                [depth, width, height]
            seg (:class:`numpy.array`): MRI image segmentation with shape
                [depth, width, height]
            i (int): Index of scan in the patient's list of scans.
            preprocessed (bool): Whether the scans been preprocessed.
        """
#        print('--------------------------------------------------------------')
#        print('concat_scan_seg:_scans_shape: ' +str(scans.shape))
#        print('concat_scan_seg:_seg_shape: ' +str(segs.shape))
#        print('--------------------------------------------------------------')
       
        #concat_scan_seg:_scans_shape: (8, 32, 32)
        #concat_scan_seg:_seg_shape: (8, 32, 32)
        
        scan = scans
        seg = segs
        #print('length seg: ' +str(len(segs)))
        # if we have preprocessed data use class 2 from target/segmentation
        # class indexes: 0, 1, 2, ..., 14
        # 2nd class is : 1
        if preprocessed:
            seg = seg * scan.max()
        else:
            seg = seg * (scan.max() / 2)
        return np.hstack([scan, seg])
