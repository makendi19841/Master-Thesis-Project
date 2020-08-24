"""
Data feeding function for train, eval and test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from src.data_utils import Dataset


def input_fn(training, params):
    """
    Simple input_fn for our 3D U-Net estimator, handling train and test data
    preparation.

    Args:
        training (bool): Whether we are training or testing.
        params (dict): Params for setting up the data. Expected keys are:
            max_scans (int): Maximum number of scans we see in any patient.
            train_img_size (int): Width and height of resized training images.
            batch_size (int): Number of of patient in each batch for training.
            num_classes (int): Number of mutually exclusive output classes.
            train_dataset_path (str): Path to pickled
                :class:`src.data_utils.Dataset` object.
            test_dataset_path (str): Path to pickled
                :class:`src.data_utils.Dataset` object.

    Returns:
        :class:`tf.dataset.Dataset`: An instantiated Dataset object.
    """
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    
    
    #--------------------------------------------------------------------------
    # Create A Training Dataset 
    #      - shuffle it
    #      - batch and pad each 3D scan to have an equal depth, width and height have already been set to 128 in preprocessing
    # -------------------------------------------------------------------------
     
    max_s = params['max_scans']
    w = h = params['train_img_size']
    
    if training==1:
        dataset = Dataset.load_dataset(
            os.path.join(package_root, params['train_dataset_path'])
        ).create_tf_dataset().shuffle(
            # we have 70 train examples, this will provide good shuffling
            buffer_size=70 
        ).repeat().padded_batch(
            batch_size=params['batch_size'],
            padded_shapes=(
                [max_s, w, h, 1], [max_s, w, h, params['num_classes']]
            )
        )
    
    
    
    
    #--------------------------------------------------------------------------
    # Create A Evaluation Dataset with the following characteristics
    #      - unscaled images with their original dimensions
    #      - batch and pad each 3D scan to have an equal depth, width and height have already been set to original dimensions
    #      - we still pad the depth dimension to max_s though
    #--------------------------------------------------------------------------
    
    if training == 2:
        # predicting a resized dataset, i.e. all have same width height?
        resized = 'resized' in params['eval_dataset_path']
        dataset = Dataset.load_dataset(
            os.path.join(package_root, params['eval_dataset_path'])
        ).create_tf_dataset(
            resized=resized
        ).padded_batch(
            # we have different sized test scans so we need batch 1
            batch_size=1,
            padded_shapes=(
                [max_s, None, None, 1],
                [max_s, None, None, params['num_classes']]
            )
        )
    
    
    
    
    #--------------------------------------------------------------------------
    # Create A Testing Dataset with the following characteristics
    #      - unscaled images with their original dimensions
    #      - batch and pad each 3D scan to have an equal depth, width and height have already been set to original dimensions
    #      - we still pad the depth dimension to max_s though
    #--------------------------------------------------------------------------
    
    if training == 3:
        # predicting a resized dataset, i.e. all have same width height?
        resized = 'resized' in params['test_dataset_path']
        dataset = Dataset.load_dataset(
            os.path.join(package_root, params['test_dataset_path'])
        ).create_tf_dataset(
            resized=resized
        ).padded_batch(
            # we have different sized test scans so we need batch 1
            batch_size=1,
            padded_shapes=(
                [max_s, None, None, 1],
                [max_s, None, None, params['num_classes']]
            )
        )
    
    
    
    
    
    
    #--------------------------------------------------------------------------
    # Create one iterator and initialize it with dataset output types and shapes
    #   - iterator: represents the state of iterating through a Dataset and to retrieve real values.
    #   - there are four types of iterators: one shot, initializable, reinitializable, and feedable.
    #   - iterator.get_next(): yields the next element of a dataset and returns a tensor objects corresponding to the next elemwnt
    #   - e.g images, labels = iterator.get_next()
    #--------------------------------------------------------------------------
    
    # we create a generic Iterator
    iterator = tf.data.Iterator.from_structure(
        dataset.output_types,
        dataset.output_shapes
    )
    
    # create the initialisation operations
    dataset_init_op = iterator.make_initializer(dataset)
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, dataset_init_op)
    
    # get the next element as before
    next_element = iterator.get_next()
    
    
    
    
    
    #--------------------------------------------------------------------------
    # Get labels from estimator!
    #   - extremely hack way of getting tf.estimator to return labels at pred time
    #   - see https://github.com/tensorflow/tensorflow/issues/17824
    #   - x: image
    #   - y:labels
    #--------------------------------------------------------------------------
    
    features = {'x': next_element[0], 'y': next_element[1]}
    return features, next_element[1]