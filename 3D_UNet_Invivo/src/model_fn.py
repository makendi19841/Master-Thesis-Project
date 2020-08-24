"""
3D U-Net network, for MRA Atherosclerotic Carotid Scans.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from src.network import unet_3d_network


def model_fn(features, labels, mode, params):
    """
    # Custom estimator setup as per docs and guide:---good one
    # https://www.tensorflow.org/guide/custom_estimators

    # Several ideas taken from standford:       
    # https://github.com/cs230-stanford/cs230-code-examples/tree/master/tensorflow/vision
    
    
    # How to choose cross-entropy loss in tensorflow?
    # https://stackoverflow.com/questions/47034888/how-to-choose-cross-entropy-loss-in-tensorflow
    
    
    # Reading tf.estimator.Estimator:
    # https://medium.com/learning-machine-learning/introduction-to-tensorflow-estimators-part-1-39f9eb666bc7
    # https://medium.com/@tijmenlv/an-advanced-example-of-tensorflow-estimators-part-1-3-c9ffba3bff03
    # https://medium.com/@tijmenlv/an-advanced-example-of-tensorflow-estimators-part-2-3-5569fb93d9f8
    # https://gist.github.com/hadifar/4d47c1498db060b6cc0124b984e9ebbe
    # https://towardsdatascience.com/first-contact-with-tensorflow-estimator-69a5e072998d
    # https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
    # https://torres.ai/
    
       
    # Tensorflow model analysis:
    # https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis
    # https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/eval_saved_model/example_trainers
    
    # What does actually the logits mean?
    # https://stats.stackexchange.com/questions/52825/what-does-the-logit-value-actually-mean
    
    # Cross entropy for tensorflow
    # https://mmuratarat.github.io/2018-12-21/cross-entropy
    # https://stackoverflow.com/questions/47034888/how-to-choose-cross-entropy-loss-in-tensorflow
    
    
    Args:
        features: This is batch_features from input_fn.
        labels: This is batch_labels from input_fn.
        mode (:class:`tf.estimator.ModeKeys`): Train, eval, or predict.
        params (dict): Params for setting up the model. Expected keys are:
            depth (int): Depth of the architecture.
            n_base_filters (int): Number of conv3d filters in the first layer.
            num_classes (int): Number of mutually exclusive output classes.
            class_weights (:class:`numpy.array`): Weight of each class to use.
            learning_rate (float): LR to use with Adam.
            batch_norm (bool): Whether to use batch_norm in the conv3d blocks.
            display_steps (int): How often to log about progress.

    Returns:
        :class:`tf.estimator.Estimator`: A 3D U-Net network, as TF Estimator.
    """
    print('--------------------------------------------------------------------')
    print('labels are: ' + str(features['y']))
    print('features are: ' + str(features['x']))
    print('--------------------------------------------------------------------')
    

    # -------------------------------------------------------------------------
    # get logits from 3D U-Net
    # -------------------------------------------------------------------------

    training = mode == tf.estimator.ModeKeys.TRAIN
    logits = unet_3d_network(inputs=features['x'], params=params, training=training)

    # -------------------------------------------------------------------------
    # predictions - for PREDICT and EVAL modes
    # -------------------------------------------------------------------------

    prediction = tf.argmax(logits, axis=-1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes': prediction,
            'truth': tf.argmax(features['y'], -1),
            'probabilities': tf.nn.softmax(logits, axis=-1)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # -------------------------------------------------------------------------
    # loss - for TRAIN and EVAL modes
    # -------------------------------------------------------------------------
    
    #case0: 
    # weighted softmax, see https://stackoverflow.com/a/44563055
    # https://stackoverflow.com/questions/40198364/how-can-i-implement-a-weighted-cross-entropy-loss-in-tensorflow-using-sparse-sof/46984951#46984951
    # https://stackoverflow.com/questions/40698709/tensorflow-interpretation-of-weight-in-weighted-cross-entropy
    
   
    class_weights = tf.cast(tf.constant(params['class_weights']), tf.float32)
    class_weights = tf.reduce_sum(
        tf.cast(features['x'], tf.float32) * class_weights, axis=-1
    )
      
    loss = tf.losses.softmax_cross_entropy(
        logits=logits,
        onehot_labels=labels,
        weights=class_weights
    )
    
    tf.summary.scalar('loss', loss)
		
    
    # case1: tf.nn.sparse_softmax_cross_entropy_with_logits
#    labels = tf.argmax(labels, -1) 
#    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
#        labels=labels,
#        logits=logits
#    )
    # Add weight decay to the loss
#    loss += params['weight_decay'] * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'kernel' in v.name])
#    tf.summary.scalar('loss', loss)
    

    # -------------------------------------------------------------------------
    # metrics: mean IOU - for TRAIN and EVAL modes
    # -------------------------------------------------------------------------
    
    # case0:
    
    labels_dense = tf.argmax(labels, -1)
    iou = tf.metrics.mean_iou(
        labels=labels_dense,
        predictions=tf.cast(prediction, tf.int32),
        num_classes=params['num_classes'],
        name='iou_op'
    )
    metrics = {'iou': iou}
    tf.summary.scalar('iou', iou[0])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)
    
    #case1:
#    iou = tf.metrics.mean_iou(
#        labels=labels,
#        predictions=tf.cast(prediction, tf.int32),
#        num_classes=params['num_classes'],
#        name='iou_op'
#    )
#    metrics = {'iou': iou}
#    tf.summary.scalar('iou', iou[0])
#
#    if mode == tf.estimator.ModeKeys.EVAL:
#        return tf.estimator.EstimatorSpec(
#            mode, loss=loss, eval_metric_ops=metrics)

    # -------------------------------------------------------------------------
    # train op: training optimization
    # train op - for TRAIN
    # -------------------------------------------------------------------------

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    global_step = tf.train.get_or_create_global_step()

    if params['batch_norm']:
        # as per TF batch_norm docs and also following https://goo.gl/1UVeYK
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)
    else:
        train_op = optimizer.minimize(loss, global_step=global_step)
        accuracy = tf.metrics.accuracy(labels, predictions['classes'])
        metrics = {'accuracy': accuracy}
        
        # create a tensor named train_accuracy for logging purposes
        tf.identity(accuracy[1], name='train_accuracy')
        tf.summary.scalar('train_accuracy', accuracy[1])
        
    
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
