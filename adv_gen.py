import tensorflow as tf
from Xception import *
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import foolbox
def create_fmodel():
	NUM_CLASSES = 200
	with tf.name_scope('input'):
		input_images = tf.placeholder(tf.float32,  shape = (1, 64, 64, 3), name = 'input_images')
		input_images_re = tf.image.resize_nearest_neighbor(input_images,(299,299))
	logits = XceptionModel(input_images_re, NUM_CLASSES, is_training = False, data_format='channels_last')
	saver = tf.train.Saver(var_list = tf.global_variables())
	sess = tf.InteractiveSession()
	init = tf.global_variables_initializer()
	sess.run(init)
	fmodel = foolbox.models.TensorFlowModel(input_images, logits, (0, 255),channel_axis=3, preprocessing=(0,1))
	saver.restore(sess, "./model_save_early/center_loss.ckpt") #restore weight
	return fmodel

