import tensorflow as tf
import numpy as np
import scipy.io

def build_net(model_path, input_image):


	layers = (
	    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
	    
	    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
	    
	    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
	    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
	    
	    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
	    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
	    
	    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
	    'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
	)


	h, w, d = input_image.shape

	raw_data = scipy.io.loadmat(model_path)
	weights = raw_data['layers'][0]

	mean = raw_data['meta'][0][0][2][0][0][2]
	mean = mean.reshape((1,1,1,3))

	model = {}

	model['input'] = tf.Variable(np.zeros((1, h, w, d), dtype=np.float32))
	current_layer = model['input']
	for i, layer_name in enumerate(layers):

		layer_type = layer_name[:4]

		if layer_type == 'conv':
			W, b = weights[i][0][0][2][0]
			W = tf.constant(W)
			b = tf.constant(np.reshape(b, (-1)))
			current_layer = conv_layer(current_layer, W, b)

		elif layer_type == 'relu':
			current_layer = relu_layer(current_layer)

		elif layer_type == 'pool':
			current_layer = pool_layer(current_layer)

		model[layer_name] = current_layer

	assert len(model) == len(layers)+1
	return model, mean


def conv_layer(layer_input, weights, bias):
	conv = tf.nn.conv2d(layer_input, weights, strides=(1, 1, 1, 1), padding='SAME')
	return tf.nn.bias_add(conv, bias)

def relu_layer(layer_input):
	return tf.nn.relu(layer_input)

def pool_layer(layer_input):
	return tf.nn.max_pool(layer_input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')