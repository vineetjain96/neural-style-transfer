import numpy as np
import tensorflow as tf

def content_layer_loss(p, x):
	loss = 0.5 * tf.reduce_sum(tf.square(x-p))
	return loss

def style_layer_loss(a, x):
	A_gram = gram_matrix(a)
	G_gram = gram_matrix(x)
	area, depth = A_gram.get_shape()
	M = area.value
	N = depth.value
	loss = (1./(4 * M**2 * N**2)) * tf.reduce_sum(tf.square(G_gram-A_gram))
	return loss

def gram_matrix(x):
	shape = x.get_shape()
	d = shape[3].value
	F = tf.reshape(x, shape=[-1, d])
	G = tf.matmul(tf.transpose(F), F)
	return G

def total_content_loss(sess, model, content_image, content_layers):
	sess.run(model['input'].assign(content_image))
	content_loss = 0
	for layer in content_layers:
		P = sess.run(model[layer])
		P = tf.convert_to_tensor(P)
		F = model[layer]
		content_loss += content_layer_loss(P, F)
	return content_loss


def total_style_loss(sess, model, style_images, style_blend_weights, style_layers):
	total_style_loss = 0
	for style, weight in zip(style_images, style_blend_weights):
		sess.run(model['input'].assign(style))
		style_loss = 0
		for layer in style_layers:
			A = sess.run(model[layer])
			A = tf.convert_to_tensor(A)
			G = model[layer]
			style_loss += style_layer_loss(A, G)
		total_style_loss += style_loss * weight
	return total_style_loss
