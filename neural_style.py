import numpy as np
import tensorflow as tf
import scipy.misc
import vgg19
import losses

contentPath = "examples/willy_wonka.jpg"
stylePath = "examples/style.jpg"
outputPath = "examples/output.jpg"
modelPath = "imagenet-vgg-verydeep-19.mat"

content_layers = ['relu5_2']
style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

def stylize(model, content_image, style_image, init_image):
	with tf.Session() as sess:
		L_content = losses.total_content_loss(sess, model, content_image, content_layers)
		L_style = losses.total_style_loss(sess, model, style_image, style_layers)

		L_total = L_content + L_style

		learning_rate = 10
		max_iters = 1000

		optimizer = tf.train.AdamOptimizer(learning_rate)

		train_op = optimizer.minimize(L_total)
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		sess.run(model['input'].assign(init_image))

		i = 0
		while (i < max_iters):
			sess.run(train_op)
			print 'Iteration '+str(i)+'/1000'
			i += 1

		output_image = sess.run(model['input'])

	return output_image

def preprocess(image, mean):
	image = image.astype(np.float32)
	image = image[np.newaxis,:,:,:]
	image -= mean
	return image

def postprocess(image, mean):
	image += mean
	image = image[0]
	image = np.clip(image, 0, 255).astype('uint8')
	return image

content_image = scipy.misc.imread(contentPath).astype(np.float32)
style_image = scipy.misc.imread(stylePath).astype(np.float32)

model, mean = vgg19.build_net(modelPath, content_image)

shape = content_image.shape
style_image = scipy.misc.imresize(style_image, shape)

content_image = preprocess(content_image, mean)
style_image = preprocess(style_image, mean)

init_image = content_image

output_image = stylize(model, content_image, style_image, init_image)

output_image = postprocess(output_image, mean)

scipy.misc.imsave(outputPath, output_image)