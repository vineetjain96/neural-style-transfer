import numpy as np
import tensorflow as tf
import scipy.misc
from argparse import ArgumentParser
import vgg19
import losses

CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e4
ITERATIONS = 1000
LEARNING_RATE = 1e1
OUTPUT = 'result.jpg'
MODEL_PATH = 'imagenet-vgg-verydeep-19.mat'

content_layers = ['relu5_2']
style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

def build_argparser():
	parser = ArgumentParser()

	parser.add_argument('--content',
			dest='content',	help='Content Image',
			type=str, required=True)
	parser.add_argument('--style',
			dest='style', help='Style Image',
			type=str, required=True)
	parser.add_argument('--output',
			dest='output', help='Output Image',
			type=str, default=OUTPUT)
	parser.add_argument('--content-weight',
			dest='content_weight', help='Content Image Weight',
			type=float, default=CONTENT_WEIGHT)
	parser.add_argument('--style-weight',
			dest='style_weight', help='Style Image Weight',
			type=float, default=STYLE_WEIGHT)
	parser.add_argument('--width',
			dest='width', help='Width of Output Image', type=int)
	parser.add_argument('--iterations',
			dest='iterations', help='Number of Iterations (default: %(default)s)',
			type=int, default=ITERATIONS)
	parser.add_argument('--model',
			dest='model', help='File containing Model Parameters (default: %(default)s)',
			type=str, default=MODEL_PATH)
	parser.add_argument('--learning-rate',
			dest='learning_rate', help='Learning Rate (default: %(default)s)',
			type=float, default=LEARNING_RATE)
	return parser



def stylize(model_file, init_image, content_image, style_image, content_weight,
			style_weight, iterations, learning_rate):
	model, mean = vgg19.build_net(model_file, content_image)

	content_image = preprocess(content_image, mean)
	style_image = preprocess(style_image, mean)
	init_image = preprocess(init_image, mean)


	with tf.Session() as sess:
		L_content = losses.total_content_loss(sess, model, content_image, content_layers)
		L_style = losses.total_style_loss(sess, model, style_image, style_layers)

		L_total = L_content + L_style

		optimizer = tf.train.AdamOptimizer(learning_rate)

		train_op = optimizer.minimize(L_total)
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		sess.run(model['input'].assign(init_image))

		i = 0
		while (i < iterations):
			sess.run(train_op)
			print 'Iteration '+str(i)+'/1000'
			i += 1

		output_image = sess.run(model['input'])

	output_image = postprocess(output_image, mean)
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

def main():
	parser = build_argparser()
	args = parser.parse_args()

	content_image = scipy.misc.imread(args.content).astype(np.float32)
	style_image = scipy.misc.imread(args.style).astype(np.float32)

	width = args.width
	if width is not None:
		h = int(math.floor((float(content_image.shape[0])*width/float(content_image.shape[1]))))
		w = width
		content_image = scipy.misc.imresize(content_image, (h,w))
	style_image = scipy.misc.imresize(style_image, content_image.shape)

	init_image = content_image

	output_image = stylize(model_file=args.model, 
						init_image=init_image,
						content_image=content_image,
						style_image=style_image,
						content_weight = args.content_weight,
						style_weight = args.style_weight,
						iterations=args.iterations,
						learning_rate=args.learning_rate)

	
	scipy.misc.imsave(args.output, output_image)
	return

if __name__ == '__main__':
    main()