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
POOLING = 'max'

content_layers = ['relu5_2']
style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

def build_argparser():
	parser = ArgumentParser()

	parser.add_argument('--content',
			dest='content',	help='Content Image',
			type=str, required=True)
	parser.add_argument('--styles',
			dest='styles', help='One or more Style Images',
			nargs='+', type=str, required=True)
	parser.add_argument('--output',
			dest='output', help='Output Image',
			type=str, default=OUTPUT)
	parser.add_argument('--content-weight',
			dest='content_weight', help='Content Image Weight (default: %(default)s)',
			type=float, default=CONTENT_WEIGHT)
	parser.add_argument('--style-weight',
			dest='style_weight', help='Style Image Weight (default: %(default)s)',
			type=float, default=STYLE_WEIGHT)
	parser.add_argument('--style-blend-weights',
			dest='style_blend_weights', help='Individual Weights for blending Style Image',
			nargs='+', type=float)
	parser.add_argument('--width',
			dest='width', help='Width of Output Image', type=int)
	parser.add_argument('--iterations',
			dest='iterations', help='Number of Iterations (default: %(default)s)',
			type=int, default=ITERATIONS)
	parser.add_argument('--learning-rate',
			dest='learning_rate', help='Learning Rate (default: %(default)s)',
			type=float, default=LEARNING_RATE)
	parser.add_argument('--model',
			dest='model', help='File containing Model Parameters (default: %(default)s)',
			type=str, default=MODEL_PATH)
	parser.add_argument('--pooling', 
			dest='pooling', help='Max or Average Pooling',
			choices=['max', 'avg'], default=POOLING)
	return parser



def stylize(model_file, initial, content, styles, content_weight,
			style_weight, style_blend_weights, iterations, learning_rate, pooling):
	model, mean = vgg19.build_net(model_file, initial, pooling)

	content_image = preprocess(content, mean)
	style_images = [preprocess(style, mean) for style in styles]
	init_image = preprocess(initial, mean)


	with tf.Session() as sess:
		L_content = losses.total_content_loss(sess, model, content_image, content_layers)
		L_style = losses.total_style_loss(sess, model, style_images, style_blend_weights, style_layers)

		L_total = content_weight*L_content + style_weight*L_style

		optimizer = tf.train.AdamOptimizer(learning_rate)

		train_op = optimizer.minimize(L_total)
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		sess.run(model['input'].assign(init_image))

		i = 0
		while (i < iterations):
			sess.run(train_op)
			print 'Iteration '+str(i+1)+'/'+str(iterations)
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

def get_image(path):
	img = scipy.misc.imread(path)
	img = img.astype(np.float32)
	if len(img.shape) == 2:
		img = np.dstack((img,img,img))
	return img


def main():
	parser = build_argparser()
	args = parser.parse_args()

	content_image = get_image(args.content)
	style_images = [get_image(style) for style in args.styles]

	width = args.width
	if width is not None:
		h = int(math.floor((float(content_image.shape[0])*width/float(content_image.shape[1]))))
		content_image = scipy.misc.imresize(content_image, (h,width))
	for i in range(len(style_images)):
		style_images[i] = scipy.misc.imresize(style_images[i], content_image.shape)

	style_blend_weights = args.style_blend_weights
	if style_blend_weights is None:
		style_blend_weights = [1.0/len(style_images) for _ in style_images]
	else:
		total_blend_weight = sum(style_blend_weights)
		style_blend_weights = [weight/total_blend_weight for weight in style_blend_weights]

	init_image = content_image

	output_image = stylize(model_file=args.model, 
					initial=init_image,
					content=content_image,
					styles=style_images,
					content_weight=args.content_weight,
					style_weight=args.style_weight,
					style_blend_weights=style_blend_weights,
					iterations=args.iterations,
					learning_rate=args.learning_rate,
					pooling=args.pooling)

	
	scipy.misc.imsave(args.output, output_image)
	return

if __name__ == '__main__':
    main()