require 'torch'

local cmd = torch.CmdLine()


-- training option
cmd:option('-vgg',				'models/vgg16.t7',								'Path to model')
cmd:option('-content_image',	'input/picture/tubingen.jpg',			'Path to input image')
cmd:option('-style_image',		'input/art/starry_night.jpg',			'Path to input art')
cmd:option('-output_path',		'output/',								'Path to output ')
cmd:option('-image_size',		256,										'Image crop size')
cmd:option('-cuda',				false,										'Using Cuda')
cmd:option('-cudnn',			false,										'Using Cudnn')
cmd:option('-content_layers',	'relu3_3',									'Layer for content')
cmd:option('-style_layers',		'relu1_2,relu2_2,relu3_3,relu4_3',			'Layer for style')
cmd:option('-content_weight',	1.0)
cmd:option('-style_weight',		5.0)
cmd:option('-TV_weight',		1e-6)

cmd:option('-lr',				1e-3,										'learning rate')
cmd:option('-momentum',			0.5)
cmd:option('-max_iteration',	1000,										'Max Iteration number')
cmd:option('-optimizer',		'adam',										'lbfgs|adam')
cmd:option('-check_point',		10)

cmd:option('-output_model_path',	'trained_model/')
cmd:option('-jobid',			'defaultName')



-- data loader
cmd:option('-h5_file',			'input/train.h5')
cmd:option('-batch_size',		4)
cmd:option('-max_train',		20000)



-- image transform net forward option
cmd:option('-use_model',		'trained_model/model_mosaic.t7',			'Path to trained model')
cmd:option('-input_img',		'input/picture/tubingen.jpg')
cmd:option('-input_size',		512)
cmd:option('-output_img',		'output/output.jpg')


local opt = cmd:parse(arg)
return opt