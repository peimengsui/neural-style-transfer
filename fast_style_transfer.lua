require 'nn'
require 'image'
require 'torch'
require 'source-fast.utility'
require 'source-fast.Shave'
require 'source-fast.TVLoss'

local opt = require 'source-fast.option'


local function main(opt)
	torch.setdefaulttensortype('torch.FloatTensor')
	local model = torch.load(opt.use_model,'ascii')
	assert(model,'No model found!')
	local input = image.load(opt.input_img,3)
	local input = image.scale(input,opt.input_size)
	input:resize(1,input:size(1),input:size(2),input:size(3))
	input = preprocess(input)
	if opt.cuda==true then
		print('-- Using GPU')
		require 'cutorch'
		require 'cunn'
		model:cuda()
		input = input:cuda()
	else
		print('-- Using CPU')
	end
	print('-- Generating Image')
	local pred = model:forward(input)
	pred = depreprocess(pred:float())[1]
	print('-- Writing image to ',opt.output_img)
	image.save(opt.output_img,pred)
end


main(opt)