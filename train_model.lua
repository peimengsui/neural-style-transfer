require 'torch'
require 'image'
require 'nn'
require 'optim'
require 'source-fast.PerceptualCriterion'
require 'source-fast.utility'
require 'source-fast.ps_DataLoader'
require 'source-fast.img_transform_net'

local opt = require 'source-fast.option'


--local testopt = require 'testoption'
--local testmodels = require 'testmodels'

--assert(testmodels,'Model file not loaded')
local function main(opt)
	print("-- Entering main")
	if opt.cuda==true then
		print('-- Using GPU')
		require 'cutorch'
		require 'cunn'
		cutorch.setDevice(1)
		if opt.cudnn==true then
		    require 'cudnn' 
		    cudnn.benchmark = true
		end
	else
		print('-- Using CPU')
	end
	-- Variables
	torch.setdefaulttensortype('torch.FloatTensor')


	-- style image preprocess
	local style_image = image.load(opt.style_image,3)	-- style image
	style_image = image.scale(style_image,opt.image_size)
	local C, H, W = style_image:size(1), style_image:size(2), style_image:size(3)
	style_image:resize(1, C, H, W)
	print('-- Preprocessing style image')
	style_image = preprocess(style_image,opt.image_size)
	if opt.cuda == true then
		style_image = style_image:cuda()
	end

	print('-- Loading Image Transformation Network')
	local imgTransNet = build_model(opt)
	--local imgTransNet = testmodels.build_model(testopt)
	assert(imgTransNet,'Error loading Image Transformation Network')
	if opt.cuda==true then
		imgTransNet:cuda()
		if opt.cudnn==true then
			cudnn.convert(imgTransNet,cudnn)
		end
	end

	local params, grad_params = imgTransNet:getParameters()
	print('-- Loading Perceptual Criterion')
	local perceptCrit = nn.PerceptualCriterion(opt)
	if opt.cuda==true then
		perceptCrit:cuda()
	end
	perceptCrit:setStyleTarget(style_image)

	--  feval
	local num_iter = 0
	local loader = DataLoader(opt)

	local function feval(x)
		num_iter = num_iter+1

		--local inpImg = image.load(opt.content_image,3)
		--assert(inpImg,'Error when loading training image')
		--inpImg = preprocess(inpImg,opt.image_size)
		local inpImg = loader:getBatch('train')
		if opt.cuda==true then
			inpImg = inpImg:cuda()
		end
		local pred = imgTransNet:forward(inpImg)
		local target = {content = inpImg}
		local percept_loss = perceptCrit:forward(pred,target)
		local gradCrit = perceptCrit:backward(pred,target)
		grad_params:zero()
		imgTransNet:backward(inpImg,gradCrit)

		--saveTempImg(num_iter,opt.output_path,opt.jobid..'_input',inpImg,opt)
		--saveTempImg(num_iter,opt.output_path,opt.jobid..'_output',pred,opt)
		--saveTempModel(num_iter,opt.output_model_path,imgTransNet)
		--saveGradImg(num_iter,grad)
		return percept_loss, grad_params
	end

	-- optim configuration
	local config ={
			learningRate = opt.lr
		  }


	-- running optimizer
	print('-- training using ADAM optim --')
	imgTransNet:training()
	local max_iteration = opt.max_train / 2
	local check_point = max_iteration / 10
	for i = 1, max_iteration do
		local _, train_loss = optim.adam(feval,params,config)
		printState(i,train_loss[1],'train loss')
		if i%check_point==0 then
			-- switch to Valid dataset
			imgTransNet:evaluate()
			loader:reset('val')
			local inpImg = loader:getBatch('val')
			--local inpImg = image.load(opt.content_image,3)
			--inpImg = preprocess(inpImg,opt.image_size)
			if opt.cuda==true then
				inpImg = inpImg:cuda()
			end
			local pred = imgTransNet:forward(inpImg)
			local valid_loss = perceptCrit:forward(pred,{content=inpImg})
			printState(i,valid_loss,'validation loss')
			imgTransNet:training()
		end
	end
	
	print('-- training finished!')
	imgTransNet:clearState()
	if opt.cuda==true then
		if opt.cudnn==true then
			cudnn.convert(imgTransNet,nn)
		end
		imgTransNet:float()
	end
	torch.save(opt.output_model_path..'model_'..opt.jobid..'.t7',imgTransNet,'ascii')
	print('The End!')

end







main(opt)




