require 'torch'
require 'nn'
require 'source-fast.Shave'
require 'source-fast.TVLoss'

--local norm = nn.SpatialBatchNormalization
--local norm = nn.InstanceNormalization

local function add_conv(model,nIn,nOut,kW,kH,dW,dH,padW,padH)
	model:add(nn.SpatialConvolution(nIn,nOut,kW,kH,dW,dH,padW,padH))
	model:add(nn.SpatialBatchNormalization(nOut))
	model:add(nn.ReLU())
end

local function add_upsample(model,nIn,nOut,kW,kH,dW,dH,padW,padH,adjW,adjH)
	model:add(nn.SpatialFullConvolution(nIn,nOut,kW,kH,dW,dH,padW,padH,adjW,adjH))
	model:add(nn.SpatialBatchNormalization(nOut))
	model:add(nn.ReLU())
end

local function build_res(nMap)
	local res_block = nn.Sequential()
	local concat = nn.ConcatTable()
	local conv_part = nn.Sequential()
	conv_part:add(nn.SpatialConvolution(nMap,nMap,3,3))
	conv_part:add(nn.SpatialBatchNormalization(nMap))
	conv_part:add(nn.ReLU())
	conv_part:add(nn.SpatialConvolution(nMap,nMap,3,3))
	conv_part:add(nn.SpatialBatchNormalization(nMap))
	concat:add(conv_part)
	concat:add(nn.Shave(2))
	res_block:add(concat):add(nn.CAddTable())
	return res_block
end


function build_model(opt)
	local model = nn.Sequential()
	-- reflection necessary?
	model:add(nn.SpatialReflectionPadding(40,40,40,40))
	add_conv(model, 3,  32,  9,9,1,1,4,4)
	add_conv(model, 32, 64,  3,3,2,2,1,1)
	add_conv(model, 64, 128, 3,3,2,2,1,1)

	for i=1, 5 do
		local layer = build_res(128)
		model:add(layer)
	end

	add_upsample(model, 128,64,3,3,2,2,1,1,1,1)
	add_upsample(model, 64, 32,3,3,2,2,1,1,1,1)

	model:add(nn.SpatialConvolution(32,3,9,9,1,1,4,4))
	model:add(nn.Tanh())
	model:add(nn.MulConstant(150))
	model:add(nn.TVLoss(opt.TV_weight))
	return model
end






















