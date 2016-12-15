require 'nn'
require 'torch'
require 'image'
--[[
-- return a module that calculate a C*C matrix from a C*H*W matrix
-- or calculate a N*C*C matrix from a N*C*H*W matrix
function GramModule()
	local mod = nn.Sequential()
	mod:add(nn.View(-1):setNumInputDims(2))
	local concat = nn.ConcatTable()
	concat:add(nn.Identity())
	concat:add(nn.Identity())
	mod:add(concat)
	mod:add(nn.MM(false, true))
	return mod
end
--]]
-- return a module that calculate a C*C matrix from a C*H*W matrix
-- or calculate a N*C*C matrix from a N*C*H*W matrix
gram,parent = torch.class('nn.GramModule','nn.Module')

function gram:__init()
	parent.__init(self)
	self.tmp = torch.Tensor()
end


function gram:updateOutput(input)
	self.output = nil
	local C,H,W
	if input:dim()==3 then
		C, H, W = input:size(1), input:size(2), input:size(3)
		local flat = input:view(C,H*W)
		self.output = torch.mm(flat,flat:t())
	elseif input:dim()==4 then
		local N = input:size(1)
		C, H, W = input:size(2), input:size(3), input:size(4)
		local flat = input:view(N,C,H*W)
		self.output = torch.bmm(flat,flat:transpose(2,3))
	end
	self.output:div(C * H * W)
	return self.output
end


--  ??????? 
function gram:updateGradInput(input,gradOutput)
	local C, H, W
	if input:dim() == 3 then
	    C, H, W = input:size(1), input:size(2), input:size(3)
	    local flat = input:view(C, H * W)
	    self.tmp:resize(C, H * W)
	    self.tmp:mm(gradOutput, flat)
	    self.tmp:addmm(gradOutput:t(), flat)
	    self.gradInput = self.tmp:view(C, H, W)
	elseif input:dim() == 4 then
	    local N = input:size(1)
	    C, H, W = input:size(2), input:size(3), input:size(4)
	    local flat = input:view(N, C, H * W)
	    self.tmp:resize(N, C, H * W)
	    self.tmp:bmm(gradOutput, flat)
	    self.tmp:baddbmm(gradOutput:transpose(2, 3), flat)
	    self.gradInput = self.tmp:view(N, C, H, W)
	end
	self.tmp:div(C * H * W)
	assert(self.gradInput:isContiguous())
	return self.gradInput
end


-- preprocess the image for VGG-19
-- input: (N, C, H, W)
-- 1. convert the image from RGB to BGR,
-- 2. convert the scale from [0,1] to [0,255]
-- 3. subtract the mean pixel
function preprocess(img,size)
	--img = image.scale(img,size,size)
	local pic_mean = torch.FloatTensor{103.939, 116.779, 123.68}
	local bgr = torch.LongTensor{3,2,1}
	img = img:index(2,bgr):mul(255)
	pic_mean = pic_mean:view(1,3,1,1):expandAs(img)
	img:add(-1,pic_mean)
	return img
end


-- reverse the preprocessing procedure
function depreprocess(img)
	--local C, H, W = img:size(2), img:size(3), img:size(4)
	--img:resize(C, H, W)
	local pic_mean = torch.FloatTensor{103.939, 116.779, 123.68}
	local rgb = torch.LongTensor{3,2,1}
	pic_mean = pic_mean:view(1,3,1,1):expandAs(img)
	img:add(pic_mean)
	img = img:index(2,rgb):div(255)
	return img
end


-- debugging utilities for training procedure
function printState(iter,loss,name,opt)
	if (iter % 1) ==0 then
		print('-- Iteration: ',iter,' ',name,':',loss)
	end
end

function saveTempImg(iter,path,name,img,opt)
	if iter>0 and iter%(opt.max_train/20)==0 then
		local tmp = depreprocess(img:float())
		image.save(path..'Interim_'..name..iter..'.jpg',tmp[1])
	end
end


function saveGradImg(iter,path,img,opt)
	if iter>0 and iter%(opt.max_train/20)==0 then
		image.save(path..'grad_'..iter..'.jpg',img)
	end
end


function saveTempModel(iter,path,model,opt)
	if iter>0 and iter%(opt.max_train/20)==0 then
		print('-- Saving model at iteration ',iter)
		local saveModel = model:clone():clearState()
		if opt.cuda==true then
			require 'cudnn'
			--cudnn.convert(saveModel,nn)
			saveModel:float()
		end
		torch.save(path..'model_'..iter..'.t7',saveModel,'ascii')
	end
end


