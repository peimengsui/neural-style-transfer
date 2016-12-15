require 'nn'
require 'source-fast.ContentLoss'
require 'source-fast.StyleLoss'
require 'source-fast.utility'
--require 'loadcaffe'

local percept,parent = torch.class('nn.PerceptualCriterion','nn.Criterion')

function percept:__init(opt)
	parent.__init(self);
	--local vgg = loadcaffe.load(opt.vgg_path .. 'VGG_ILSVRC_19_layers_deploy.prototxt', opt.vgg_path .. 'VGG_ILSVRC_19_layers.caffemodel')
	local vgg = torch.load(opt.vgg)
	local content_layer_names = opt.content_layers:split(',')
	local style_layer_names = opt.style_layers:split(',')
	self.content_layer = {}
	self.style_layer = {}
	self.style_loss = 0
	self.content_loss = 0
	local content_idx, style_idx = 1, 1
	self.net = nn.Sequential()	-- perceptual network
	print('-- Preparing Perceptual Criterion --')
	for i = 1, #vgg do 
		if (content_idx <= #content_layer_names or style_idx<= #style_layer_names) then
			local layer = vgg:get(i)
			-- replacing max-polling with average-polling
			if (torch.type(layer)=='nn.SpatialMaxPooling') then
				local kW ,kH, dW, dH = layer.kW, layer.kH, layer.dW, layer.dH
				local newLayer = nn.SpatialAveragePooling(kW,kH,dW,dH)
				self.net:add(newLayer)
			else
				self.net:add(layer)
			end

			local name = layer.name

			-- setting up ContentLoss layer
			if name == content_layer_names[content_idx] then
				print('-- Insert ContentLoss module at layer: ',name,' --')
				local content_mod = nn.ContentLoss(opt.content_weight)
				self.net:add(content_mod)
				table.insert(self.content_layer,content_mod)
				content_idx = content_idx+1
			end			

			-- setting up StyleLoss layer
			if name == style_layer_names[style_idx] then
				print('-- Insert StyleLoss module at layer: ',name,' --')
				local style_mod = nn.StyleLoss(opt.style_weight)
				self.net:add(style_mod)
				table.insert(self.style_layer,style_mod)
				style_idx = style_idx+1
			end
		end
	end
	print('-- Perceptual Preparation finished')
	vgg = nil
	collectgarbage()
end

function percept:setContentTarget(content)
	for _, layer in ipairs(self.content_layer) do
		layer:setMode('capture')
	end
	for _,layer in ipairs(self.style_layer) do
		layer:setMode('none')
	end
	--print('-- Setting content target.')
	self.net:forward(content)
end

function percept:setStyleTarget(style)
	for _, layer in ipairs(self.content_layer) do
		layer:setMode('none')
	end
	for _, layer in ipairs(self.style_layer) do
		layer:setMode('capture')
	end
	print('-- Setting style target.')
	self.net:forward(style)
end

function percept:updateOutput(input,target)
	if target.content then
		self:setContentTarget(target.content)
	end
	if target.style then
		self:setStyleTarget(target.style)
	end
	for _, layer in ipairs(self.content_layer) do
		layer:setMode('loss')
	end
	for _, layer in ipairs(self.style_layer) do
		layer:setMode('loss')
	end 
	local pred = self.net:forward(input)
	-- pass zero to gradOutput when backpropagate
	-- because gradOutput will accumulate at each ContentLoss/StyleLoss layer
	self.net_gradOutput = pred:clone():zero()
	self.content_loss = 0
	self.style_loss = 0

	-- calculate loss
	for _, layer in ipairs(self.content_layer) do
		self.content_loss = self.content_loss + layer.loss
	end
	for _, layer in ipairs(self.style_layer) do
		self.style_loss = self.style_loss + layer.loss
	end

	self.output = self.content_loss + self.style_loss
	return self.output
end

function percept:updateGradInput(input,target)
	self.gradInput = self.net:updateGradInput(input,self.net_gradOutput)
	return self.gradInput
end
























