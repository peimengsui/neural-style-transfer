
require 'nn'
require 'torch'


local ContentLoss, parent = torch.class('nn.ContentLoss','nn.Module')


function ContentLoss:__init(strength)
	parent.__init(self)
	self.strength = strength
	self.target = torch.Tensor()
	self.loss = 0
	self.crit = nn.MSECriterion()
	self.mode = 'none'
end


function ContentLoss:updateOutput(input)
	if self.mode == 'capture' then
		self.target = input:clone()
	elseif self.mode=='loss' then
		self.loss = self.crit:forward(input,self.target) * self.strength
	end
	self.output = input
	return self.output
end


function ContentLoss:updateGradInput(input, gradOutput)
	if self.mode=='loss' then
		self.gradInput = self.crit:backward(input,self.target) *self.strength
		self.gradInput:add(gradOutput)
	elseif self.mode=='capture' or self.mode=='none' then
		self.gradInput = gradOutput
	end
	return self.gradInput
end

function ContentLoss:setMode(mode)
	if mode~='loss' and mode~='capture' and mode~='none' then
		error("ContentLoss can only be 'loss' or 'capture' or 'none'! ")
	end
	self.mode = mode
end