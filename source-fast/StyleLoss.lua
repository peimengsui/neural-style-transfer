require 'nn'
require 'torch'
require 'source-fast.utility'


local StyleLoss, parent = torch.class('nn.StyleLoss','nn.Module')


function StyleLoss:__init(strength)
	parent.__init(self)
	self.strength = strength
	self.target = nil
	self.crit = nn.MSECriterion()
	self.Gram = nn.GramModule()
	self.pred = nil 		-- the calulated GramMtraix from input
	self.loss = 0
	self.mode ='none'
end



function StyleLoss:updateOutput(input)
	if self.mode =='capture' then
		self.target = self.Gram:forward(input):clone()
	elseif self.mode =='loss' then
		self.pred = self.Gram:forward(input)
		-- deal with batch input
		local target = self.target
		if self.pred:size(1)>1 and target:size(1)==1 then
			target = target:expandAs(self.pred)
		end
		self.loss = self.crit:forward(self.pred, target) * self.strength
		-- deal with batch input when back propagate
		self.ntarget = target
	end
	self.output = input
	return self.output
end



function StyleLoss:updateGradInput(input, gradOutput)
	if self.mode == 'loss' then
		local gradCrit = self.crit:backward(self.pred, self.ntarget) * self.strength
		self.gradInput = self.Gram:backward(input, gradCrit) 
		self.gradInput:add(gradOutput)
	elseif self.mode == 'capture' or self.mode == 'none' then
		self.gradInput = gradOutput
	end
	return self.gradInput
end



function StyleLoss:setMode(mode)
	if mode~='loss' and mode~='capture' and mode~='none' then
		error("StyleLoss can only be 'loss' or 'capture' or 'none'! ")
	end
	self.mode = mode
end

