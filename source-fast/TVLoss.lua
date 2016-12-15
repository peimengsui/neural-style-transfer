require 'torch'
require 'nn'

local tv, parent = torch.class('nn.TVLoss','nn.Module')

function tv:__init(strength)
	parent.__init(self)
	self.strength = strength
end

function tv:updateOutput(input)
	self.output = input
	return self.output
end

function tv:updateGradInput(input,gradOutput)
	local x_diff = input[{{}, {}, {1, -2}, {1, -2}}] - input[{{}, {}, {1, -2}, {2, -1}}]
	local y_diff = input[{{}, {}, {1, -2}, {1, -2}}] - input[{{}, {}, {2, -1}, {1, -2}}]
	self.gradInput:resizeAs(input):zero()
	self.gradInput[{{}, {}, {1, -2}, {1, -2}}]:add(x_diff):add(y_diff)
	self.gradInput[{{}, {}, {1, -2}, {2, -1}}]:add(-1,x_diff)
	self.gradInput[{{}, {}, {2, -1}, {1, -2}}]:add(-1,x_diff)
	self.gradInput:mul(self.strength)
	self.gradInput:add(gradOutput)
	return self.gradInput
end