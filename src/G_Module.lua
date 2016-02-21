local G_Module, parent = torch.class('nn.G_Module', 'nn.Container')

function G_Module:__init(nStates, nUnmasked, nOutstates)
    parent.__init(self)

    self.nStates = nStates
    self.nUnmasked, self.nOutstates = nUnmasked, nOutstates
end


--FORWARD
function G_Module:updateOutput(input)
    local input_vector, mask = unpack(input)
    self.output:resize(self.nOutstates, self.nUnmasked)

    input_vector = input_vector:index(2, mask)
    self.output:copy( self.modules[1]:forward(input_vector:t()) )

    return self.output
end


--BACKWARD
function G_Module:updateGradInput(input, gradOutput)
    local input_vector, mask = unpack(input)
    self.gradInput:resizeAs(input_vector):zero()

    input_vector = input_vector:index(2, mask)
    local out = self.modules[1]:backward(input_vector:t(), gradOutput:t())
    self.gradInput:indexCopy(2, mask, out:t())

    return self.gradInput
end

function G_Module:accGradParameters(input, gradOutput, scale)
    local input_vector, mask = unpack(input)
    scale = scale / self.nUnmasked

    input_vector = input_vector:index(2, mask)
    self.modules[1]:accGradParameters(input_vector:t(), gradOutput:t(), scale)
end

function G_Module:accUpdateGradParameters(input, gradOutput, scale)
    self:accGradParameters(input, gradOutput, 1)

    for _, mod in pairs(self.modules[1].modules) do
        mod:updateParameters(scale)
        mod:zeroGradParameters()
    end
end