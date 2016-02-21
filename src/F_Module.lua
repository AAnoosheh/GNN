local F_Module, parent = torch.class('nn.F_Module', 'nn.Container')

function F_Module:__init(nStates, maxSteps, forwardStopCoef, maxIter, backwardStopCoef)
    parent.__init(self)

    self.nStates = nStates
    self.maxSteps, self.forwardStopCoef = maxSteps, forwardStopCoef
    self.maxIter, self.backwardStopCoef = maxIter, backwardStopCoef

    self.numIter, self.labels, self.stepGrads = nil, nil, nil
end


--FORWARD
function F_Module:updateOutput(input)
    local childOfArc, fatherOfArc, childToArcMatrix, nNodes, nArcs, label, mask = unpack(input)

    self.labels = torch.cat(label:index(2, childOfArc),
                             label:index(2, fatherOfArc), 1)
    local input_vector = torch.cat(torch.zeros(self.nStates, nArcs),
                                   self.labels, 1)

    self.output = torch.zeros(self.nStates, nNodes)

    for _=1,self.maxSteps do
        input_vector[{{1,self.nStates},{}}] = self.output:index(2, fatherOfArc)

        local new_out = self.modules[1]:forward(input_vector:t())
        new_out = new_out:t() * childToArcMatrix

        local stabCoef = relative_diff(self.output, new_out)
        self.output:copy(new_out)

        if stabCoef < self.forwardStopCoef then break end
    end

    self.output = {torch.cat(self.output,
                            label, 1),
                   mask}
    return self.output
end


--BACKWARD
function F_Module:updateGradInput(input, gradOutput)
    local childOfArc, fatherOfArc, childToArcMatrix, nNodes, nArcs, label, mask = unpack(input)

    gradOutput = gradOutput[{{1,self.nStates},{}}]
    local out = self.output[1][{{1,self.nStates},{}}]

    self.gradInput:resizeAs(gradOutput):copy(gradOutput)
    self.stepGrads = {}

    local input_vector = torch.cat(out:index(2, fatherOfArc),
                                   self.labels, 1)
    self.numIter = self.maxIter

    for i=1,self.maxIter do
        local grads = self.gradInput:index(2, fatherOfArc)
        table.insert(self.stepGrads, grads)

        local new_out = self.modules[1]:updateGradInput(input_vector:t(), grads:t())
        new_out = new_out:t()[{{1,self.nStates}}]
        new_out = new_out * childToArcMatrix

        local stabCoef = relative_diff(self.gradInput, new_out)
        self.gradInput:copy(new_out)

        if stabCoef < self.backwardStopCoef then
            self.numIter = i
            break
        end
    end

    return self.gradInput
end


function F_Module:accGradParameters(input, gradOutput, scale)
    local fatherOfArc, nArcs = input[2], input[5]
    scale = scale / (self.numIter * nArcs)

    local out = self.output[1][{{1,self.nStates},{}}]
    local input_vector = torch.cat(out:index(2, fatherOfArc),
                                   self.labels, 1)

    for _,stepGrad in pairs(self.stepGrads) do
        self.modules[1]:accGradParameters(input_vector:t(), stepGrad:t(), scale)
    end
end

function F_Module:accUpdateGradParameters(input, gradOutput, scale)
    self:accGradParameters(input, gradOutput, 1)
    for _, mod in pairs(self.modules[1].modules) do
        mod:updateParameters(scale)
        mod:zeroGradParameters()
    end
end


function relative_diff(first, second)
    local denom = torch.abs(second):add(1e-30)
    num = torch.abs(first-second)
    return torch.cdiv(num,denom):sum()
end