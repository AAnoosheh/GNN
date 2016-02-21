require 'preprocess'
require 'nn'
require 'F_Module'
require 'G_Module'


-- RETRIEVE DATA
local data_path = 'data/final_dataset.mat'
dataSet = getDataset(data_path)
function dataSet:size() return #dataSet end

-- SET PARAMS
local nStates = 1
local useLabelledEdges = false
local maxSteps = 10
local forwardStopCoef = 1e-3
local maxIter = 25
local backwardStopCoef = 1e-3
local lr = .005

local numHiddensF = 3;
local numHiddensG = 15;

-- INFER LAYER INPUT SIZES
local numLabels = dataSet[1][1][6]:size(1)
local numUnmasked = dataSet[1][1][7]:size(1)
local numOutstates = dataSet[1][2]:size(1)

-- BUILD NET
net = nn.Sequential()
local F = nn.F_Module(nStates, maxSteps, forwardStopCoef, maxIter, backwardStopCoef)
local G = nn.G_Module(nStates, numUnmasked, numOutstates)

local F_internal = nn.Sequential()
F_internal:add( nn.Linear(nStates+2*numLabels, numHiddensF) );
F_internal:add( nn.Tanh() )
F_internal:add( nn.Linear(numHiddensF, nStates) );

local G_internal = nn.Sequential()
G_internal:add( nn.Linear(nStates+numLabels, numHiddensG) )
G_internal:add( nn.Tanh() )
G_internal:add( nn.Linear(numHiddensG, numOutstates) );

F:add(F_internal)
G:add(G_internal)
net:add(F)
net:add(G)


-- TRAIN
trainer = nn.StochasticGradient(net, nn.MSECriterion())
trainer.learningRate = lr
trainer.maxIteration = 2
trainer:train(dataSet)


-- TEST



-- TEMPORARY TESTS
-- for _,d in pairs(dataSet) do
--     local yp = net:forward(d[1])
-- end

-- local grad = torch.Tensor({-.2,.2,.5,-.9,.4,-.4}):resize(6,1)

-- local t = torch.rand(16,6)
-- local msk = dataSet[1][1][7]
-- local i = {t,msk}
-- local o = G:forward(i)
-- local gp = G:backward(i, grad)
-- print(gp)

-- local i = dataSet[1][1]
-- local o = net:forward(i)
-- local gp = net:backward(i, grad)