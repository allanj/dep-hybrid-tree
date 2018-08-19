require 'nn'
require 'optim'
require 'rnn'
stringx = require 'pl.stringx'

include 'nn-crf-interface/neural_server/AbstractNeuralNetwork.lua'
include 'nn-crf-interface/neural_server/MultiLayerPerceptron.lua'
include 'nn-crf-interface/neural_server/BidirectionalLSTM.lua'
include 'nn-crf-interface/neural_server/SimpleBiLSTM.lua'
include 'nn-crf-interface/neural_server/TagBiLSTM.lua'
include 'nn-crf-interface/neural_server/ContinuousFeature.lua'
include 'nn-crf-interface/neural_server/EmbeddingLayer.lua'
include 'nn-crf-interface/neural_server/OneHot.lua'
include 'nn-crf-interface/neural_server/Utils.lua'
include 'nn-crf-interface/neural_server/optim/sgdgc.lua'

local SEED = 1337
torch.manualSeed(SEED)

-- GPU setup
local gpuid = -1
function setupGPU()
    if gpuid >= 0 then
        local ok, cunn = pcall(require, 'cunn')
        local ok2, cutorch = pcall(require, 'cutorch')
        if not ok then print('package cunn not found!') end
        if not ok2 then print('package cutorch not found!') end
        if ok and ok2 then
            print('using CUDA on GPU ' .. gpuid .. '...')
            cutorch.setDevice(gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
            cutorch.manualSeed(SEED)
        else
            print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
            print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
            print('Falling back on CPU mode')
            gpuid = -1 -- overwrite user setting
        end
    else
        print("CPU mode")
    end
end

local net
function initialize(javadata, ...)
    local timer = torch.Timer()
    local isTraining = javadata:get("isTraining")
    local optimizeInTorch = not javadata:get("optimizeNeural")
    gpuid = javadata:get("gpuid")
    if gpuid == nil then gpuid = -1 end
    setupGPU()
    if isTraining or net == nil then
        -- re-seed
        torch.manualSeed(SEED)
        if gpuid >= 0 then cutorch.manualSeed(SEED) end
        local networkClass = javadata:get("class")
        if networkClass == "MultiLayerPerceptron" then
            net = MultiLayerPerceptron(optimizeInTorch, gpuid)
        elseif networkClass == "BidirectionalLSTM" then
            net = BidirectionalLSTM(optimizeInTorch, gpuid)
        elseif networkClass == "SimpleBiLSTM" then
            net = SimpleBiLSTM(optimizeInTorch, gpuid)
        elseif networkClass == "TagBiLSTM" then
            net = TagBiLSTM(optimizeInTorch, gpuid)
        elseif networkClass == "ContinuousFeature" then
            net = ContinuousFeature(optimizeInTorch, gpuid)
        elseif networkClass == "EmbeddingLayer" then
            net = EmbeddingLayer(optimizeInTorch, gpuid)
        else
            error("Unsupported network class " .. networkClass)
        end
    end
    local outputAndGradOutputPtr = {... }
    local paramsPtr, gradParamsPtr = net:initialize(javadata, unpack(outputAndGradOutputPtr))
    local time = timer:time().real
    print(string.format("Init took %.4fs", time))
    if paramsPtr ~= nil and gradParamsPtr ~= nil then
        return paramsPtr, gradParamsPtr
    end
end

function forward(training, batchInputIds)
    local batch
    if batchInputIds ~= nil then
        batch = torch.LongTensor(listToTable(batchInputIds))
    else
        batch = nil
    end
    local timer = torch.Timer()
    net:forward(training, batch)
    local time = timer:time().real
    ---print(string.format("Forward took %.4fs", time))
end

function backward()
    local timer = torch.Timer()
    net:backward()
    local time = timer:time().real
    ---print(string.format("Backward took %.4fs", time))
end

function save_model(prefix)
    local timer = torch.Timer()
    torch.save(prefix, net)
    local time = timer:time().real
    print(string.format("Saving model took %.4fs", time))
end

function load_model(prefix, gpuID)
    local timer = torch.Timer()
    if gpuID >= 0 then
        gpuid = gpuID
        setupGPU()
    end
    net = torch.load(prefix)
    local time = timer:time().real
    print(string.format("Loading model took %.4fs", time))
end
