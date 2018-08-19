require "nn"
include 'nn-crf-interface/neural_server/OneHot.lua'
require 'optim'

opt={}
opt.gpuid = -1

SEED = 1337
torch.manualSeed(SEED)

-- GPU setup
gpuid = opt.gpuid
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

local ret -- return value to client
local inputLayer
local mlp -- our neural net
local params, gradParams, paramsPtr, gradParamsPtr -- mlp's params
local x, fixEmbedding -- input to nn, which is fixed
local wordList, word2idx
local numInput
local outputDimList
local doOptimization, optimizer, optimState

local glove
function loadGlove(wordList, dim)
    if glove == nil then
        glove = require 'glove_torch/glove'
    end
    glove:load(dim)

    specialSymbols = {}
    specialSymbols['<PAD>'] = torch.Tensor(dim):normal(0,1)
    specialSymbols['<S>'] = torch.Tensor(dim):normal(0,1)
    specialSymbols['</S>'] = torch.Tensor(dim):normal(0,1)

    ltw = nn.LookupTable(#wordList, dim)
    for i=1,#wordList do
        local emb = torch.Tensor(dim)
        local p_emb = glove:word2vec(wordList[i])
        if p_emb == nil then
            p_emb = specialSymbols[wordList[i]]
        end
        for j=1,dim do
            emb[j] = p_emb[j]
        end
        ltw.weight[i] = emb
    end
    return ltw
end

local bansal
function loadBansal(wordList)
    if bansal == nil then
        bansal = require 'syntacticEmbeddings/bansal'
    end
    bansal:load()
    ltw = nn.LookupTable(#wordList, 100)
    for i=1,#wordList do
        local emb = torch.Tensor(100)
        local p_emb = bansal:word2vec(wordList[i])
        if p_emb == nil then
            p_emb = bansal:word2vec('*UNKNOWN*')
        end
        for j=1,100 do
            emb[j] = p_emb[j]
        end
        ltw.weight[i] = emb
    end

    return ltw
end

local senna
function loadSenna(lt)
    -- http://www-personal.umich.edu/~rahuljha/files/nlp_from_scratch/ner_embeddings.lua
    ltw = nn.LookupTable(130000, 50)

    -- initialize lookup table with embeddings
    embeddingsFile = torch.DiskFile('./senna/embeddings.txt');
    embedding = torch.DoubleStorage(50)

    embeddingsFile:readDouble(embedding);
    for i=2,130000 do 
       embeddingsFile:readDouble(embedding);
       local emb = torch.Tensor(50)
       for j=1,50 do 
          emb[j] = embedding[j]
       end
       ltw.weight[i-1] = emb;
    end
    return ltw
    -- misc note: PADDING index is 1738
end

local polyglot
function loadPolyglot(wordList, lang)
    if polyglot == nil then
        polyglot = require 'polyglot/polyglot'
    end
    polyglot:load(lang)
    ltw = nn.LookupTable(#wordList, 64)
    for i=1,#wordList do
        local emb = torch.Tensor(64)
        local p_emb = polyglot:word2vec(wordList[i])
        for j=1,64 do
            emb[j] = p_emb[j]
        end
        ltw.weight[i] = emb
    end
    return ltw
end

function listToTable(list)
    local res = {}
    for i = 1, list:size() do
        table.insert(res, list:get(i-1))
    end
    return res
end

function listToTable2D(list)
    local res = {}
    for i = 1, list:size() do
        table.insert(res, listToTable(list:get(i-1)))
    end
    return res
end

local gradOutput
local outputPtr = {}

function init_MLP(javadata, ...)

    local outputAndGradOutputPtr = {... }

    outputPtr = {}
    gradOutput = {}
    for i=1,#outputAndGradOutputPtr do
        local ptr = outputAndGradOutputPtr[i]
        if i <= #outputAndGradOutputPtr/2 then
            table.insert(outputPtr, torch.pushudata(ptr, "torch.DoubleTensor"))
        else    
            table.insert(gradOutput, torch.pushudata(ptr, "torch.DoubleTensor"))
        end
    end
    
    -- for i=1,#gradOutputPtr do
    --     local ptr = gradOutputPtr[i]
    --     table.insert(gradOutput, torch.pushudata(ptr, "torch.DoubleTensor"))
    -- end

    -- numInputList, inputDimList, embSizeList, outputDim,
    -- numLayer, hiddenSize, activation, dropout
    -- vocab

    timer = torch.Timer()

    -- re-seed
    torch.manualSeed(SEED)
    if gpuid >= 0 then cutorch.manualSeed(SEED) end

    local data = {}
    data.vocab = listToTable2D(javadata:get("vocab"))
    data.numInputList = listToTable(javadata:get("numInputList"))
    data.embSizeList = listToTable(javadata:get("embSizeList"))
    data.fixEmbedding = javadata:get("fixEmbedding")
    data.wordList = listToTable(javadata:get("wordList"))
    data.inputDimList = listToTable(javadata:get("inputDimList"))
    data.outputDimList = listToTable(javadata:get("outputDimList"))
    data.numLayer = javadata:get("numLayer")
    data.hiddenSize = javadata:get("hiddenSize")
    data.activation = javadata:get("activation")
    data.dropout = javadata:get("dropout")
    data.optimizer = javadata:get("optimizer")

    -- what to forward
    x = prepare_input(data.vocab, data.numInputList, data.embSizeList)
    fixEmbedding = data.fixEmbedding
    wordList = data.wordList
    word2idx = {}
    for i=1,#wordList do
        word2idx[wordList[i]] = i
    end
    numInput = x[1]:size(1)

    -- input layer
    local pt = nn.ParallelTable()
    local totalInput = 0
    local totalDim = 0
    for i=1,#data.inputDimList do
        local inputDim = data.inputDimList[i]
        local lt
        if data.embSizeList[i] == 0 then
            lt = OneHot(inputDim)
            totalDim = totalDim + data.numInputList[i] * inputDim
        else
            if data.embedding ~= nil then
                if data.embedding[i] == 'senna' then
                    lt = loadSenna()
                elseif data.embedding[i] == 'glove' then
                    lt = loadGlove(data.wordList, data.embSizeList[i])
                elseif data.embedding[i] == 'polyglot' then
                    lt = loadPolyglot(data.wordList, data.lang)
                elseif data.embedding[i] == 'bansal' then
                    lt = loadBansal(data.wordList)
                else -- unknown/no embedding, defaults to random init
                    lt = nn.LookupTable(inputDim, data.embSizeList[i])
                end
            else
                lt = nn.LookupTable(inputDim, data.embSizeList[i])
            end
            totalDim = totalDim + data.numInputList[i] * data.embSizeList[i]
        end
        if data.fixEmbedding then
            lt.accGradParameters = function() end
        end
        pt:add(nn.Sequential():add(lt):add(nn.View(numInput,-1)))
        totalInput = totalInput + data.numInputList[i]
    end

    local jt = nn.JoinTable(2)

    mlp = nn.Sequential()
    if data.fixEmbedding then
        inputLayer = nn.Sequential()
        inputLayer:add(pt)
        inputLayer:add(jt)
    else
        mlp:add(pt)
        mlp:add(jt)
    end
   
    outputDimList = data.outputDimList
    ct = nn.ConcatTable()
    numNetworks = #outputDimList
    for n=1,numNetworks do
        middleLayers = nn.Sequential()
        -- hidden layer
        for i=1,data.numLayer do
            if data.dropout ~= nil and data.dropout > 0 then
                middleLayers:add(nn.Dropout(data.dropout))
            end

            local ll
            if i == 1 then
                ll = nn.Linear(totalDim, data.hiddenSize)
            else
                ll = nn.Linear(data.hiddenSize, data.hiddenSize)
            end
            middleLayers:add(ll)

            local act
            if data.activation == nil or data.activation == "relu" then
                act = nn.ReLU()
            elseif data.activation == "tanh" then
                act = nn.Tanh()
            elseif data.activation == "hardtanh" then
                act = nn.HardTanh()
            elseif data.activation == "identity" then
                -- do nothing
            else
                error("activation " .. activation .. " not supported")
            end
            if act ~= nil then
                middleLayers:add(act)
            end
        end

        if data.dropout ~= nil and data.dropout > 0 then
            middleLayers:add(nn.Dropout(data.dropout))
        end

        -- output layer (passed to CRF)
        local outputDim = data.outputDimList[n]
        local lastInputDim
        if data.numLayer == 0 then
            lastInputDim = totalDim
        else
            lastInputDim = data.hiddenSize
        end
        if data.useOutputBias then
            middleLayers:add(nn.Linear(lastInputDim, outputDim))
        else
            middleLayers:add(nn.Linear(lastInputDim, outputDim):noBias())
        end
        ct:add(middleLayers)
    end
    mlp:add(ct)

    if gpuid >= 0 then
        if data.fixEmbedding then inputLayer:cuda() end
        mlp:cuda()
    end

    -- set optimizer. If nil, optimization is done by caller.
    print(string.format("Optimizer: %s", data.optimizer))
    doOptimization = data.optimizer ~= nil and data.optimizer ~= 'none'
    if doOptimization == true then
        if data.optimizer == 'sgd' then
            optimizer = optim.sgd
            optimState = {learningRate=data.learningRate}
        elseif data.optimizer == 'adagrad' then
            optimizer = optim.adagrad
            optimState = {learningRate=data.learningRate}
        elseif data.optimizer == 'adam' then
            optimizer = optim.adam
            optimState = {learningRate=data.learningRate}
        elseif data.optimizer == 'adadelta' then
            optimizer = optim.adadelta
            optimState = {learningRate=data.learningRate}
        elseif data.optimizer == 'lbfgs' then
            optimizer = optim.lbfgs
            optimState = {tolFun=10e-10, tolX=10e-16}
        end
    end

    params, gradParams = mlp:getParameters()

    if data.fixEmbedding then print(inputLayer) end
    print(mlp)

    local time = timer:time().real
    print(string.format("Init took %.4fs", time))

    if not doOptimization then -- no return array if optim is done here
        params:retain()
        paramsPtr = torch.pointer(params)
        gradParams:retain()
        gradParamsPtr = torch.pointer(gradParams)
        return paramsPtr, gradParamsPtr    
    end
end

function fwd_MLP(training)
    local timer = torch.Timer()
    if training == true then
        mlp:training()
    else
        mlp:evaluate()
    end

    local input_x = x
    
    if fixEmbedding then
        input_x = inputLayer:forward(x)
    end

    local output = mlp:forward(input_x)

    -- local outputPtr = {}
    for i=1,#output do
        outputPtr[i]:copy(output[i])
        -- output[i]:retain()
        -- table.insert(outputPtr, torch.pointer(output[i]))
    end

    local time = timer:time().real
    print(string.format("Forward took %.4fs", time))
    -- return unpack(outputPtr)
end

function feval(params) -- for optim
    return 0, gradParams
end

function bwd_MLP()
    -- local gradOutputPtr = {... }
    -- local gradOutput = {}
    
    -- for i=1,#gradOutputPtr do
    --     local ptr = gradOutputPtr[i]
    --     table.insert(gradOutput, torch.pushudata(ptr, "torch.DoubleTensor"))
    -- end

    local timer = torch.Timer()
    gradParams:zero()

    local input_x = x

    if fixEmbedding then
        input_x = inputLayer:forward(x)
    end

    mlp:backward(input_x, gradOutput)
    
    if doOptimization then
        optimizer(feval, params, optimState)
    end

    local time = timer:time().real
    print(string.format("Backward took %.4fs", time))
end

function save_model(prefix)
    local timer = torch.Timer()
    local obj = {}
    obj["mlp"] = mlp
    obj["word2idx"] = word2idx
    torch.save("model/" .. prefix .. ".t7",obj)
    -- torch.save("model/" .. prefix .. ".mlp.t7",mlp)
    -- torch.save("model/" .. prefix .. ".vocab.t7",word2idx)

    local time = timer:time().real
    print(string.format("Saving model took %.4fs", time))
end

function load_model(prefix)
    local timer = torch.Timer()
    local saved_obj = torch.load("model/" .. prefix .. ".t7")
    local saved_mlp = saved_obj.mlp
    local saved_word2idx = saved_obj.word2idx
    local saved_lt = saved_mlp:get(1):get(1):get(1)
    local orig_lt = mlp:get(1):get(1):get(1)
    for i=1,#wordList do
        saved_word_idx = saved_word2idx[wordList[i]]
        if saved_word_idx ~= nil then
            orig_lt.weight[i]:copy(saved_lt.weight[saved_word_idx])
        end
    end
    saved_lt.weight = orig_lt.weight
    mlp = saved_mlp
    mlp:get(1):get(1):get(2):resetSize(numInput,-1)
    
    local time = timer:time().real
    print(string.format("Loading model took %.4fs", time))
end

function prepare_input(vocab, numInputList, embSizeList)
    local result = {}
    local startIdx = 0
    for i=1,#numInputList do
        table.insert(result, torch.Tensor(#vocab, numInputList[i]))
        for j=1,#vocab do
            for k=1,numInputList[i] do
                result[i][j][k] = vocab[j][startIdx+k]
            end
        end
        startIdx = startIdx + numInputList[i]
        if gpuid >= 0 then result[i] = result[i]:cuda() end
    end
    return result
end

-- while true do
--     --  Wait for next request from client
--     local request = socket:recv()
--     -- print("Received Hello [" .. request .. "]")
--     -- print(request)
--     if request ~= nil then
--         request = mp.unpack(request)
--         if request.cmd == "init" then
--             timer = torch.Timer()
--             init_MLP(request)
--             if request.fixEmbedding then print(inputLayer) end
--             print(mlp)
--             if doOptimization then -- no return array if optim is done here
--                 ret = 1
--             else
--                 ret = serialize(params)
--             end
--             time = timer:time().real
--             print(string.format("Init took %.4fs", time))
--         elseif request.cmd == "fwd" then
--             local timer = torch.Timer()
--             local newParams
--             if not doOptimization then
--                 newParams = deserialize(request.weights, 1, -1)
--             end
--             local fwd_out = fwd_MLP(mlp, x, newParams, request.training)
--             ret = serialize2(fwd_out)
--             time = timer:time().real
--             print(string.format("Forward took %.4fs", time))
--         elseif request.cmd == "bwd" then
--             local timer = torch.Timer()
--             local gradOut = deserialize2(request.grad, numInput, outputDimList)
--             bwd_MLP(mlp, x, gradOut)
--             if doOptimization then
--                 ret = 1
--             else
--                 ret = serialize(gradParams)
--             end
--             time = timer:time().real
--             print(string.format("Backward took %.4fs", time))
--         elseif request.cmd == "save" then
--             local timer = torch.Timer()
--             save_model(request.savePrefix)
--             time = timer:time().real
--             print(string.format("Saving model took %.4fs", time))
--             ret = 1
--         elseif request.cmd == "load" then
--             local timer = torch.Timer()
--             load_model(request.savePrefix)
--             time = timer:time().real
--             print(string.format("Loading model took %.4fs", time))
--             ret = 1
--         end
--         ret = mp.pack(ret)
--         socket:send(ret)
--     end
-- end
