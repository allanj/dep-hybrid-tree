local MultiLayerPerceptron, parent = torch.class('MultiLayerPerceptron', 'AbstractNeuralNetwork')

function MultiLayerPerceptron:__init(doOptimization, gpuid)
    parent.__init(self, doOptimization)
    self.data = {}
    self.gpuid = gpuid
end

function MultiLayerPerceptron:initialize(javadata, ...)
    local gpuid = self.gpuid

    -- numInputList, inputDimList, embSizeList, outputDim,
    -- numLayer, hiddenSize, activation, dropout
    -- vocab

    self.data = {}
    local data = self.data

    data.vocab = listToTable2D(javadata:get("vocab"))
    data.numInputList = listToTable(javadata:get("numInputList"))
    data.embedding = listToTable(javadata:get("embedding"))
    data.embSizeList = listToTable(javadata:get("embSizeList"))
    data.fixInputLayer = javadata:get("fixInputLayer")
    data.wordList = listToTable(javadata:get("wordList"))
    data.inputDimList = listToTable(javadata:get("inputDimList"))
    data.numLayer = javadata:get("numLayer")
    data.hiddenSize = javadata:get("hiddenSize")
    data.activation = javadata:get("activation")
    data.dropout = javadata:get("dropout")
    data.optimizer = javadata:get("optimizer")
    data.lang = javadata:get("lang")
    data.numLabels = javadata:get("numLabels")
    
    local isTraining = javadata:get("isTraining")
    local outputAndGradOutputPtr = {... }
    if isTraining then
        self.outputPtr = torch.pushudata(outputAndGradOutputPtr[1], "torch.DoubleTensor")
        self.gradOutputPtr = torch.pushudata(outputAndGradOutputPtr[2], "torch.DoubleTensor")
    end

    -- what to forward
    self.x = self:prepare_input()
    self.fixInputLayer = data.fixInputLayer
    self.wordList = data.wordList
    if isTraining then self.word2idx = {} end
    local wordList = self.wordList
    local word2idx = self.word2idx
    for i=1,#wordList do
        word2idx[wordList[i]] = i
    end
    self.numInput = self.x[1]:size(1)
    
    if isTraining then
        self:createNetwork()
        if data.fixInputLayer then print(self.inputLayer) end
        print(self.net)

        self.params, self.gradParams = self.net:getParameters()
        if doOptimization then
            self:createOptimizer()
            -- no return array if optim is done here
        else
            self.params:retain()
            self.paramsPtr = torch.pointer(self.params)
            self.gradParams:retain()
            self.gradParamsPtr = torch.pointer(self.gradParams)
            return self.paramsPtr, self.gradParamsPtr
        end
    else
        self:createDecoderNetwork()
    end
end

function MultiLayerPerceptron:createNetwork()
    local data = self.data
    local gpuid = self.gpuid

    
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
        if data.fixInputLayer then
            lt.accGradParameters = function() end
        end
        pt:add(nn.Sequential():add(lt):add(nn.View(self.numInput,-1)))
        totalInput = totalInput + data.numInputList[i]
    end

    local jt = nn.JoinTable(2)
    self.net = nn.Sequential()
    local mlp = self.net
    if data.fixInputLayer then
        self.inputLayer = nn.Sequential()
        self.inputLayer:add(pt)
        self.inputLayer:add(jt)
    else
        mlp:add(pt)
        mlp:add(jt)
    end
   
    -- hidden layer
    for i=1,data.numLayer do
        if data.dropout ~= nil and data.dropout > 0 then
            mlp:add(nn.Dropout(data.dropout))
        end

        local ll
        if i == 1 then
            ll = nn.Linear(totalDim, data.hiddenSize)
        else
            ll = nn.Linear(data.hiddenSize, data.hiddenSize)
        end
        mlp:add(ll)

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
            mlp:add(act)
        end
    end

    -- output layer (passed to CRF)
    local outputDim = data.numLabels
    local lastInputDim
    if data.numLayer == 0 then
        lastInputDim = totalDim
    else
        lastInputDim = data.hiddenSize
    end
    mlp:add(nn.Linear(lastInputDim, outputDim))
    --- remove the noBias if we need the bias

    if gpuid >= 0 then
        if data.fixInputLayer then self.inputLayer:cuda() end
        mlp:cuda()
    end
end

function MultiLayerPerceptron:createDecoderNetwork()
    local data = self.data

    -- Handling of unseen tokens
    if self.fixInputLayer then
        self.decoderInputLayer = self.inputLayer:clone()
    end
    self.decoderNet = self.net:clone()
    for i=1,#data.inputDimList do
        local inputDim = data.inputDimList[i]
        local nnView
        if self.fixInputLayer then
            nnView = self.decoderInputLayer:get(1):get(i):get(2)
        else
            nnView = self.decoderNet:get(1):get(i):get(2)
        end
        -- adjust the number of inputs
        nnView:resetSize(self.numInput,-1)
    end
end

function MultiLayerPerceptron:createOptimizer()
    local data = self.data

    -- set optimizer. If nil, optimization is done by caller.
    print(string.format("Optimizer: %s", data.optimizer))
    self.doOptimization = data.optimizer ~= nil and data.optimizer ~= 'none'
    if self.doOptimization == true then
        if data.optimizer == 'sgd' then
            self.optimizer = optim.sgd
            self.optimState = {learningRate=data.learningRate}
        elseif data.optimizer == 'adagrad' then
            self.optimizer = optim.adagrad
            self.optimState = {learningRate=data.learningRate}
        elseif data.optimizer == 'adam' then
            self.optimizer = optim.adam
            self.optimState = {learningRate=data.learningRate}
        elseif data.optimizer == 'adadelta' then
            self.optimizer = optim.adadelta
            self.optimState = {learningRate=data.learningRate}
        elseif data.optimizer == 'lbfgs' then
            self.optimizer = optim.lbfgs
            self.optimState = {tolFun=10e-10, tolX=10e-16}
        end
    end
end

function MultiLayerPerceptron:prepare_input()
    local gpuid = self.gpuid
    local data = self.data

    local vocab = data.vocab
    local numInputList = data.numInputList
    local embSizeList = data.embSizeList
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

function MultiLayerPerceptron:forward(isTraining)
    local mlp, inputLayer
    if isTraining then
        mlp = self.net
        inputLayer = self.inputLayer
        mlp:training()
    else
        mlp = self.decoderNet
        inputLayer = self.decoderInputLayer
        mlp:evaluate()
    end
    local x = self.x
    local input_x = x
    if self.fixInputLayer then
        input_x = inputLayer:forward(x)
    end
    local output = mlp:forward(input_x)
    if not self.outputPtr:isSameSizeAs(output) then
        self.outputPtr:resizeAs(output)
    end
    self.outputPtr:copy(output)
end

function MultiLayerPerceptron:backward()
    self.gradParams:zero()
    local x = self.x
    local input_x = x
    if self.fixInputLayer then
        input_x = self.inputLayer:forward(x)
    end
    if self.net:get(1) ~= nil then
        self.net:backward(input_x, self.gradOutputPtr)
    end
    if self.doOptimization then
        self.optimizer(self.feval, self.params, self.optimState)
    end
end

function MultiLayerPerceptron:save_model(prefix)
    local obj = {}
    obj["mlp"] = self.net
    obj["word2idx"] = self.word2idx
    torch.save("model/" .. prefix .. ".t7",obj)
end

function MultiLayerPerceptron:load_model(path)
    local saved_obj = torch.load("model/" .. prefix .. ".t7")
    local saved_mlp = saved_obj.mlp
    local saved_word2idx = saved_obj.word2idx
    local saved_lt = saved_mlp:get(1):get(1):get(1)
    local mlp = self.net
    local orig_lt = mlp:get(1):get(1):get(1)
    local wordList = self.wordList
    for i=1,#wordList do
        saved_word_idx = saved_word2idx[wordList[i]]
        if saved_word_idx ~= nil then
            orig_lt.weight[i]:copy(saved_lt.weight[saved_word_idx])
        end
    end
    saved_lt.weight = orig_lt.weight
    mlp = saved_mlp
    mlp:get(1):get(1):get(2):resetSize(self.numInput,-1)
end
