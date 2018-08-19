local MultiLayerPerceptron, parent = torch.class('MultiLayerPerceptron', 'AbstractNeuralNetwork')

function MultiLayerPerceptron:__init(doOptimization, gpuid)
    parent.__init(self, doOptimization)
    self.data = {}
    self.gpuid = gpuid
end

function MultiLayerPerceptron:defineGlobalString()
    self.unkToken = "<UNK>"
end

function MultiLayerPerceptron:loadEmbObj(lang)
    local data = self.data
    self.embeddingSize = data.embeddingSize
    if data.embedding == 'polyglot' then
        self.embeddingObject = loadPolyglotEmbObj(lang)
        self.embeddingSize = 64
    elseif data.embedding == 'random' then 
        print("using random embedding")
    else
        error('unknown embedding type: '.. data.embedding)
    end
end

function MultiLayerPerceptron:initialize(javadata, ...)
    local gpuid = self.gpuid
    self.data = {}
    local data = self.data
    data.numLabels = javadata:get("numLabels")
    data.rawInputs = listToTable(javadata:get("nnInputs"))
    local isTraining = javadata:get("isTraining")
    local modelPath = javadata:get("nnModelFile")
    self.hiddenSize = javadata:get("hiddenSize")
    self.embeddingSize = 100
    data.isTraining = isTraining
    data.embedding = javadata:get("embedding")
    self.fixEmbedding = javadata:get("fixEmbedding")
    local lang = javadata:get("lang")
    self.type = javadata:get("type")
    self.numWordsInput =  2 * javadata:get("windowSize") + 1
    self:defineGlobalString()
    if isTraining then
        self:loadEmbObj(lang)
        self.x = self:prepare_input(isTraining)
        self:buildLookupTable()
        if self.fixEmbedding then self.x = self.lt:forward(self.x):clone() end
    end

    if self.net == nil and isTraining then
        -- means is initialized process and we don't have the input yet.
        self:createNetwork()
        print(self.net)
    end

    if self.net == nil then 
        self:load_model(modelPath)
    end

    if not isTraining then 
        self.testInput = self:prepare_input(isTraining)
        if self.fixEmbedding then self.testInput = self.lt:forward(self.testInput):clone() end
    end

    self.output = torch.Tensor()
    self.gradOutput = torch.Tensor()
    local outputAndGradOutputPtr = {... }
    if #outputAndGradOutputPtr > 0 then
        self.outputPtr = torch.pushudata(outputAndGradOutputPtr[1], "torch.DoubleTensor")
        self.gradOutputPtr = torch.pushudata(outputAndGradOutputPtr[2], "torch.DoubleTensor")
        return self:obtainParams()
    end
end

function MultiLayerPerceptron:buildLookupTable()
    local lt = nn.LookupTable(self.vocabSize, self.embeddingSize)
    if self.data.embedding ~= 'random' then
        print("copying polyglot embedding")
        for i =1, self.vocabSize do
            lt.weight[i]:copy(self.embeddingObject:word2vec(self.idx2word[i]))
        end
    end
    self.lt = lt
    if self.fixEmbedding then
        self.lt.accGradParameters = function() end
        self.lt.parameters = function() end
        if self.gpuid >= 0 then self.lt:cuda() end
    end
end

function MultiLayerPerceptron:createNetwork()
    local data = self.data
    local gpuid = self.gpuid
    self.numLabels = data.numLabels
    local mlp = nn.Sequential()
    local hiddenSize = self.hiddenSize
    
    if not self.fixEmbedding then
        mlp:add(self.lt)
    end
    if self.type == "mlp" then
        local view = nn.View(-1):setNumInputDims(2)
        mlp:add(view)
        mlp:add(nn.Linear(2 * self.embeddingSize, hiddenSize))
        mlp:add(nn.Tanh())
        mlp:add(nn.Linear(hiddenSize, self.numLabels))
    elseif self.type == "bilinear" then
        mlp:add(nn.SplitTable(2))
        mlp:add(nn.Bilinear(self.embeddingSize, self.embeddingSize, self.numLabels, false))
    elseif self.type == "cnn" then
        mlp:add(nn.TemporalConvolution(self.numWordsInput * self.embeddingSize, hiddenSize, 3, 1))
        mlp:add(nn.Tanh())
        mlp:add(nn.Max(2))
        mlp:add(nn.Linear(hiddenSize, self.numLabels))
    elseif self.type == "bilinear-mlp" then
        mlp:add(nn.SplitTable(2))
        mlp:add(nn.Bilinear(self.embeddingSize, self.embeddingSize, hiddenSize, false))
        mlp:add(nn.Tanh())
        mlp:add(nn.Linear(hiddenSize, self.numLabels))
    else
        error("invalid type: ".. self.type)
    end
    if gpuid >= 0 then
        mlp:cuda()
    end
    self.net = mlp
end

function MultiLayerPerceptron:obtainParams()
    --make sure we will not replace this variable
    self.params, self.gradParams = self.net:getParameters()
    print("Number of parameters: " .. self.params:nElement())
    if self.doOptimization then
        self:createOptimizer()
        -- no return array if optim is done here
    else
        if self.gpuid >= 0 then
            -- since the the network is gpu network.
            self.paramsDouble = self.params:double()
            self.paramsDouble:retain()
            self.params:retain()
            self.paramsPtr = torch.pointer(self.paramsDouble)
            self.gradParamsDouble = self.gradParams:double()
            self.gradParamsDouble:retain()
            self.gradParams:retain()
            self.gradParamsPtr = torch.pointer(self.gradParamsDouble)
            return self.paramsPtr, self.gradParamsPtr
        else
            self.params:retain()
            self.paramsPtr = torch.pointer(self.params)
            self.gradParams:retain()
            self.gradParamsPtr = torch.pointer(self.gradParams)
            return self.paramsPtr, self.gradParamsPtr
        end
    end
end


function MultiLayerPerceptron:forward(isTraining, batchInputIds)
    if self.gpuid >= 0 and not self.doOptimization and isTraining then
        self.params:copy(self.paramsDouble:cuda())
    end
    if isTraining then
        self.net:training()
    else
        self.net:evaluate()
    end
    local input_x = self:getForwardInput(isTraining, batchInputIds)
    local output
    if isTraining then
        output = self.net:forward(input_x)
    else
        -- lstmOutput = self.net:forward(nnInput)
        output = torch.Tensor()
        if self.gpuid >=0 then output = output:cuda() end
        local instSize = input_x:size(1) --number of sentences 
        local testBatchSize = 10   ---test batch size = 10
        for i = 1, instSize, testBatchSize do
            if i + testBatchSize - 1 > instSize then testBatchSize =  instSize - i + 1 end
            local tmpOut = self.net:forward(input_x:narrow(1, i, testBatchSize))
            output = torch.cat(output, tmpOut, 1)
        end
    end
    if self.gpuid >= 0 then
        output = output:double()
    end
    if not self.outputPtr:isSameSizeAs(output) then
        self.outputPtr:resizeAs(output)
    end
    self.outputPtr:copy(output)
end

function MultiLayerPerceptron:getForwardInput(isTraining, batchInputIds)
    local input = nil
    if isTraining then
        if batchInputIds ~= nil then
            batchInputIds:add(1) -- because the sentence is 0 indexed\
            self.batchInput = self.x:index(1, batchInputIds)
            input = self.batchInput
        else
            input = self.x
        end
    else
        input = self.testInput
    end
    return input
end

function MultiLayerPerceptron:getBackwardInput()
    local input = nil
    if self.batchInput ~= nil then
        input =  self.batchInput
    else
        input =  self.x
    end
    return input
end

function MultiLayerPerceptron:backward()
    self.gradParams:zero()
    local gradOutputTensor = self.gradOutputPtr
    if self.gpuid >= 0 then
        gradOutputTensor = gradOutputTensor:cuda()
    end
    local input = self:getBackwardInput()
    self.net:backward(input, gradOutputTensor)
    if self.doOptimization then
        self.optimizer(self.feval, self.params, self.optimState)
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
    local rawInputs = data.rawInputs
    local word_toks = stringx.split(rawInputs[1], " ")
    local num_words = #word_toks
    local all_word_toks = {}
    print("number of word pairs: "..#rawInputs)
    for i=1, #rawInputs do
        local words = rawInputs[i]
        word_toks = stringx.split(words, " ")
        -- print(word_toks)
        table.insert(all_word_toks, word_toks)
    end


    local results = torch.IntTensor(#rawInputs, num_words)

    self:buildVocab(all_word_toks)
    for i=1,#rawInputs do
        local words = all_word_toks[i]
        for j=1,#words do
            local tok = words[j]
            local tok_id = self.word2idx[tok]
            if tok_id == nil then
                tok_id = self.word2idx[self.unkToken]  
            end
            results[i][j] = tok_id
        end
    end
    if gpuid >= 0 then results = results:cuda() end
    print("number of word pairs: "..#rawInputs)
    print("number of positions: "..num_words)
    -- print(results)
    return results
end

function MultiLayerPerceptron:buildVocab(wordInputs)
    if self.word2idx ~= nil then
        --means the vocabulary is already created
        return 
    end
    local embeddingObject = self.embeddingObject or nil
    local embW2V = nil
    if embeddingObject ~= nil then
        embW2V = embeddingObject.w2vvocab
    end

    self.word2idx = {}
    self.idx2word = {}
    self.vocabSize = 1
    self.word2idx[self.unkToken] = self.vocabSize
    self.idx2word[self.vocabSize] = self.unkToken
    self.unkTokens = {}
    self:buildVocabForTokens(wordInputs, embW2V)
    print("number of unique words (including unknown):".. self.vocabSize.." (unknown words are replaced by unk)")
    print("number of unknown tokens (not unique): ".. countTable(self.unkTokens))
end

function MultiLayerPerceptron:buildVocabForTokens(wordInputs, embW2V)
    for i=1,#wordInputs do
        local words = wordInputs[i]
        -- print(words)
        for j=1,#words do
            local tok = words[j]
            local tokInEmbId = nil 
            if embW2V ~= nil then tokInEmbId = embW2V[tok] end
            if embW2V ~= nil and tokInEmbId == nil and self.word2idx[tok] == nil then
                ---not in the pretrained embedding, just use unk
                self.word2idx[tok] = self.word2idx[self.unkToken]
                if self.unkTokens[tok] == nil then
                    self.unkTokens[tok] = 1 --dummy value
                end
            else
                --in the pretraining embedding table
                local tok_id = self.word2idx[tok]
                if tok_id == nil then 
                    self.vocabSize = self.vocabSize + 1
                    self.word2idx[tok] = self.vocabSize
                    self.idx2word[self.vocabSize] = tok
                end
            end
        end
    end
end

function printTable(table)
    local count = 0
    for i,k in pairs(table) do print(i .. " " .. k) end
    return count
end

function countTable(table)
    local count = 0
    for _ in pairs(table) do count = count + 1 end
    return count
end
