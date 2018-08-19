local EmbeddingLayer, parent = torch.class('EmbeddingLayer', 'AbstractNeuralNetwork')

function EmbeddingLayer:__init(doOptimization, gpuid)
    parent.__init(self, doOptimization)
    self.data = {}
    self.gpuid = gpuid
end

function EmbeddingLayer:initialize(javadata, ...)
    self.data = {}
    local data = self.data
    data.words = listToTable(javadata:get("nnInputs"))
    self.numLabels = javadata:get("numLabels")
    data.hiddenSize = javadata:get("hiddenSize")
    data.embedding = javadata:get("embedding")
    self.fixEmbedding = javadata:get("fixEmbedding")
    local modelPath = javadata:get("nnModelFile")
    local isTraining = javadata:get("isTraining")
    data.isTraining = isTraining

    if isTraining then
        self.x = self:prepare_input()
    end

    if self.net == nil and isTraining then
        -- means is initialized process and we don't have the input yet.
        self:createNetwork()
        if self.fixEmbedding then
            print(self.lookupTable)
        end
        print(self.net)
    end
    print("[Warning] EmbeddingLayer do not support evaluate on development set and continue training yet. Simply ignore this if you are not doing this.")

    self.output = torch.Tensor()
    local outputAndGradOutputPtr = {... }
    if #outputAndGradOutputPtr > 0 then
        self.outputPtr = torch.pushudata(outputAndGradOutputPtr[1], "torch.DoubleTensor")
        self.gradOutputPtr = torch.pushudata(outputAndGradOutputPtr[2], "torch.DoubleTensor")
        return self:obtainParams()
    end
end

--The network is only created once is used.
function EmbeddingLayer:createNetwork()
    local data = self.data
    local hiddenSize = data.hiddenSize
    local sharedLookupTable
    if data.embedding ~= nil then
        if data.embedding == 'glove' then
            sharedLookupTable = loadGlove(self.idx2word, hiddenSize, false)
        elseif data.embedding == 'google' then
            sharedLookupTable = loadGoogle(self.idx2word, hiddenSize, false)
        else -- unknown/no embedding, defaults to random init
            print ("unknown embedding type, use random embedding..")
            sharedLookupTable = nn.LookupTable(self.vocabSize, hiddenSize)
            print("lookup table parameter: ".. sharedLookupTable:getParameters():nElement())
        end
    else
        print ("Not using any embedding, just use random embedding")
        sharedLookupTable = nn.LookupTable(self.vocabSize, hiddenSize)
    end
    if self.fixEmbedding then 
        sharedLookupTable.accGradParameters = function() end
    end

    self.lookupTable = sharedLookupTable

    local embeddingNN = nn.Sequential()
    if not self.fixEmbedding then
        embeddingNN:add(sharedLookupTable)
    end
    embeddingNN:add(nn.Linear(hiddenSize, self.numLabels))
    self.net = embeddingNN
end

function EmbeddingLayer:obtainParams()
    --make sure we will not replace this variable
    self.params, self.gradParams = self.net:getParameters()
    print("Number of parameters: " .. self.params:nElement())
    self.params:retain()
    self.paramsPtr = torch.pointer(self.params)
    self.gradParams:retain()
    self.gradParamsPtr = torch.pointer(self.gradParams)
    return self.paramsPtr, self.gradParamsPtr
end

function EmbeddingLayer:forward(isTraining, batchInputIds)
    local nnInput_x = self.x
    if self.fixEmbedding then
        nnInput_x =  self.lookupTable:forward(self.x)
    end
    self.output = self.net:forward(nnInput_x)
    if not self.outputPtr:isSameSizeAs(self.output) then
        self.outputPtr:resizeAs(self.output)
    end
    self.outputPtr:copy(self.output)
end

function EmbeddingLayer:backward()
    self.gradParams:zero()
    local nnInput_x = self.x
    if self.fixEmbedding then
        nnInput_x =  self.lookupTable:forward(self.x)
    end
    self.gradOutput = self.gradOutputPtr
    self.net:backward(nnInput_x, self.gradOutput)
end

function EmbeddingLayer:prepare_input()
    local data = self.data
    local words = data.words
    self:buildVocab(words)    
    local inputs = torch.LongTensor(#words)
    for i=1,#words do
        inputs[i] = self.word2idx[words[i]]
    end
    return inputs
end

function EmbeddingLayer:buildVocab(words)
    if self.idx2word ~= nil then
        --means the vocabulary is already created
        return 
    end
    self.idx2word = {}
    self.word2idx = {}
    self.word2idx['<PAD>'] = 0
    self.idx2word[0] = '<PAD>'
    self.word2idx['<UNK>'] = 1
    self.idx2word[1] = '<UNK>'
    self.vocabSize = 2
    for i=1, #words do
        local tok = words[i]
        local tok_id = self.word2idx[tok]
        if tok_id == nil then
            self.vocabSize = self.vocabSize+1
            self.word2idx[tok] = self.vocabSize
            self.idx2word[self.vocabSize] = tok
        end
    end
    print("number of unique words:" .. #self.idx2word)
end

function EmbeddingLayer:save_model(path)
    --need to save the vocabulary as well.
    torch.save(path, {self.net, self.idx2word, self.word2idx})
end

function EmbeddingLayer:load_model(path)
    local object = torch.load(path)
    self.net = object[1]
    self.idx2word = object[2]
    self.word2idx = object[3]
end
