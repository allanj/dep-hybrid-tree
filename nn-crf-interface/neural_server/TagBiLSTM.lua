local TagBiLSTM, parent = torch.class('TagBiLSTM', 'AbstractNeuralNetwork')

function TagBiLSTM:__init(doOptimization, gpuid)
    parent.__init(self, doOptimization)
    self.data = {}
    self.gpuid = gpuid
end

function TagBiLSTM:initialize(javadata, ...)
    self.data = {}
    local data = self.data
    data.sentences = listToTable(javadata:get("nnInputs"))
    data.hiddenSize = javadata:get("hiddenSize")
    data.numLabels = javadata:get("numLabels")

    self.input = self:prepare_input()
    self.output = torch.Tensor()

    if self.net == nil then
        local outputAndGradOutputPtr = {... }
        self.outputPtr = torch.pushudata(outputAndGradOutputPtr[1], "torch.DoubleTensor")
        self.gradOutputPtr = torch.pushudata(outputAndGradOutputPtr[2], "torch.DoubleTensor")
        self.gradOutput = {}
        self:createNetwork()
        print(self.net)
        return self:obtainParams()
    end
    --- hiddenSize 
    -- obtain the data from java.
    -- build the vocabulary.
    -- create the neural network structure. lstm
end

function TagBiLSTM:createNetwork()
    local data = self.data
    local hiddenSize = data.hiddenSize
    --- vocabulary x hiddenSize

    local embeddingLayer = nn.LookupTableMaskZero(self.vocabSize, hiddenSize)

    local fwdLSTM = nn.LSTM(hiddenSize, hiddenSize):maskZero(1)

    local fwd = nn.Sequential():add(embeddingLayer)
                :add(fwdLSTM)
    local fwdSeq = nn.Sequencer(fwd)
    local bwd = nn.Sequential():add(embeddingLayer:sharedClone())
                :add(nn.LSTM(hiddenSize, hiddenSize):maskZero(1))
    local bwdSeq = nn.Sequential()
            :add(nn.Sequencer(bwd))
            :add(nn.ReverseTable())

    local merge = nn.JoinTable(1, 1)
    local mergeSeq = nn.Sequencer(merge)

    local parallel = nn.ParallelTable()
    parallel:add(fwdSeq)
    parallel:add(bwdSeq)
    local brnn = nn.Sequential()
       :add(parallel)
       :add(nn.ZipTable())
       :add(mergeSeq)
    local rnn = nn.Sequential()
        :add(brnn) 
        :add(nn.Sequencer(nn.MaskZero(nn.Linear(2 * hiddenSize, data.numLabels), 1))) 

    self.net = rnn
end

function TagBiLSTM:obtainParams()
    --make sure we will not replace this variable
    self.params, self.gradParams = self.net:getParameters()
    print("Number of parameters: " .. self.params:nElement())
    self.params:retain()
    self.paramsPtr = torch.pointer(self.params)
    self.gradParams:retain()
    self.gradParamsPtr = torch.pointer(self.gradParams)
    return self.paramsPtr, self.gradParamsPtr
end

function TagBiLSTM:prepare_input()
    local data = self.data
    local sentences = data.sentences
    local sentence_toks = {}
    local maxLen = 0
    for i=1,#sentences do
        local tokens = stringx.split(sentences[i]," ")
        table.insert(sentence_toks, tokens)
        if #tokens > maxLen then
            maxLen = #tokens
        end
    end

    --note that inside if the vocab is already created
    --just directly return
    self:buildVocab(sentences, sentence_toks)    

    local inputs = {}
    local inputs_rev = {}
    for step=1,maxLen do
        inputs[step] = torch.LongTensor(#sentences)
        for j=1,#sentences do
            local tokens = sentence_toks[j]
            if step > #tokens then
                inputs[step][j] = 0 --padding token
            else
                local tok = sentence_toks[j][step]
                local tok_id = self.word2idx[tok]
                if tok_id == nil then
                    tok_id = self.word2idx['<UNK>']
                end
                inputs[step][j] = tok_id
            end
        end
    end
    print("max sentencen length:"..maxLen)
    for step=1,maxLen do
        inputs_rev[step] = torch.LongTensor(#sentences)
        for j=1,#sentences do
            local tokens = sentence_toks[j]
            inputs_rev[step][j] = inputs[maxLen-step+1][j]
        end
    end
    self.maxLen = maxLen
    return {inputs, inputs_rev}
end

function TagBiLSTM:buildVocab(sentences, sentence_toks)
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
    for i=1,#sentences do
        local tokens = sentence_toks[i]
        for j=1,#tokens do
            local tok = tokens[j]
            local tok_id = self.word2idx[tok]
            if tok_id == nil then
                self.vocabSize = self.vocabSize+1
                self.word2idx[tok] = self.vocabSize
                self.idx2word[self.vocabSize] = tok
            end
        end
    end
    print("number of unique words:" .. #self.idx2word)
end

function TagBiLSTM:forward(isTraining, batchInputIds)
    local output_table = self.net:forward(self.input)
    self.output = torch.cat(self.output, output_table, 1)
    if not self.outputPtr:isSameSizeAs(self.output) then
        self.outputPtr:resizeAs(self.output)
    end
    self.outputPtr:copy(self.output)
    
end

function TagBiLSTM:backward()
    self.gradParams:zero()
    local gradOutputTensor = self.gradOutputPtr
    torch.split(self.gradOutput, gradOutputTensor, #self.data.sentences, 1)
    self.net:backward(self.input, self.gradOutput)
    -- back propagation
    -- backward
end


