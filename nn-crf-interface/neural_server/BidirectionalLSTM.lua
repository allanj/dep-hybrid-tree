local BidirectionalLSTM, parent = torch.class('BidirectionalLSTM', 'AbstractNeuralNetwork')

function BidirectionalLSTM:__init(doOptimization, gpuid)
    parent.__init(self, doOptimization)
    self.data = {}
    self.gpuid = gpuid
end

function BidirectionalLSTM:initialize(javadata, ...)
    self.data = {}
    local data = self.data
    data.sentences = listToTable(javadata:get("sentences"))
    data.hiddenSize = javadata:get("hiddenSize")
    data.optimizer = javadata:get("optimizer")
    self.bidirection = javadata:get("bidirection")
    self.numLabels = javadata:get("numLabels")
    data.embedding = javadata:get("embedding")
    self.isTraining = javadata:get("isTraining")
    self.x = self:prepare_input()
    self.numSent = #data.sentences
    self.output = torch.Tensor()
    self.gradOutput = {}

    if self.isTraining then
        local outputAndGradOutputPtr = {... }
        self.outputPtr = torch.pushudata(outputAndGradOutputPtr[1], "torch.DoubleTensor")
        self.gradOutputPtr = torch.pushudata(outputAndGradOutputPtr[2], "torch.DoubleTensor")
        ---the following 4 is for batch training.
        self.batchInput = {}
        self.x1Tab = {}
        self.x1 = torch.LongTensor()
        self.x2Tab = {}
        self.x2 = torch.LongTensor()
        self.gradBatchOutput = torch.Tensor()
        if self.gpuid >= 0 then 
            self.x1 = self.x1:cuda()
            self.x2 = self.x2:cuda()
            self.gradBatchOutput = self.gradBatchOutput:cuda()
        end
        self:createNetwork()
        print(self.net)
        return self:obtainParams()
    end
end

function BidirectionalLSTM:createNetwork()
    local data = self.data

    local hiddenSize = data.hiddenSize

    local sharedLookupTable
    if data.embedding ~= nil then
        if data.embedding == 'glove' then
            sharedLookupTable = loadGlove(self.idx2word, hiddenSize, true)
        else -- unknown/no embedding, defaults to random init
            print ("Not using any embedding..")
            sharedLookupTable = nn.LookupTableMaskZero(self.vocabSize, hiddenSize)
        end
    else
        print ("Not using any embedding..")
        sharedLookupTable = nn.LookupTableMaskZero(self.vocabSize, hiddenSize)
    end

    -- forward rnn
    local fwd = nn.Sequential()
       :add(sharedLookupTable)
       :add(nn.FastLSTM(hiddenSize, hiddenSize):maskZero(1))

    -- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
    local fwdSeq = nn.Sequencer(fwd)

    -- backward rnn (will be applied in reverse order of input sequence)
    local bwd, bwdSeq
    if self.bidirection then
        bwd = nn.Sequential()
           :add(sharedLookupTable:sharedClone())
           :add(nn.FastLSTM(hiddenSize, hiddenSize):maskZero(1))
           
        bwdSeq = nn.Sequential()
            :add(nn.Sequencer(bwd))
            :add(nn.ReverseTable())
    end

    -- merges the output of one time-step of fwd and bwd rnns.
    -- You could also try nn.AddTable(), nn.Identity(), etc.
    local merge = nn.JoinTable(1, 1)
    local mergeSeq = nn.Sequencer(merge)

    -- Assume that two input sequences are given (original and reverse, both are right-padded).
    -- Instead of ConcatTable, we use ParallelTable here.
    local parallel = nn.ParallelTable()
    parallel:add(fwdSeq)
    
    if self.bidirection then
        parallel:add(bwdSeq)
    end
    local brnn = nn.Sequential()
       :add(parallel)
       :add(nn.ZipTable())
       :add(mergeSeq)
    local mergeHiddenSize = hiddenSize
    if self.bidirection then
        mergeHiddenSize = 2 * hiddenSize
    end
    local rnn = nn.Sequential()
        :add(brnn) 
        :add(nn.Sequencer(nn.MaskZero(nn.Linear(mergeHiddenSize, self.numLabels), 1))) 
       --- if don't use bias, use LinearNoBias or call :noBias()
    if self.gpuid >=0 then rnn:cuda() end
    self.net = rnn
end

function BidirectionalLSTM:obtainParams()
    --make sure we will not replace this variable
    self.params, self.gradParams = self.net:getParameters()
    print("Number of parameters: " .. self.params:nElement())
    if self.doOptimization then
        self:createOptimizer()
        -- no return array if optim is done here
    else
        if self.gpuid >=0 then
            self.paramsDouble = self.params:double()
            self.paramsDouble:retain()
            self.paramsPtr = torch.pointer(self.paramsDouble)
            self.gradParamsDouble = self.gradParams:double()
            self.gradParamsDouble:retain()
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

function BidirectionalLSTM:createOptimizer()
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

function BidirectionalLSTM:forward(isTraining, batchInputIds)
    if self.gpuid >= 0 and not self.doOptimization then
        --paramsDouble point to java and it's double tensor
        --need to convert back to cudaTensor if using gpu
        self.params:copy(self.paramsDouble:cuda())
    end
    local nnInput = self:getForwardInput(batchInputIds)
    local output_table = self.net:forward(nnInput)
    if self.gpuid >= 0 then
        --convert the cuda tensor back to double tensor for java to read
        --th4j only support double tensor / float tensor
        nn.utils.recursiveType(output_table, 'torch.DoubleTensor')
    end
    --- this is to converting the table into tensor.
    self.output = torch.cat(self.output, output_table, 1)
    if self.batchInputIds ~= nil then 
        self.outputPtr:resize(self.maxLen, self.numSent, self.numLabels)
        self.output:resize(self.maxLen, self.batchInputIds:size(1), self.numLabels)
        self.outputPtr:indexCopy(2, self.batchInputIds, self.output)
        self.outputPtr:resize(self.maxLen * self.numSent, self.numLabels)
    else
        if not self.outputPtr:isSameSizeAs(self.output) then
            self.outputPtr:resizeAs(self.output)
        end
        self.outputPtr:copy(self.output)
    end
end

function BidirectionalLSTM:getForwardInput(batchInputIds)
    self.batchInputIds = batchInputIds
    if batchInputIds ~= nil then 
        self.batchInputIds:add(1) -- because the sentence is 0 indexed.
        --select the specific indexes from the table
        self.x1 = torch.cat(self.x1, self.x[1], 2):index(1, batchInputIds)
        self.x1:resize(self.x1:size(1)*self.x1:size(2))
        torch.split(self.x1Tab, self.x1, batchInputIds:size(1), 1)
        self.x2 = torch.cat(self.x2, self.x[2], 2):index(1, batchInputIds)
        self.x2:resize(self.x2:size(1)*self.x2:size(2))
        torch.split(self.x2Tab, self.x2, batchInputIds:size(1), 1)
        self.batchInput = {self.x1Tab, self.x2Tab}
        return self.batchInput
    else
        return self.x
    end
end

function BidirectionalLSTM:backward()
    self.gradParams:zero()
    local gradOutputTensor
    local backwardInput
    local backwardSentNum
    if self.batchInputIds ~= nil then
        if self.gradBatchOutput == nil then
            self.gradBatchOutput =  torch.Tensor() 
        end
        self.gradOutputPtr:resize(self.maxLen, self.numSent, self.numLabels)
        self.gradBatchOutput:resize(self.maxLen, self.batchInputIds:size(1), self.numLabels)
        self.gradBatchOutput:copy(self.gradOutputPtr:index(2, self.batchInputIds))
        self.gradBatchOutput:resize(self.maxLen * self.batchInputIds:size(1), self.numLabels)
        self.gradOutputPtr:resize(self.maxLen * self.numSent, self.numLabels)
        gradOutputTensor = self.gradBatchOutput
        backwardInput = self.batchInput
        backwardSentNum = self.batchInputIds:size(1)
    else
        gradOutputTensor = self.gradOutputPtr
        backwardInput = self.x
        backwardSentNum = self.numSent
    end
    torch.split(self.gradOutput, gradOutputTensor, backwardSentNum, 1)
    if self.gpuid >= 0 then
        nn.utils.recursiveType(self.gradOutput, 'torch.CudaTensor')
    end
    self.net:backward(backwardInput, self.gradOutput)
    if self.doOptimization then
        self.optimizer(self.feval, self.params, self.optimState)
    end
    if self.gpuid >= 0 and not self.doOptimization then
        --put back the gradParam by converting the cudaTensor to double
        --don't use =
        self.gradParamsDouble:copy(self.gradParams:double())
    end
end

function BidirectionalLSTM:prepare_input()
    local gpuid = self.gpuid
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

    if self.isTraining then
        self:buildVocab(sentences, sentence_toks)
    end

    local inputs = {}
    local inputs_rev = {}
    for step=1,maxLen do
        inputs[step] = torch.LongTensor(#sentences)
        for j=1,#sentences do
            local tokens = sentence_toks[j]
            if step > #tokens then
                inputs[step][j] = 0
            else
                local tok = sentence_toks[j][step]
                local tok_id = self.word2idx[tok]
                if tok_id == nil then
                    tok_id = self.word2idx['<UNK>']
                end
                inputs[step][j] = tok_id
            end
        end
        if gpuid >= 0 then inputs[step] = inputs[step]:cuda() end
    end
    
    -- only forward
    if not self.bidirection then
        return {inputs}
    end

    for step=1,maxLen do
        inputs_rev[step] = torch.LongTensor(#sentences)
        for j=1,#sentences do
            local tokens = sentence_toks[j]
            inputs_rev[step][j] = inputs[maxLen-step+1][j]
        end
        if gpuid >= 0 then inputs_rev[step] = inputs_rev[step]:cuda() end
    end
    self.maxLen = maxLen
    return {inputs, inputs_rev}
end

function BidirectionalLSTM:buildVocab(sentences, sentence_toks)
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
end

function BidirectionalLSTM:save_model(path)
    torch.save(path,self.net)
end

function BidirectionalLSTM:load_model(path)
    self.net = torch.load(path)
end
