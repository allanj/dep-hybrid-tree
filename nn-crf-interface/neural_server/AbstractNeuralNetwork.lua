local AbstractNeuralNetwork = torch.class('AbstractNeuralNetwork')

function AbstractNeuralNetwork:__init(doOptimization)
    self.outputPtr = {}
    self.gradOutputPtr = {}
    self.net = nil
    self.x = nil
    self.params = nil
    self.paramsPtr = nil
    self.gradParams = nil
    self.gradParamsPtr = nil
    self.doOptimization = doOptimization
    if self.doOptimization then
        self.optimizer = optim.sgd
        self.feval = function (params) return 0, self.gradParams end
    end
    -- You may initialize other member attributes here
end

function AbstractNeuralNetwork:initialize(data, ...)
    -- Define the network here according to supplied data
    local outputAndGradOutputPtr = {... }
    -- In practice, this should point to the Tensor in the Java program
    self.outputPtr = torch.Tensor()
    self.gradOutputPtr = torch.Tensor()
    self.net = nn.Module()
    self.params, self.gradParams = self.net:getParameters()
    self:prepare_input()
    if not self.doOptimization then
        self.params:retain()
        self.paramsPtr = torch.pointer(self.params)
        self.gradParams:retain()
        self.gradParamsPtr = torch.pointer(self.gradParams)
        return self.paramsPtr, self.gradParamsPtr 
    end
end

function AbstractNeuralNetwork:prepare_input()
    -- Set the batch input ``x'' accordingly
    self.x = torch.Tensor()
end

function AbstractNeuralNetwork:forward(isTraining, batchInputIds)
    -- Implement forward computation.
    self.isTraining = isTraining
    local output = self.net:forward(self.x)
    -- Update outputPtr accordingly.
    self.outputPtr:copy(output)
end

function AbstractNeuralNetwork:backward()
    -- Implement backward computation.
    -- gradOutput is assumed to be set externally in the Java program
    self.gradParams:zero()
    self.net:backward(self.x, self.gradOutput)
    if self.doOptimization then
        self.optimizer(self.feval, self.params, self.optimState)
    end
end

function AbstractNeuralNetwork:save_model(path)
    torch.save(path,self.net)
end

function AbstractNeuralNetwork:load_model(path)
    self.net = torch.load(path)
end
