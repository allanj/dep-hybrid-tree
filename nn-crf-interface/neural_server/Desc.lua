local Desc, parent = torch.class('nn.Desc', 'nn.Module')

--select the descriptor
---assuming always the batch input
function Desc:__init(dimension)
    parent.__init(self)
    self.dimension = dimension
    self.gradInput = {self.gradInput, self.gradInput.new()}
end

function Desc:updateOutput(input)
    local t = input[1]
    local descs = input[2]
    local new_sizes = t:size()
    new_sizes[self.dimension] = descs:size(2)
    self.output:resize(new_sizes)
    for i = 1, t:size(1) do
        self.output[i] = t[i]:index(self.dimension-1, descs[i])
    end
    return self.output
end

function Desc:updateGradInput(input, gradOutput)
    local t = input[1]
    local desc = input[2]
    self.gradInput[2]:resize(desc:size()):zero()
    local gradInput = self.gradInput[1] -- no gradient for the Desc variable
    local new_sizes = t:size()
    gradInput:resize(t:size()):zero()
    for i = 1, t:size(1) do
        gradInput[i]:indexAdd(self.dimension-1, desc[i], gradOutput[i])
    end
    return self.gradInput
end

function Desc:clearState()
    self.gradInput[1]:set()
    self.gradInput[2]:set()
    self.output:set()
    return self
end