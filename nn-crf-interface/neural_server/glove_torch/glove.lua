torch.setdefaulttensortype('torch.DoubleTensor')

-- opt = {
--     binfilename = 'glove_torch/glove.6B.50d.txt',
--     outfilename = 'glove_torch/glove.6B.50d.t7'
-- }
local GloVe = {}
-- if not paths.filep(opt.outfilename) then
-- 	GloVe = require('bintot7.lua')
-- else
-- 	GloVe = torch.load(opt.outfilename)
-- 	print('Done reading GloVe data.')
-- end

GloVe.load = function (self,dim)
    local gloveFile = 'nn-crf-interface/neural_server/glove_torch/glove.6B.' .. dim .. 'd.t7'
    if not paths.filep(gloveFile) then
        error('Please run bintot7.lua to preprocess Glove data!')
    else
        GloVe.glove = torch.load(gloveFile)
        print('Done reading GloVe data.')
    end
end

GloVe.distance = function (self,vec,k)
	local k = k or 1	
	--self.zeros = self.zeros or torch.zeros(self.M:size(1));
	local norm = vec:norm(2)
	vec:div(norm)
	local distances = torch.mv(self.glove.M ,vec)
	distances , oldindex = torch.sort(distances,1,true)
	local returnwords = {}
	local returndistances = {}
	for i = 1,k do
		table.insert(returnwords, self.glove.v2wvocab[oldindex[i]])
		table.insert(returndistances, distances[i])
	end
	return {returndistances, returnwords}
end

GloVe.word2vec = function (self,word,throwerror)
   local throwerror = throwerror or false
   local ind = self.glove.w2vvocab[word]
   if throwerror then
		assert(ind ~= nil, 'Word does not exist in the dictionary!')
   end   
	if ind == nil then
		ind = self.glove.w2vvocab['UNK']
        if ind == nil then
            ind = self.glove.w2vvocab['unk']
        end
	end
   return self.glove.M[ind]
end

return GloVe
