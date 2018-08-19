-- Helper functions --

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

function array2Tensor2D(array)
    local res = torch.Tensor(#array, #array[1])
    for i = 1, #array do
        for j = 1, #array[1] do
            res[i][j] = array[i][j]
        end
    end
    return res
end

function loadGlove(wordList, dim, sharedLookupTable)
    sharedLookupTable = sharedLookupTable or false
    if glove == nil then
        ---- TODO: need to make this path more general later.
        glove = require 'nn-crf-interface/neural_server/glove_torch/glove'
    end
    glove:load(dim)

    specialSymbols = {}
    specialSymbols['<PAD>'] = torch.Tensor(dim):normal(0,1)
    specialSymbols['<S>'] = torch.Tensor(dim):normal(0,1)
    specialSymbols['</S>'] = torch.Tensor(dim):normal(0,1)

    local ltw
    local maskZero = false
    if sharedLookupTable then
        ltw = nn.LookupTableMaskZero(#wordList, dim)
        maskZero = true
    else 
        ltw = nn.LookupTable(#wordList, dim)
    end
    for i=1,#wordList do
        local emb = torch.Tensor(dim)

        local p_emb = glove:word2vec(wordList[i])
        if p_emb == nil then
            p_emb = specialSymbols[wordList[i]]
        end
        for j=1,dim do
            emb[j] = p_emb[j]
        end
        if maskZero then
            --becareful about this, check nn.LookupTableMaskZero documentation
            ltw.weight[i + 1] = emb
        else
            ltw.weight[i] = emb
        end
    end
    return ltw
end

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

-- function loadPolyglot(wordList, lang)
--     if polyglot == nil then
--         polyglot = require 'nn-crf-interface/neural_server/polyglot/polyglot'
--     end
--     polyglot:load(lang)
--     ltw = nn.LookupTable(#wordList, 64)
--     for i=1,#wordList do
--         local emb = torch.Tensor(64)
--         local p_emb = polyglot:word2vec(wordList[i])
--         for j=1,64 do
--             emb[j] = p_emb[j]
--         end
--         ltw.weight[i] = emb
--     end
--     return ltw
-- end

function loadPolyglotEmbObj(lang)
    local embFile = 'nn-crf-interface/neural_server/polyglot/polyglot-'..lang..'.t7'
    if not paths.filep(embFile) then
        error('Please run bintot7.lua to preprocess Polyglot data!')
    else
        polyglot = torch.load(embFile)
        print('Done reading Polyglot data.')
    end
    polyglot.unkToken = "<UNK>"
    polyglot.word2vec = function ( self, word )
        local ind = self.w2vvocab[word] 
        if ind == nil then
            ind = self.w2vvocab[self.unkToken]
        end
        return self.M[ind]
    end
    return polyglot
end

