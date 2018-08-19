## GloVe.torch7

GloVe wrapper for Torch7.

### Installation
git clone https://github.com/rotmanmi/glove.torch

get the pre-trained GloVe word files from:
http://nlp.stanford.edu/projects/glove/


Make sure you specify the location of the pre-trained file, for instance 'glove.twitter.27B.25d.txt' (you can choose whichever datafile you want) file in 'glove.lua'. It is also suggested you specify a t7 file for fast access.



### [Tensor] word2vec(self,word,throwerror)
This function gets a word, and returns its word2vec representation, a tensor with the size 300. If throwerror is false (default) and the word doesn't exist it returns nil, otherwise, it will throw an exception.

### [table] distance(self,word,k)
This function returns the k-nearest neighbours to the given word. It returns a table with a list of words, and a corresponding list of cosine distances.


###Example
Getting the word2vec representation of the world 'Hello' and finding its k's nearest words.

```Lua

    local glove = require 'glove'
    local k = 3
    hellorep = glove:word2vec('Hello')
    neighbors = glove:distance(hellorep,k)
    

```
