python get_raw_sents.py ../data/conll2003/eng.train > eng.train.raw
python get_raw_sents.py ../data/conll2003/eng.testb > eng.testb.raw
th ../senna-torch/runme.lua < eng.train.raw > eng.train.feats
th ../senna-torch/runme.lua < eng.testb.raw > eng.testb.feats
python toCoNLL.py ../data/conll2003/eng.train eng.train.feats > eng.train.conll
python toCoNLL.py ../data/conll2003/eng.testb eng.testb.feats > eng.testb.conll