#!/bin/bash

thread=40
l2s=(0.02 0.01 0.05 0.05)
iter=4000
langs=(zh id sv fa)
trainNum=600
testFile=test
sm=true
rm=false


for (( l=0; l<${#l2s[@]}; l++ )) do
  l2=${l2s[$l]}
  lang=${langs[$l]}
  # for (( la=0; la<${#langs[@]}; la++ )) do
     # lang=${langs[$la]}
     java -cp dht-1.0.jar org.statnlp.example.depsemtree.DepHybridTree -t ${thread} -l2 ${l2} \
       -iter ${iter} -lang ${lang} -tid ${trainNum} -tstid ${testFile} -sm ${sm} -uef false -aef true\
       -rm ${rm} > logs/dep_current_${lang}_test_${testFile}_bow_em_${l2}_base.log 2>&1
  # done
done


