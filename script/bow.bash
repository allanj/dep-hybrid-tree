#!/bin/bash

thread=40
l2s=(0.05 0.05 0.05 0.01 0.02 0.01 0.05 0.05)
iter=4000
langs=(en th de el zh id sv fa)
trainNum=600
testFile=test
sm=false
rm=false
sgmw=false
uef=false
aef=false
bowFeats=true
hmFeats=true


for (( l=0; l<${#l2s[@]}; l++ )) do
  l2=${l2s[$l]}
  lang=${langs[$l]}
  #for (( la=0; la<${#langs[@]}; la++ )) do
     #lang=${langs[$la]}
     java -cp dht-1.0.jar org.statnlp.example.depsemtree.DepHybridTree -t ${thread} -l2 ${l2} -bf ${bowFeats} -hmf ${hmFeats} \
       -iter ${iter} -lang ${lang} -tid ${trainNum} -tstid ${testFile} -sm ${sm} -uef ${uef} -aef ${aef} \
       -rm ${rm} -sgmw ${sgmw} > logs/dep_best_${lang}_test_${testFile}_${l2}_bow_${bowFeats}_hm_${hmFeats}.log 2>&1
  #done
done


