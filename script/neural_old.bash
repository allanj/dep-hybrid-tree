#!/bin/bash

thread=40
l2s=(0.05 0.01 0.05 0.05 0.05 0.05)
iter=7200
langs=(de el zh id sv fa)
trainNum=600
testFile=test
sm=true
rm=false
nn=mlp
batch=1
useBatch=true
optim=sgdclip
emb=polyglot
dropout=0.0
hiddenSize=100
gpu=0
evalDev=false
evalFreq=600
types=(bilinear)
epochLim=7 ##limit to evaluate the deve set
lpw=false  ##loading the pretrained graphical model weights
fpw=false
fe=true  ##fix embedding
sgmw=true
dotest=false

for (( l=0; l<${#l2s[@]}; l++ )) do
  l2=${l2s[$l]}
  #for (( la=0; la<${#langs[@]}; la++ )) do
    lang=${langs[$l]}
    for (( t=0; t<${#types[@]}; t++ )) do
      type=${types[$t]}
      java -cp dht-1.0.jar org.statnlp.example.depsemtree.DepHybridTree -t ${thread} -l2 ${l2} \
        -iter ${iter} -lang ${lang} -tid ${trainNum} -tstid ${testFile} -sm ${sm} -os linux -fe ${fe} -sgmw ${sgmw} -dt ${dotest} \
        -rm ${rm} -nn ${nn} -ub ${useBatch} --batch ${batch} -optim ${optim} --hiddenSize ${hiddenSize} -fpw ${fpw} \
        -emb ${emb} --dropout ${dropout} --gpuid ${gpu} --evalDev ${evalDev} --type ${type} --epochLim ${epochLim} \
          -ef ${evalFreq} -lpw ${lpw} > logs/dep_baseline_${lang}_test_${testFile}_bow_em_${l2}_${nn}_${type}_fe_${fe}_sm_${sm}_rm_${rm}.log 2>&1 
    done
  #done
done


