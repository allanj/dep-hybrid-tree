#!/bin/bash

thread=40
l2s=(0.05)
iter=12000
langs=(en)
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
hiddenSize=200
gpu=1
evalDev=true
evalFreq=100
types=(bilinear)
epochLim=9
lpw=false

for (( l=0; l<${#l2s[@]}; l++ )) do
  l2=${l2s[$l]}
  for (( la=0; la<${#langs[@]}; la++ )) do
    lang=${langs[$la]}
    for (( t=0; t<${#types[@]}; t++ )) do
      type=${types[$t]}
      java -cp dht-1.0.jar org.statnlp.example.depsemtree.DepHybridTree -t ${thread} -l2 ${l2} \
        -iter ${iter} -lang ${lang} -tid ${trainNum} -tstid ${testFile} -sm ${sm} -os linux \
        -rm ${rm} -nn ${nn} -ub ${useBatch} --batch ${batch} -optim ${optim} --hiddenSize ${hiddenSize} \
        -emb ${emb} --dropout ${dropout} --gpuid ${gpu} --evalDev ${evalDev} --type ${type} --epochLim ${epochLim} \
          -ef ${evalFreq} -lpw ${lpw} > logs/dep_baseline_${lang}_test_${testFile}_bow_em_${l2}_${nn}_${type}.log 2>&1 &  
    done
  done
done


