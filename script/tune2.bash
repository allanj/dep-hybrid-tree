#!/bin/bash

thread=40
l2s=(0.05)
#iter=6000
langs=(en)
trainNum=600
testFile=test
sm=true
rm=false
nn=mlp
#batch=1
useBatch=true
#optim=sgdclip
emb=polyglot
dropout=0.0
hiddenSize=200
gpu=1
evalDev=true
evalFreq=100
types=(bilinear)
epochLim=8 ##limit to evaluate the deve set
lpw=false  ##loading the pretrained graphical model weights
fpw=false
#fe=false  ##fix embedding


batches=(1 1 1 10 20 50)
fes=(false true false true true true)
optims=(sgdclip adam adam sgdclip sgdclip sgdclip)
iters=(7200 7200 7200 3000 1500 600)

for (( l=0; l<${#l2s[@]}; l++ )) do
  l2=${l2s[$l]}
  for (( la=0; la<${#langs[@]}; la++ )) do
    lang=${langs[$la]}
    for (( t=0; t<${#types[@]}; t++ )) do
      type=${types[$t]}
      for (( k=0; k<${#batches[@]}; k++ )) do
        batch=${batches[$k]}
        fe=${fes[$k]}
        optim=${optims[$k]}
        iter=${iters[$k]}
        java -cp dht-1.0.jar org.statnlp.example.depsemtree.DepHybridTree -t ${thread} -l2 ${l2} \
          -iter ${iter} -lang ${lang} -tid ${trainNum} -tstid ${testFile} -sm ${sm} -os linux -fe ${fe} \
          -rm ${rm} -nn ${nn} -ub ${useBatch} --batch ${batch} -optim ${optim} --hiddenSize ${hiddenSize} -fpw ${fpw} \
          -emb ${emb} --dropout ${dropout} --gpuid ${gpu} --evalDev ${evalDev} --type ${type} --epochLim ${epochLim} \
            -ef ${evalFreq} -lpw ${lpw} > logs/dep_baseline_${lang}_test_${testFile}_${l2}_${nn}_${type}_batch_${batch}_${optim}_fe_${fe}.log 2>&1 
      done
    done
  done
done


