#!/bin/bash

thread=40
l2s=(0.02)
iter=12000
langs=(zh)
trainNum=600
testFile=test
sm=false
rm=true
nn=mlp
batch=1
useBatch=true
optim=sgdclip
emb=polyglot
dropout=0.0
hiddenSize=100
gpu=0
evalDev=false
evalFreq=100
types=(bilinear)
epochLim=7 ##limit to evaluate the deve set
lpw=false  ##loading the pretrained graphical model weights
fe=true  ##fix embedding
fpw=false
regn=false
window=1
dotest=true
moreNeural=false
saveEpochModel=false
loadEpochModel=true
loadEpochModelNums=(7 8 9 10 11 12 13 14 15 16 17 18 19)

for (( l=0; l<${#l2s[@]}; l++ )) do
  l2=${l2s[$l]}
  lang=${langs[$l]}
  #for (( la=0; la<${#langs[@]}; la++ )) do
   # lang=${langs[$la]}
    for (( t=0; t<${#types[@]}; t++ )) do
      type=${types[$t]}
      for (( ln=0; ln<${#loadEpochModelNums[@]}; ln++ )) do
        loadEpochModelNum=${loadEpochModelNums[$ln]}
        java -cp dht-1.0.jar org.statnlp.example.depsemtree.DepHybridTree -t ${thread} -l2 ${l2} -regn ${regn} -mhws ${window}\
          -iter ${iter} -lang ${lang} -tid ${trainNum} -tstid ${testFile} -sm ${sm} -os linux -fe ${fe} -fpw ${fpw} -mn ${moreNeural} \
          -rm ${rm} -nn ${nn} -ub ${useBatch} --batch ${batch} -optim ${optim} --hiddenSize ${hiddenSize} -dt ${dotest} \
          -emb ${emb} --dropout ${dropout} --gpuid ${gpu} --evalDev ${evalDev} --type ${type} --epochLim ${epochLim} \
            -ef ${evalFreq} -sem ${saveEpochModel} -lem ${loadEpochModel} -lemn ${loadEpochModelNum} \
            -lpw ${lpw} > logs/dep_best_${lang}_test_${testFile}_win_${window}_${l2}_${nn}_${type}_hidd_${hiddenSize}_fe_${fe}_more_${moreNeural}_${loadEpochModelNum}.log 2>&1
     done 
   done
  #done
done
