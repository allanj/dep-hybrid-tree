#!/bin/bash

l2=0.01
langs=(en th de el zh id sv fa)
for (( l=0; l<${#langs[@]}; l++ )) do
  lang=${langs[$l]}
  java -cp hybridtree.jar com.statnlp.sp.main.SemTextExperimenter_Discriminative 16 ${lang} > logs/hybridtree_${lang}.log 2>&1
done
