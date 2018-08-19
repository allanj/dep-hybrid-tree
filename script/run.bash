#!/bin/bash

thread=40
l2=0.001
iter=4000
lang=en
trainNum=600
testFile=test
sm=true
rm=false

java -cp dht-1.0.jar org.statnlp.example.depsemtree.DepHybridTree -t ${thread} -l2 ${l2} \
   -iter ${iter} -lang ${lang} -tid ${trainNum} -tstid ${testFile} -sm ${sm} \
   -rm ${rm} > logs/dep_baseline_test_${testFile}_bow.log 2>&1


