### Dependency-basded Hybrid Tree for Semantic Parsing
This repository is the implementation of the paper _"Dependency-based Hybrid Tree for Semantic parsing"_ appeared in the Empirical Methods in Natural Language Processing (EMNLP), 2018. The paper will be released soon. 


#### Requirements
* Java 1.8
* Perl (running the script to query GeoQeuery database)
* [Torch](http://torch.ch/docs/getting-started.html#) (optional, required if running the neural version)

#### Usage

##### Feature-based model
We compiled a jar file (including all the required external library in maven) `dht.jar`.
```bash
java -cp dht.jar org.statnlp.example.depsemtree.DepHybridTree --thread 40 -lang en
```
You should be able to obtain exactly same performance in the paper. To change another language, simply replace `en` with other languages indicated in the paper. For futher customized configuration settings (e.g., L2, features, etc), please check the main class and we will list the details with another README document soon.

##### Neural network-augmented model
Make sure you have installed the latest Torch package. 
```bash
java -cp dht.jar org.statnlp.example.depsemtree.DepHybridTree --thread 40 -lang en -type bilinear
```
