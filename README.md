### Dependency-basded Hybrid Tree for Semantic Parsing
This repository is the implementation of the paper _"Dependency-based Hybrid Tree for Semantic parsing"_ appeared in the Empirical Methods in Natural Language Processing (EMNLP), 2018. The paper will be released soon. 


#### Requirements
* Java 1.8
* Perl and swipl (running the script to query GeoQeuery database)
* [Torch](http://torch.ch/docs/getting-started.html#) (optional, required if running the neural version)

#### Usage

##### Feature-based model
We compiled a jar file (including all the required external library in Maven) `dht.jar`.
```bash
	java -cp dht.jar org.statnlp.example.depsemtree.DepHybridTree --thread 40 --language en
```
You should be able to obtain exactly same performance in the paper with the above command. To change another language (e.g., `th`, `de`, `el`, `zh`, `id`, `sv`, `fa`), simply replace `en` with other languages indicated in the paper. For further customized configuration settings (e.g., L2, features, etc), please check the main class and we will list the details with another README document soon. 

###### Using embeddings as feature values 
First, download the embedding file from this [link](https://drive.google.com/open?id=1lV4nwFrkFkyBtKGiD_5FoSRlQEQ5EtQ-) and put them under `nn-crf-interface/neural_server/polyglot/` directory. Then you can run the jar file using the following command:
```bash
	java -cp dht.jar org.statnlp.example.depsemtree.DepHybridTree --thread 40 --language en --useEmbFeats true
```
You should be able to obtain exactly same performance in the paper with the above command. 

##### Neural network-augmented model
Make sure you have installed the latest Torch package. 
```bash
	java -cp dht.jar org.statnlp.example.depsemtree.DepHybridTree --thread 40 --language en --type bilinear
```
