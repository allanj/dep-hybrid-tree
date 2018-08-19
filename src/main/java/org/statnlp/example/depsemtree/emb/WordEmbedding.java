package org.statnlp.example.depsemtree.emb;

public interface WordEmbedding {
	
	void readEmbedding(String file);
	
	double[] getEmbedding(String word);
	
	void clearEmbeddingMemory();
	
	int getDimension();
}
