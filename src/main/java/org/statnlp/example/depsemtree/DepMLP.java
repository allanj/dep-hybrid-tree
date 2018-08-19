package org.statnlp.example.depsemtree;

import org.statnlp.hypergraph.neural.NeuralNetworkCore;

public class DepMLP extends NeuralNetworkCore {

	private static final long serialVersionUID = -3027039349148738046L;

	public DepMLP(String className, 
			int numLabels, int hiddenSize, int gpuId, String embedding, boolean fixEmbedding, double dropout,
			String language, String type, int windowSize) {
		super(numLabels, gpuId);
		config.put("class", className);
		config.put("embedding", embedding);
		config.put("fixEmbedding", fixEmbedding);
		config.put("dropout", dropout);
		config.put("lang", language);
		config.put("hiddenSize", hiddenSize);
		config.put("type", type);
		config.put("windowSize", windowSize);
	}

	@Override
	public int hyperEdgeInput2OutputRowIndex(Object edgeInput) {
		int wordWindowID = this.getNNInputID(edgeInput); 
		return wordWindowID;
	}

	@Override
	public Object hyperEdgeInput2NNInput(Object edgeInput) {
		return edgeInput;
	}

}
