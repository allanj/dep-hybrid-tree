package org.statnlp.example.depsemtree;

import java.util.AbstractMap.SimpleImmutableEntry;

import org.statnlp.hypergraph.neural.NeuralNetworkCore;

public class GeoLSTM extends NeuralNetworkCore {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8501658392325628263L;

	public GeoLSTM(String className, int numLabels, int mlpHiddenSize, int gpuid, 
			String embedding, boolean fixEmbedding, String lang, String type) {
		super(numLabels, gpuid);
		config.put("class", className);
		config.put("mlpHiddenSize", mlpHiddenSize);
		config.put("embedding", embedding);
		config.put("fixEmbedding", fixEmbedding);
		config.put("lang", lang);
		config.put("type", type);
	}

	@Override
	public int hyperEdgeInput2OutputRowIndex(Object edgeInput) {
		@SuppressWarnings("unchecked")
		SimpleImmutableEntry<String, Integer> eInput = (SimpleImmutableEntry<String, Integer>)edgeInput;
		int position = eInput.getValue();
		return position * this.getNNInputSize() + this.getNNInputID(eInput.getKey());
	}

	@Override
	public Object hyperEdgeInput2NNInput(Object edgeInput) {
		@SuppressWarnings("unchecked")
		SimpleImmutableEntry<String, Integer> eInput = (SimpleImmutableEntry<String, Integer>)edgeInput;
		return eInput.getKey();
	}

}
