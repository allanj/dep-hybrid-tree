package org.statnlp.example.dependency;

import org.statnlp.commons.types.Sentence;
import org.statnlp.example.base.BaseInstance;

public class DepInstance extends BaseInstance<DepInstance, Sentence, int[]> {

	public DepInstance(int instanceId, double weight) {
		super(instanceId, weight);
	}
	
	public DepInstance(int instanceId, double weight, Sentence sent) {
		super(instanceId, weight);
		this.input = sent;
	}

	private static final long serialVersionUID = 7472469003829845696L;

	@Override
	public int size() {
		return input.length();
	}

	public Sentence duplicateInput(){
		return input;
	}
	
	public int[] duplicateOutput() {
		return this.output.clone();
	}
	
}
