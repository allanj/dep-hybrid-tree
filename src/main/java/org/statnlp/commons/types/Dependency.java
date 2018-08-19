package org.statnlp.commons.types;

import org.statnlp.example.depsemtree.SemanticUnit;

public class Dependency {

	int headIdx;
	int modifierIdx;
	SemanticUnit unit;
	
	public Dependency(int h, int m, SemanticUnit u) {
		this.headIdx = h;
		this.modifierIdx = m;
		this.unit = u;
	}

	@Override
	public String toString() {
		return "Dependency [headIdx=" + headIdx + ", modifierIdx=" + modifierIdx + ", unit=" + unit + "]\n";
	}
	
	
	
}
