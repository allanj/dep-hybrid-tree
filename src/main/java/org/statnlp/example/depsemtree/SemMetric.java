package org.statnlp.example.depsemtree;

import org.statnlp.hypergraph.decoding.Metric;

public class SemMetric implements Metric {

	double acc;
	
	public SemMetric(double acc) {
		this.acc = acc;
	}

	@Override
	public boolean isBetter(Metric other) {
		SemMetric metric = (SemMetric)other;
		return acc > metric.acc;
	}

	@Override
	public Object getMetricValue() {
		return this.acc;
	}

	@Override
	public String toString() {
		return "SemMetric [acc=" + acc + "]";
	}
	

}
