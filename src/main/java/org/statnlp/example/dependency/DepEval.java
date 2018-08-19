package org.statnlp.example.dependency;

import java.util.Arrays;
import java.util.HashSet;

import org.statnlp.commons.types.Instance;
import org.statnlp.commons.types.Sentence;


public class DepEval {

	
	private static HashSet<String> punct = new HashSet<>(Arrays.asList("''", ",", ".", ":", "``", "-LRB-", "-RRB-"));
	
	public static void evalDep(Instance[] testInsts){
		int dp_corr=0;
		int noPunc_corr = 0;
		int noPunc_total = 0;
		int dp_total=0;
		for (int index = 0; index < testInsts.length; index++) {
			DepInstance inst = (DepInstance)(testInsts[index]);
			Sentence sent = inst.getInput();
			int[] prediction = inst.getPrediction();
			int[] output = inst.getOutput();
			for (int i = 1; i < prediction.length; i++) {
				if (output[i] == prediction[i]) {
					dp_corr++;
				}
				dp_total++;
				if (!punct.contains(sent.get(i).getTag())) {
					if (output[i] == prediction[i]) {
						noPunc_corr++;
					}
					noPunc_total++;
				}
			}
		}
		System.out.println("**Evaluating Dependency Result**");
		System.out.println("[Dependency] Correct: "+ dp_corr);
		System.out.println("[Dependency] total: "+ dp_total);
		System.out.printf("[Dependency] UAS: %.2f\n", dp_corr*1.0/dp_total*100);
		System.out.println("[Dependency] (Without Punctuation) Correct: "+ noPunc_corr);
		System.out.println("[Dependency] (Without Punctuation)  Total: "+ noPunc_total);
		System.out.printf("[Dependency] (Without Punctuation)  UAS: %.2f\n", noPunc_corr*1.0/noPunc_total*100);
		System.out.println("*************************");
	}
	
	
}
