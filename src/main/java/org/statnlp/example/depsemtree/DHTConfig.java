package org.statnlp.example.depsemtree;

public class DHTConfig {

	public static final String SEP = " ### ";
	public static final String dummyUnit = "dummy";
	
	public static int _SEMANTIC_PARSING_NGRAM = 1;//2;
	public static int _SEMANTIC_FOREST_MAX_DEPTH = 20;//the max depth of the forest when creating the semantic forest.
	
	
	/***
	 * Read the tuned L2 parameters for non-neural model
	 * @param lang
	 * @return
	 */
	public static double tunedL2CRF(String lang) {
		if (lang.equals("en") || lang.equals("th") || lang.equals("de") || lang.equals("sv")|| lang.equals("fa")) {
			return 0.05;
		} else if (lang.equals("el") || lang.equals("id")){
			return 0.01;
		} else if (lang.equals("zh")){
			return 0.02;
		} else {
			throw new RuntimeException("unknown language: " + lang); 
		}
	}
	
	public static double tunedL2EmbCRF(String lang) {
		if (lang.equals("en") || lang.equals("th") || lang.equals("de") || lang.equals("sv")|| lang.equals("fa") || lang.equals("zh") || lang.equals("id")) {
			return 0.05;
		} else if (lang.equals("el")){
			return 0.01;
		} else {
			throw new RuntimeException("unknown language: " + lang); 
		}
	}
	
}
