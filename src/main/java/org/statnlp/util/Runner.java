/**
 * 
 */
package org.statnlp.util;

/**
 * The runner class for models developed using this StatNLP framework.
 */
public class Runner {

	public static void main(String[] args){
		Runner.run(args);
	}
	
	public static void run(String[] args){
		new GenericPipeline().parseArgs(args).execute();
	}
}
