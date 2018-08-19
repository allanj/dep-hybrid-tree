/** Statistical Natural Language Processing System
    Copyright (C) 2014-2015  Lu, Wei
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.statnlp.example.sempar_bak;

import java.io.IOException;
import java.util.ArrayList;

import org.statnlp.commons.types.Instance;
import org.statnlp.hypergraph.DiscriminativeNetworkModel;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.NetworkModel;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;

public class DepHybridTree {
	
	public static int numThreads = 8;
	public static String language = "en";
	public static int numIteration = 100;
	
	public static void main(String args[]) throws IOException, InterruptedException{
		
		System.err.println(DepHybridTree.class.getCanonicalName());
		processArgs(args);
		
		String lang = language;
		String inst_filename = "data/geoquery/geoFunql-"+lang+".corpus";
		String init_filename = "data/geoquery/geoFunql-"+lang+".init.corpus";
		String g_filename = "data/hybridgrammar.txt";
		
		String train_ids = "data/geoquery-2012-08-27/splits/split-880/run-0/fold-0/train-N600";//+args[1];
		String test_ids = "data/geoquery-2012-08-27/splits/split-880/run-0/fold-0/test";
		
		boolean isGeoquery = true;
		
		NetworkConfig.NUM_THREADS = numThreads;
		NetworkConfig.AVOID_DUPLICATE_FEATURES = true;
		
		int numIterations = numIteration;
		
		
		SemTextDataManager dm = new SemTextDataManager();
		
		ArrayList<SemTextInstance> inits = SemTextInstanceReader.readInit(init_filename, dm);
		ArrayList<SemTextInstance> insts_train = SemTextInstanceReader.read(inst_filename, dm, train_ids, true);
		ArrayList<SemTextInstance> insts_test = SemTextInstanceReader.read(inst_filename, dm, test_ids, false);
		dm.printStat();
		
		int size = insts_train.size();
		
		SemTextInstance train_instances[] = new SemTextInstance[size];
		for(int k = 0; k<insts_train.size(); k++){
			train_instances[k] = insts_train.get(k);
			train_instances[k].setInstanceId(k);
			train_instances[k].setLabeled();
		}
		
		
		System.err.println("Read.."+train_instances.length+" instances.");
		
		HybridGrammar g = HybridGrammarReader.read(g_filename);
		
		SemanticForest forest_global = SemTextInstanceReader.toForest(dm);
		
		SemTextFeatureManager_Discriminative fm = new SemTextFeatureManager_Discriminative(new GlobalNetworkParam(), g, dm);
		
		SemTextNetworkCompiler compiler = new SemTextNetworkCompiler(g, forest_global, dm);
		
		NetworkModel model = DiscriminativeNetworkModel.create(fm, compiler);
		
		model.train(train_instances, numIterations);
		
		SemTextInstance test_instances[];
		Instance[] output_instances_unlabeled;
		
		test_instances = new SemTextInstance[insts_test.size()];
		for(int k = 0; k<test_instances.length; k++){
			test_instances[k] = insts_test.get(k);
			test_instances[k].setUnlabeled();
		}
		output_instances_unlabeled = model.decode(test_instances);
		
		double total = output_instances_unlabeled.length;
		double corr = 0;
		
		GeoqueryEvaluator eval = new GeoqueryEvaluator();
		
		ArrayList<String> expts = new ArrayList<String>();
		ArrayList<String> preds = new ArrayList<String>();
		
		for(int k = 0; k<output_instances_unlabeled.length; k++){
			Instance output_inst_U = output_instances_unlabeled[k];
			boolean r = output_inst_U.getOutput().equals(output_inst_U.getPrediction());
			System.err.println(output_inst_U.getInstanceId()+":\t"+r);
			if(r){
				corr ++;
			}
			System.err.println("=INPUT=");
			System.err.println(output_inst_U.getInput());
			System.err.println("=OUTPUT=");
			System.err.println(output_inst_U.getOutput());
			System.err.println("=PREDICTION=");
			System.err.println(output_inst_U.getPrediction());
			
			String expt = eval.toGeoQuery((SemanticForest)output_inst_U.getOutput());
			String pred = eval.toGeoQuery((SemanticForest)output_inst_U.getPrediction());
			
			expts.add(expt);
			preds.add(pred);
			
			if(isGeoquery){
				System.err.println("output:\t"+expt);
				System.err.println("predic:\t"+pred);
			}
		}
		
		System.err.println("text accuracy="+corr/total+"="+corr+"/"+total);
		eval.eval(preds, expts);
		
	}
	
	private static void processArgs(String[] args) {
		ArgumentParser parser = ArgumentParsers.newArgumentParser("")
				.defaultHelp(true).description("Dependency-based Hybrid Tree for Semantic Parsing");
	}
}