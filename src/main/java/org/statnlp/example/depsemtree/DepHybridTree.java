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
package org.statnlp.example.depsemtree;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import org.statnlp.commons.io.RAWF;
import org.statnlp.commons.ml.opt.GradientDescentOptimizer.BestParamCriteria;
import org.statnlp.commons.ml.opt.OptimizerFactory;
import org.statnlp.commons.types.Instance;
import org.statnlp.example.depsemtree.emb.FastTextWordEmbedding;
import org.statnlp.example.depsemtree.emb.PolyGlotWordEmbedding;
import org.statnlp.example.depsemtree.emb.WordEmbedding;
import org.statnlp.hypergraph.DiscriminativeNetworkModel;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.hypergraph.NetworkConfig.StoppingCriteria;
import org.statnlp.hypergraph.NetworkModel;
import org.statnlp.hypergraph.decoding.Metric;
import org.statnlp.hypergraph.neural.GlobalNeuralNetworkParam;
import org.statnlp.hypergraph.neural.NeuralNetworkCore;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;

public class DepHybridTree {
	
	public static int numThreads = 8;
	public static String language = "en";
	public static int numIteration = 100;
	public static double l2 = 0.01;
	public static String trainIDFile = "600";
	public static String testIDFile = "test";
	public static boolean saveModel = false;
	public static boolean readModel = false;
	public static String modelFolder = "models/";
	public static StoppingCriteria criteria = StoppingCriteria.SMALL_RELATIVE_CHANGE;
	public static NeuralType nn = NeuralType.none;
	public static OptimizerFactory optimizer = OptimizerFactory.getLBFGSFactory();
	public static int gpuId = -1;
	public static String embedding = "polyglot";
	public static double dropout = 0.0;
	public static int embeddingSize = 50;
	public static int hiddenSize = 0;
	public static int mlpHalfWindowSize = 0;
	public static boolean evalDev = false;
	public static int evalK = 11;
	public static String type = "none";
	public static boolean fixEmbedding = false; //by default do not fix the embedding.
	public static boolean useEmbeddingFeature = false;
	public static WordEmbedding emb = null;
	public static boolean averageEmbFeats = true; //only use while the embedding feature is enable. False means just split true is better
	public static int epochLim = 1;
	public static boolean saveGMWeightOnly = false; //if saving the graphical model weights only
	public static boolean loadPretrainedWeight = false;
	public static boolean fixPretrainedWeight = true;
	public static String optimStr = "lbfgs"; 
	public static boolean doTest = true;
	public static boolean moreNeural = false;
	public static boolean bowFeats =true;
	public static boolean hmFeats = true;
	public static boolean saveEpochModel = false;
	public static boolean loadEpochModel = false;
	public static int loadEpochModelNum = -1;
	
	public static enum NeuralType {
		mlp, //comes with different type
		lstm,
		none
	}
	
	public static void main(String args[]) throws IOException, InterruptedException, ClassNotFoundException{
		
		System.err.println(DepHybridTree.class.getCanonicalName());
		processArgs(args);
		
		String lang = language;
		String inst_filename = "data/geoquery/geoFunql-"+lang+".corpus";
		String init_filename = "data/geoquery/geoFunql-"+lang+".init.corpus";
		String g_filename = "data/dephybridgrammar.txt";
		
		String train_ids = "data/geoquery-2012-08-27/splits/split-880/run-0/fold-0/train-N"+trainIDFile;//+args[1];
//		String train_ids = "data/geoquery-2012-08-27/splits/split-880/run-0/fold-0/"+testIDFile;//+args[1];
		String test_ids = "data/geoquery-2012-08-27/splits/split-880/run-0/fold-0/" + testIDFile;
		String dev_ids = "data/geoquery-2012-08-27/splits/split-880/run-0/fold-0/" + testIDFile;
//		String test_ids = "data/geoquery-2012-08-27/splits/split-880/run-0/fold-0/train-N"+trainIDFile;
		
		boolean isGeoquery = true;
		
		NetworkConfig.NUM_THREADS = numThreads;
		NetworkConfig.AVOID_DUPLICATE_FEATURES = true;
		NetworkConfig.L2_REGULARIZATION_CONSTANT = l2;
		NetworkConfig.FEATURE_TOUCH_TEST = false;
		NetworkConfig.STOPPING_CRITERIA = criteria;
		NetworkConfig.FEATURE_TOUCH_TEST = NetworkConfig.USE_NEURAL_FEATURES ? true : false;
		
		int numIterations = numIteration;
		String modelFile = "dht_"+language+"_l2_"+l2+"_"+trainIDFile+"_" + criteria.toString() + "_" + nn + "_"
				+ type +"_win_"+mlpHalfWindowSize+ "_hidd_" + hiddenSize + "_emb_"+useEmbeddingFeature+"_"+embedding + "_"+optimStr + "_batch_" + NetworkConfig.BATCH_SIZE+"_fe_" + 
				fixEmbedding;
		String nnModelFile = "dhtnn_"+language+"_"+ nn  +"_win_"+mlpHalfWindowSize+ "_"+ embedding + "_" + type + "_hidd_" + hiddenSize + "_"+optimStr + "_batch" + NetworkConfig.BATCH_SIZE+"_fe" + 
				fixEmbedding+ ".m";
		if (saveEpochModel) {
			NetworkConfig.SAVE_EPOCH_MODEL = true;
			NetworkConfig.EPOCH_MODEL_FILE = modelFolder + modelFile;
			NetworkConfig.EPOCH_NN_MODEL_FILE = NetworkConfig.USE_NEURAL_FEATURES ?  modelFolder + nnModelFile : null;
		}
		NetworkModel model = null;
		ArrayList<SemTextInstance> insts_test = null;
		ArrayList<SemTextInstance> insts_dev = null;
		if (!readModel) {
			SemTextDataManager dm = new SemTextDataManager();
			
			SemTextInstanceReader.readInit(init_filename, dm);
			ArrayList<SemTextInstance> insts_train = SemTextInstanceReader.read(inst_filename, dm, train_ids, true, lang);
			insts_test = SemTextInstanceReader.read(inst_filename, dm, test_ids, false, lang);
			insts_dev = evalDev ? SemTextInstanceReader.read(inst_filename, dm, dev_ids, false, lang):
				null;
			SemTextInstance[] dev_instances = null;
			if (evalDev) {
				dev_instances = new SemTextInstance[insts_dev.size()];
				for(int k = 0; k<dev_instances.length; k++){
					dev_instances[k] = insts_dev.get(k);
					dev_instances[k].setUnlabeled();
				}
			}
//			dm.printStat();
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
			
			List<NeuralNetworkCore> nets = new ArrayList<NeuralNetworkCore>();
			if(NetworkConfig.USE_NEURAL_FEATURES){
				if (nn == NeuralType.mlp) {
					nets.add(new DepMLP("MultiLayerPerceptron", dm.getAllUnits().size(), hiddenSize, gpuId,
							embedding, fixEmbedding, dropout, lang, type, mlpHalfWindowSize)
							.setModelFile(modelFolder + nnModelFile));
				} else if (nn == NeuralType.lstm) {
					nets.add(new GeoLSTM("SimpleBiLSTM", dm.getAllUnits().size(), hiddenSize, gpuId,
							embedding, fixEmbedding, lang, type)
							.setModelFile(modelFolder + nnModelFile));
				} else {
					throw new RuntimeException("unknown nn type: " + nn);
				}
				
			} 
			GlobalNetworkParam gnp = new GlobalNetworkParam(optimizer, new GlobalNeuralNetworkParam(nets));
//			gnp.setStoreFeatureReps();
			
			DHTFeatureManager fm = new DHTFeatureManager(gnp, g, dm, nn, emb, averageEmbFeats, mlpHalfWindowSize, moreNeural, bowFeats, hmFeats);
			
			DHTNetworkCompiler compiler = new DHTNetworkCompiler(g, forest_global, dm);
			
			model = DiscriminativeNetworkModel.create(fm, compiler);
			if (evalDev) {
				Function<Instance[], Metric> evalFunc = new Function<Instance[], Metric>(){
					@Override
					public Metric apply(Instance[] t) {
						try {
							return evaluate(t, isGeoquery);
						} catch (IOException e) {
							e.printStackTrace();
						}
						return null;
					}
				};
				model.train(train_instances, numIterations, dev_instances, evalFunc, evalK, epochLim);
			} else {
				model.train(train_instances, numIterations);
//				gnp.getStringIndex().buildReverseIndex();
//				StringIndex strIdx = gnp.getStringIndex();
//				strIdx.buildReverseIndex();
//				gnp.setStoreFeatureReps();
//				for (int i = 0; i < gnp.size(); i++) {
//					int[] fs = gnp.getFeatureRep(i);
//					String type = strIdx.get(fs[0]);
//					String output = strIdx.get(fs[1]);
//					String input = strIdx.get(fs[2]);
//					System.out.println("["+type+", " + output +", " + input + "]" );
//				}
			}
		} else {
			System.out.println("[Info] Reading the model..");
			String file = modelFolder + modelFile;
			if (loadEpochModel) file += loadEpochModelNum;
			ObjectInputStream ois = RAWF.objectReader(file);
			model = (NetworkModel)ois.readObject();
			ois.close();
			DHTFeatureManager sfm = (DHTFeatureManager)model.getFeatureManager();
			insts_test = SemTextInstanceReader.read(inst_filename, sfm._dm, test_ids, false, lang);
		}
		
		if (saveModel) {
			//save both stuff
			System.out.println("[Info] Saving the model...");
			ObjectOutputStream oos =  RAWF.objectWriter(modelFolder + modelFile);
			if (NetworkConfig.USE_NEURAL_FEATURES)
				model.getFeatureManager().getParam_G().getNNParamG().getAllNets().get(0).setModelFile(modelFolder + nnModelFile);
			oos.writeObject(model);
			oos.close();
			if (saveGMWeightOnly) {
				oos =  RAWF.objectWriter(modelFolder + modelFile + ".weight");
				oos.writeObject(model.getFeatureManager().getParam_G().getWeights());
				oos.close();
			}
		}
		
		if (doTest) {
			SemTextInstance test_instances[];
			Instance[] output_instances_unlabeled;
			
			test_instances = new SemTextInstance[insts_test.size()];
			for(int k = 0; k<test_instances.length; k++){
				test_instances[k] = insts_test.get(k);
				test_instances[k].setUnlabeled();
			}
			output_instances_unlabeled = model.test(test_instances);
			
			evaluate(output_instances_unlabeled, isGeoquery);
		}
		
	}
	
	public static SemMetric evaluate(Instance[] output_instances_unlabeled, boolean isGeoquery) throws IOException {
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
			} else {
//				System.err.println("=INPUT=");
//				System.err.println(output_inst_U.getInput());
//				System.err.println("=OUTPUT=");
//				System.err.println(output_inst_U.getOutput());
//				System.err.println("=PREDICTION=");
//				System.err.println(output_inst_U.getPrediction());
			}
			System.err.println("=INPUT=");
			System.err.println(output_inst_U.getInput());
			System.err.println("=OUTPUT=");
			System.err.println(output_inst_U.getOutput());
			System.err.println("=PREDICTION=");
			System.err.println(output_inst_U.getPrediction());
			System.err.println("=Dependencies=");
			System.err.println(((SemTextInstance)output_inst_U).getInput().deps.toString());
			
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
		return eval.eval(preds, expts);
	}
	
	private static void processArgs(String[] args) throws IOException, ClassNotFoundException {
		ArgumentParser parser = ArgumentParsers.newArgumentParser("")
				.defaultHelp(true).description("Dependency-based Hybrid Tree for Semantic Parsing");
		parser.addArgument("-t", "--thread").type(Integer.class).setDefault(numThreads).help("number of threads");
		parser.addArgument("--l2reg", "-l2").type(Double.class).setDefault(l2).help("L2 Regularization");
		parser.addArgument("--iteration", "-iter").type(Integer.class).setDefault(numIteration).help("The number of iteration.");
		parser.addArgument("--language", "-lang").type(String.class).setDefault(language).help("the language to use");
		parser.addArgument("--trainIDFile","-tid").type(String.class).setDefault(trainIDFile).help("Train id file, default: 600");
		parser.addArgument("--testIDFile","-tstid").type(String.class).setDefault(testIDFile).help("Test id file, default: test");
		parser.addArgument("--saveModel", "-sm").type(Boolean.class).setDefault(saveModel).help("saving the model");
		parser.addArgument("--readModel", "-rm").type(Boolean.class).setDefault(readModel).help("reading the model");
		parser.addArgument("--stopCriteria", "-sc").type(StoppingCriteria.class).setDefault(criteria).help("stopping criteria");
		parser.addArgument("--neural", "-nn").type(NeuralType.class).setDefault(nn).help("neural network type");
		parser.addArgument("-mhws", "--mlpHalfWindowSize").type(Integer.class).setDefault(mlpHalfWindowSize).help("the half window size of the MLP");
		parser.addArgument("--batch", "-b").type(Integer.class).setDefault(NetworkConfig.BATCH_SIZE).help("batch size");
		parser.addArgument("--useBatch", "-ub").type(Boolean.class).setDefault(NetworkConfig.USE_BATCH_TRAINING).help("use batch training");
		parser.addArgument("-optim", "--optimizer").type(String.class).choices("lbfgs", "sgdclip", "adam").setDefault("lbfgs").help("optimizer");
		parser.addArgument("-gi", "--gpuid").type(Integer.class).setDefault(gpuId).help("gpuid");
		parser.addArgument("-emb", "--embedding").type(String.class).choices("glove", "polyglot", "random", "fasttext").setDefault(embedding).help("embedding to use");
		parser.addArgument("-do", "--dropout").type(Double.class).setDefault(dropout).help("dropout rate for the lstm");
		parser.addArgument("-os", "--system").type(String.class).setDefault(NetworkConfig.OS).help("system for lua");
		parser.addArgument("-es", "--embeddingSize").type(Integer.class).setDefault(embeddingSize).help("embedding size");
		parser.addArgument("-hs", "--hiddenSize").type(Integer.class).setDefault(hiddenSize).help("hidden size");
		parser.addArgument("-ed", "--evalDev").type(Boolean.class).setDefault(evalDev).help("evaluate on dev set");
		parser.addArgument("-ef", "--evalFreq").type(Integer.class).setDefault(evalK).help("evaluation frequency");
		parser.addArgument("--type").type(String.class).setDefault(type).help("the type for MLP, can be mlp or bilinear or bilinear-mlp");
		parser.addArgument("-fe", "--fixEmbedding").type(Boolean.class).setDefault(fixEmbedding).help("if fix the embedding");
		parser.addArgument("-uef", "--useEmbFeats").type(Boolean.class).setDefault(useEmbeddingFeature).help("use the embedding value as features");
		parser.addArgument("-aef", "--avgEmbFeats").type(Boolean.class).setDefault(averageEmbFeats).help("average the embedding features or treat them differently");
		parser.addArgument("-el", "--epochLim").type(Integer.class).setDefault(epochLim).help("evaluate the dev set after a number of epochs");
		parser.addArgument("-sgmw", "--saveGMWeight").type(Boolean.class).setDefault(saveGMWeightOnly).help("save graphical model weights only");
		parser.addArgument("-lpw", "--loadPretrainedWeight").type(Boolean.class).setDefault(loadPretrainedWeight).help("load pretrained parameters");
		parser.addArgument("-fpw", "--fixPretrainedWeight").type(Boolean.class).setDefault(fixPretrainedWeight).help("fix the pretrained feature weight");
		parser.addArgument("-dt", "--dotest").type(Boolean.class).setDefault(doTest).help("test the data");
		parser.addArgument("-regn", "--regneural").type(Boolean.class).setDefault(NetworkConfig.REGULARIZE_NEURAL_FEATURES).help("regularizing the neural features");
		parser.addArgument("-mn", "--moreNeural").type(Boolean.class).setDefault(moreNeural).help("add the neural on modifier");
		parser.addArgument("-bf", "--bowFeats").type(Boolean.class).setDefault(bowFeats).help("add bag-of-words features");
		parser.addArgument("-hmf", "--hmFeats").type(Boolean.class).setDefault(hmFeats).help("add head and modifier features");
		parser.addArgument("-sem", "--saveEpochModel").type(Boolean.class).setDefault(saveEpochModel).help("save the models in epoch");
		parser.addArgument("-lem", "--loadEpochModel").type(Boolean.class).setDefault(loadEpochModel).help("load the epcoch model");
		parser.addArgument("-lemn", "--loadEpochModelNum").type(Integer.class).setDefault(loadEpochModelNum).help("load the epoch model num");
		Namespace ns = null;
        try {
            ns = parser.parseArgs(args);
        } catch (ArgumentParserException e) {
            parser.handleError(e);
            System.exit(1);
        }
        numThreads = ns.getInt("thread");
        l2 = ns.getDouble("l2reg");
        numIteration = ns.getInt("iteration");
        language = ns.getString("language");
        trainIDFile = ns.getString("trainIDFile");
        testIDFile = ns.getString("testIDFile");
        saveModel = ns.getBoolean("saveModel");
        readModel = ns.getBoolean("readModel");
        criteria = (StoppingCriteria)ns.get("stopCriteria");
        nn = (NeuralType)ns.get("neural");
        if (nn != NeuralType.none) {
        	NetworkConfig.USE_NEURAL_FEATURES = true;
        	NetworkConfig.REGULARIZE_NEURAL_FEATURES = ns.getBoolean("regneural");
        	NetworkConfig.IS_INDEXED_NEURAL_FEATURES = false;
        }
        NetworkConfig.USE_BATCH_TRAINING = ns.getBoolean("useBatch");
        if (NetworkConfig.USE_BATCH_TRAINING) {
        	NetworkConfig.RANDOM_BATCH = true;
        	NetworkConfig.BATCH_SIZE = ns.getInt("batch");
        }
        gpuId = ns.getInt("gpuid");
        embedding = ns.getString("embedding");
        dropout = ns.getDouble("dropout");
        embeddingSize = ns.getInt("embeddingSize");
        hiddenSize = ns.getInt("hiddenSize");
        NetworkConfig.OS = ns.getString("system");
        evalDev = ns.getBoolean("evalDev");
        evalK = ns.getInt("evalFreq");
        String optim = ns.getString("optimizer");
        switch (optim) {
        	case "lbfgs": optimizer = OptimizerFactory.getLBFGSFactory(); optimStr = "lbfgs"; break;
        	case "sgdclip": optimizer = evalDev ? OptimizerFactory.getGradientDescentFactoryUsingGradientClipping(BestParamCriteria.BEST_ON_DEV, 0.05, 5): 
        				OptimizerFactory.getGradientDescentFactoryUsingGradientClipping(BestParamCriteria.LAST_UPDATE, 0.05, 5);
        				optimStr = "sgdclip";
        							break;
        	case "adam" : optimizer = evalDev ? OptimizerFactory.getGradientDescentFactoryUsingAdaM(BestParamCriteria.BEST_ON_DEV) :
        		OptimizerFactory.getGradientDescentFactoryUsingAdaM(BestParamCriteria.LAST_UPDATE) ; 
        		optimStr = "adam";
        		break;
        	default: optimizer = OptimizerFactory.getLBFGSFactory(); break;
        }
        type = ns.getString("type");
        fixEmbedding = ns.getBoolean("fixEmbedding");
        useEmbeddingFeature = ns.getBoolean("useEmbFeats");
        averageEmbFeats = ns.getBoolean("avgEmbFeats");
        epochLim = ns.getInt("epochLim");
        saveGMWeightOnly = ns.getBoolean("saveGMWeight");
        loadPretrainedWeight = ns.getBoolean("loadPretrainedWeight");
        mlpHalfWindowSize = ns.getInt("mlpHalfWindowSize");
        if (loadPretrainedWeight) {
        	NetworkConfig.LOAD_PRETRAIN_WEIGHT = true;
        	System.out.println("[Info] loading the pretrained weights");
        	String modelFile = "dht_"+language+"_l2_"+l2+"_"+trainIDFile+"_" + criteria.toString() + "_none_none_hidd_0_emb_false.weight";
			ObjectInputStream ois = RAWF.objectReader(modelFolder + modelFile);
			NetworkConfig.pretrainWeights = (double[])ois.readObject();
			ois.close();
        }
        NetworkConfig.FIX_PRETRAIN_WEIGHT = ns.getBoolean("fixPretrainedWeight");
        if (useEmbeddingFeature) {
        	NetworkConfig.USE_FEATURE_VALUE = true;
        	System.err.println("[Info] using embedding as the feature value");
        	if (embedding.equals("polyglot")) {
        		emb = new PolyGlotWordEmbedding("nn-crf-interface/neural_server/"+embedding+"/"+embedding+"-"+language+".txt");
        	} else if (embedding.equals("fasttext")) {
        		emb = new FastTextWordEmbedding("nn-crf-interface/neural_server/fasttext/fasttext-"+language+".txt");
        	}
        }
        doTest = ns.getBoolean("dotest");
        moreNeural = ns.getBoolean("moreNeural");
        bowFeats = ns.getBoolean("bowFeats");
        hmFeats = ns.getBoolean("hmFeats");
        saveEpochModel = ns.getBoolean("saveEpochModel");
        loadEpochModel = ns.getBoolean("loadEpochModel");
        loadEpochModelNum = ns.getInt("loadEpochModelNum");
        for (String key : ns.getAttrs().keySet()) {
        	System.err.println(key + ": " + ns.get(key));
        }
        if (saveModel && readModel) {
        	throw new RuntimeException("cannot save model and read model at the same time.");
        }
	}
}