package org.statnlp.util;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.message.Message;
import org.statnlp.commons.types.Instance;
import org.statnlp.commons.types.LinearInstance;
import org.statnlp.example.base.TemplateBasedFeatureManager;
import org.statnlp.hypergraph.FeatureManager;
import org.statnlp.hypergraph.NetworkCompiler;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.hypergraph.NetworkModel;
import org.statnlp.hypergraph.StringIndex;
import org.statnlp.hypergraph.NetworkModel.TrainingIterationInformation;
import org.statnlp.ui.visualize.type.VisualizationViewerEngine;
import org.statnlp.util.instance_parser.DelimiterBasedInstanceParser;
import org.statnlp.util.instance_parser.InstanceParser;

import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.set.TIntSet;
import net.sourceforge.argparse4j.impl.Arguments;
import net.sourceforge.argparse4j.inf.Argument;
import net.sourceforge.argparse4j.inf.ArgumentAction;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;

public class GenericPipeline extends Pipeline<GenericPipeline> {

	public static final Logger LOGGER = GeneralUtils.createLogger(GenericPipeline.class);

	public GenericPipeline() {
		super();
		// Various Paths
		argParserObjects.put("--linearModelClass", argParser.addArgument("--linearModelClass")
				.type(String.class)
				.setDefault("com.statnlp.example.linear_crf.LinearCRF")
				.help("The class name of the model to be loaded (e.g., LinearCRF).\n"
						+ "Note that this generic pipeline assumes linear instances."));
		argParserObjects.put("--useFeatureTemplate", argParser.addArgument("--useFeatureTemplate")
				.type(Boolean.class)
				.action(Arguments.storeTrue())
				.help("Whether to use feature template when extracting features."));
		argParserObjects.put("--featureTemplatePath", argParser.addArgument("--featureTemplatePath")
				.type(String.class)
				.help("The path to feature template file."));
		argParserObjects.put("--trainPath", argParser.addArgument("--trainPath")
				.type(String.class)
				.help("The path to training data."));
		argParserObjects.put("--numTrain", argParser.addArgument("--numTrain")
				.type(Integer.class)
				.help("The number of training data to be taken from the training file."));
		argParserObjects.put("--devPath", argParser.addArgument("--devPath")
				.type(String.class)
				.help("The path to development data"));
		argParserObjects.put("--numDev", argParser.addArgument("--numDev")
				.type(Integer.class)
				.help("The number of development data to be taken from the development file."));
		argParserObjects.put("--testPath", argParser.addArgument("--testPath")
				.type(String.class)
				.help("The path to test data"));
		argParserObjects.put("--numTest", argParser.addArgument("--numTest")
				.type(Integer.class)
				.help("The number of test data to be taken from the test file."));
		argParserObjects.put("--modelPath", argParser.addArgument("--modelPath")
				.type(String.class)
				.help("The path to the model"));
		argParserObjects.put("--logPath", argParser.addArgument("--logPath")
				.type(String.class)
				.action(new ArgumentAction(){

					@Override
					public void run(ArgumentParser parser, Argument arg, Map<String, Object> attrs, String flag,
							Object value) throws ArgumentParserException {
						String logPath = (String)value;
						withLogPath(logPath);
					}

					@Override
					public void onAttach(Argument arg) {}

					@Override
					public boolean consumeArgument() {
						return true;
					}

				})
				.help("The path to log all information related to this pipeline execution."));
		argParserObjects.put("--writeModelAsText", argParser.addArgument("--writeModelAsText")
				.type(Boolean.class)
				.action(Arguments.storeTrue())
				.help("Whether to additionally write the model as text with .txt extension."));
		argParserObjects.put("--resultPath", argParser.addArgument("--resultPath")
				.type(String.class)
				.help("The path to where we should store prediction results."));
		argParserObjects.put("--evaluateEvery", argParser.addArgument("--evaluateEvery")
				.type(Integer.class)
				.setDefault(0)
				.metavar("n")
				.help("Evaluate on development set every n iterations."));
	}

	public GenericPipeline withFeatureTemplate(boolean useFeatureTemplate){
		setFeatureManager(new TemplateBasedFeatureManager(param));
		return getThis();
	}

	public GenericPipeline withFeatureTemplate(boolean useFeatureTemplate, String featureTemplatePath){
		setFeatureManager(new TemplateBasedFeatureManager(param, featureTemplatePath));
		return this;
	}

	/**
	 * With the specified path to training data.
	 * @param trainPath
	 * @return
	 */
	public GenericPipeline withTrainPath(String trainPath){
		setParameter("trainPath", trainPath);
		return getThis();
	}

	/**
	 * With the specified path to devlopment data.
	 * @param devPath
	 * @return
	 */
	public GenericPipeline withDevPath(String devPath){
		setParameter("devPath", devPath);
		return getThis();
	}

	/**
	 * With the specified path to test data.
	 * @param testPath
	 * @return
	 */
	public GenericPipeline withTestPath(String testPath){
		setParameter("testPath", testPath);
		return getThis();
	}

	/**
	 * With the specified number of training data used.<br>
	 * If the number is more than the number of actual training data read from trainPath,
	 * all training data is used. 
	 * @param numTrain
	 * @return
	 */
	public GenericPipeline withNumTrain(int numTrain){
		setParameter("numTrain", numTrain);
		return getThis();
	}

	/**
	 * With the specified number of development data used.<br>
	 * If the number is more than the number of actual development data read from devPath,
	 * all development data is used. 
	 * @param numDev
	 * @return
	 */
	public GenericPipeline withNumDev(int numDev){
		setParameter("numDev", numDev);
		return getThis();
	}

	/**
	 * With the specified number of test data used.<br>
	 * If the number is more than the number of actual test data read from testPath,
	 * all test data is used. 
	 * @param numTest
	 * @return
	 */
	public GenericPipeline withNumTest(int numTest){
		setParameter("numTest", numTest);
		return getThis();
	}

	/**
	 * With the specified path to model.
	 * @param modelPath
	 * @return
	 */
	public GenericPipeline withModelPath(String modelPath){
		setParameter("modelPath", modelPath);
		return getThis();
	}

	/**
	 * With the specified path to the log file.
	 * @param logPath
	 * @return
	 */
	public GenericPipeline withLogPath(String logPath) {
		setParameter("logPath", logPath);
		try{
			GeneralUtils.updateLogger(logPath);
		} catch (FileNotFoundException e){
			throw new RuntimeException(e);
		}
		return getThis();
	}

	/**
	 * With the specified path to the result file.
	 * @param resultPath
	 * @return
	 */
	public GenericPipeline withResultPath(String resultPath){
		setParameter("resultPath", resultPath);
		return getThis();
	}

	/**
	 * Whether to also write the learned model as text, showing the feature list and the corresponding weights.
	 * @param writeModelAsText
	 * @return
	 */
	public GenericPipeline withWriteModelAsText(boolean writeModelAsText){
		setParameter("writeModelAsText", writeModelAsText);
		return getThis();
	}

	/**
	 * With evaluation on development data every specified number of iterations.
	 * @param evaluateEvery
	 * @return
	 */
	public GenericPipeline withEvaluateEvery(int evaluateEvery){
		setParameter("evaluateEvery", evaluateEvery);
		return getThis();
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#initInstanceParser()
	 */
	@Override
	protected InstanceParser initInstanceParser() {
		if(instanceParser == null){
			if(instanceParserClass != null){
				try {
					return instanceParserClass.getConstructor(Pipeline.class).newInstance(this);
				} catch (InstantiationException | IllegalAccessException | IllegalArgumentException
						| InvocationTargetException | NoSuchMethodException | SecurityException e) {
					try {
						return instanceParserClass.newInstance();
					} catch (InstantiationException | IllegalAccessException e1) {
						LOGGER.fatal("[%s]Instance parser class name %s can neither be instantiated with Pipeline "
								+ "object as the argument, nor does it have a public empty constructor.",
								getCurrentTask(), instanceParserClass);
						throw new RuntimeException(LOGGER.throwing(Level.FATAL, e));
					}
				}
			}
			return new DelimiterBasedInstanceParser(this);
		} else {
			return instanceParser;
		}
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#initNetworkCompiler()
	 */
	@SuppressWarnings("unchecked")
	@Override
	protected NetworkCompiler initNetworkCompiler() {
		if(networkCompiler != null){
			return networkCompiler;
		}
		String currentTask = getCurrentTask();
		if(networkModel != null){
			return networkModel.getNetworkCompiler();
		} else {
			boolean failEarly = true;
			if(!currentTask.equals(TASK_TRAIN) && !currentTask.equals(TASK_VISUALIZE)){
				LOGGER.warn("[%s]No model has been loaded, cannot load network compiler from model, "
						+ "attempting to create a network compiler now.", currentTask);
				failEarly = false;
			}
			if(networkCompilerClass == null){
				String linearModelClassName = getParameter("linearModelClass");
				String networkCompilerClassName = linearModelClassName+"NetworkCompiler";
				try {
					networkCompilerClass = (Class<? extends NetworkCompiler>)Class.forName(networkCompilerClassName);
				} catch (ClassNotFoundException e) {
					Message message = LOGGER.getMessageFactory().newMessage(
							"[%s]Network compiler class name cannot be inferred from model class name %s",
							currentTask, linearModelClassName);
					if(failEarly){
						LOGGER.fatal(message);
						throw new RuntimeException(LOGGER.throwing(Level.FATAL, e));
					} else {
						LOGGER.warn(message);
						LOGGER.throwing(Level.WARN, e);
						LOGGER.warn("[%s]Network compiler creation failed, no network compiler was created.", currentTask);
						return null;
					}
				}
			}
			try {
				return (NetworkCompiler)networkCompilerClass.getConstructor(Pipeline.class).newInstance(this);
			} catch (InstantiationException | IllegalAccessException | IllegalArgumentException |
					InvocationTargetException | NoSuchMethodException | SecurityException e) {
				try {
					return (NetworkCompiler)networkCompilerClass.newInstance();
				} catch (InstantiationException | IllegalAccessException e1) {
					Message message = LOGGER.getMessageFactory().newMessage("[%s]Network compiler class %s can neither be instantiated with Pipeline "
							+ "object as the argument, nor does it have a public empty constructor.",
							currentTask, networkCompilerClass.getName());
					if(failEarly){
						LOGGER.fatal(message);
						throw new RuntimeException(LOGGER.throwing(Level.FATAL, e));
					} else {
						LOGGER.warn(message);
						LOGGER.throwing(Level.WARN, e);
						LOGGER.warn("[%s]Network compiler creation failed, no network compiler was created.", currentTask);
						return null;
					}
				}
			}
		}
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#initFeatureManager()
	 */
	@SuppressWarnings("unchecked")
	@Override
	protected FeatureManager initFeatureManager() {
		if(featureManager != null){
			return featureManager;
		}
		String currentTask = getCurrentTask();
		if(networkModel != null){
			return networkModel.getFeatureManager();
		} else {
			boolean failEarly = true;
			if(!currentTask.equals(TASK_TRAIN) && !currentTask.equals(TASK_VISUALIZE)){
				LOGGER.warn("[%s]No model has been loaded, cannot load feature manager from model, "
						+ "attempting to create a feature manager now.", currentTask);
				failEarly = false;
			}
			if(featureManagerClass == null){
				if(hasParameter("useFeatureTemplate") && (boolean)getParameter("useFeatureTemplate")){
					if(hasParameter("featureTemplatePath")){
						LOGGER.info("[%s]Using template-based feature manager with templates from %s.",
								currentTask, (String)getParameter("featureTemplatePath"));
						return new TemplateBasedFeatureManager(param, (String)getParameter("featureTemplatePath"));	
					} else {
						LOGGER.info("[%s]Using template-based feature manager with default template.", currentTask);
						return new TemplateBasedFeatureManager(param);
					}
				}
				String linearModelClassName = getParameter("linearModelClass");
				String featureManagerClassName = linearModelClassName+"FeatureManager";
				try {
					featureManagerClass = (Class<? extends FeatureManager>)Class.forName(featureManagerClassName);
				} catch (ClassNotFoundException e) {
					Message message = LOGGER.getMessageFactory().newMessage(
							"[%s]Feature manager class name cannot be inferred from model class name %s",
							currentTask, linearModelClassName);
					if(failEarly){
						LOGGER.fatal(message);
						throw new RuntimeException(LOGGER.throwing(Level.FATAL, e));
					} else {
						LOGGER.warn(message);
						LOGGER.throwing(Level.WARN, e);
						LOGGER.warn("[%s]Feature manager creation failed, no feature manager was created.", currentTask);
						return null;
					}
				} 
			} else {
				if(hasParameter("useFeatureTemplate") && (boolean)getParameter("useFeatureTemplate")){
					LOGGER.warn("[%s]Both useFeatureTemplate and featureManagerClass are specified. "
							+ "Using the specified featureManagerClass.", currentTask);
				}
			}
			try {
				return featureManagerClass.getConstructor(Pipeline.class).newInstance(this);
			} catch (InstantiationException | IllegalAccessException | IllegalArgumentException |
					InvocationTargetException | NoSuchMethodException | SecurityException e) {
				try {
					return featureManagerClass.newInstance();
				} catch (InstantiationException | IllegalAccessException e1) {
					Message message = LOGGER.getMessageFactory().newMessage(
							"[%s]Feature manager class %s can neither be instantiated with Pipeline "
							+ "object as the argument, nor does it have a public empty constructor.",
							currentTask, featureManagerClass.getName());
					if(failEarly){
						LOGGER.fatal(message);
						throw new RuntimeException(LOGGER.throwing(Level.FATAL, e));
					} else {
						LOGGER.warn(message);
						LOGGER.throwing(Level.WARN, e);
						LOGGER.warn("[%s]Feature manager creation failed, no feature manager was created.", currentTask);
						return null;
					}
				}
			}
		}
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#saveModel()
	 */
	@Override
	protected void saveModel() throws IOException {
		String modelPath = getParameter("modelPath");
		if(modelPath == null){
			if(getCurrentTask() == TASK_SAVE_MODEL){
				throw LOGGER.throwing(Level.ERROR, new RuntimeException("["+getCurrentTask()+"]Saving model requires --modelPath to be set."));
			} else {
				LOGGER.warn("[%s]Not saving trained model, since --modelPath is not set.", getCurrentTask());
				return;
			}
		}
		LOGGER.info("[%s]Writing model into %s...", getCurrentTask(), modelPath);
		long startTime = System.nanoTime();
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelPath));
		oos.writeObject(networkModel);
		oos.writeObject(instanceParser);
		oos.close();
		long endTime = System.nanoTime();
		LOGGER.info("[%s]Writing model...Done in %.3fs", getCurrentTask(), (endTime-startTime)/1.0e9);
		if((boolean)getParameter("writeModelAsText")){
			String modelTextPath = modelPath+".txt";
			try{
				LOGGER.info("[%s]Writing model text into %s...", getCurrentTask(), modelTextPath);
				PrintStream modelTextWriter = new PrintStream(modelTextPath);
				modelTextWriter.println(NetworkConfig.getConfig());
//				modelTextWriter.println("Labels:");
//				List<Label> labelsUsed = new ArrayList<Label>(param.LABELS.values());
//				Collections.sort(labelsUsed);
//				modelTextWriter.println(labelsUsed);
				modelTextWriter.println("Num features: "+param.countFeatures());
				modelTextWriter.println("Features:");
				TIntObjectHashMap<TIntObjectHashMap<TIntIntHashMap>> featureIntMap = param.getFeatureIntMap();
				StringIndex stringIndex = param.getStringIndex();
				stringIndex.buildReverseIndex();
				for(String featureType: sorted(stringIndex, featureIntMap.keySet())){
					modelTextWriter.println(featureType);
					TIntObjectHashMap<TIntIntHashMap> outputInputMap = featureIntMap.get(stringIndex.get(featureType));
					for(String output: sorted(stringIndex, outputInputMap.keySet())){
						modelTextWriter.println("\t"+output);
						TIntIntHashMap inputMap = outputInputMap.get(stringIndex.get(output));
						for(String input: sorted(stringIndex, inputMap.keySet())){
							int featureId = inputMap.get(stringIndex.get(input));
							modelTextWriter.printf("\t\t%s %d %.17f\n", input, featureId, param.getWeight(featureId));
						}
					}
				}
				stringIndex.removeReverseIndex();
				modelTextWriter.close();
			} catch (IOException e){
				LOGGER.warn("[%s]Cannot write model text into %s.", getCurrentTask(), modelTextPath);
				LOGGER.throwing(Level.WARN, e);
			}
		}
	}

	private static List<String> sorted(StringIndex stringIndex, TIntSet coll){
		List<String> result = new ArrayList<String>(coll.size());
		for(int key: coll.toArray()){
			result.add(stringIndex.get(key));
		}
		Collections.sort(result);
		return result;
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#handleSaveModelError(java.lang.Exception)
	 */
	@Override
	protected void handleSaveModelError(Exception e) {
		LOGGER.warn("[%s]Cannot save model to %s", getCurrentTask(), (String)getParameter("modelPath"));
		LOGGER.throwing(Level.WARN, e);
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#loadModel()
	 */
	@Override
	protected void loadModel() throws IOException {
		String currentTask = getCurrentTask();
		if(networkModel != null && (currentTask.equals(TASK_TEST) || currentTask.equals(TASK_TUNE))){
			LOGGER.info("[%s]Model already loaded, using loaded model.", getCurrentTask());
		} else {
			String modelPath = getParameter("modelPath");
			if(modelPath == null){
				throw LOGGER.throwing(Level.ERROR, new RuntimeException("["+getCurrentTask()+"]Loading model requires --modelPath to be set."));
			}
			LOGGER.info("Reading model from %s...", modelPath);
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelPath));
			long startTime = System.nanoTime();
			try {
				networkModel = (NetworkModel)ois.readObject();
				instanceParser = (InstanceParser)ois.readObject();
			} catch (ClassNotFoundException e) {
				LOGGER.warn("[%s]Cannot load the model from %s", getCurrentTask(), modelPath);
				throw new RuntimeException(LOGGER.throwing(Level.FATAL, e));
			} finally {
				ois.close();
			}
			long endTime = System.nanoTime();
			LOGGER.info("[%s]Reading model...Done in %.3fs", getCurrentTask(), (endTime-startTime)/1.0e9);
		}
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#handleLoadModelError(java.lang.Exception)
	 */
	@Override
	protected void handleLoadModelError(Exception e) {
		LOGGER.error("[%s]Cannot load model from %s", getCurrentTask(), (String)getParameter("modelPath"));
		throw new RuntimeException(LOGGER.throwing(Level.ERROR, e));
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#initTraining(java.lang.String[])
	 */
	@Override
	protected void initTraining() {
		// TODO Auto-generated method stub

	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#initTuning()
	 */
	@Override
	protected void initTuning() {
		// TODO Auto-generated method stub

	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#initTesting()
	 */
	@Override
	protected void initTesting() {
		// TODO Auto-generated method stub

	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#initEvaluation(java.lang.String[])
	 */
	@Override
	protected void initEvaluation() {
		// TODO Auto-generated method stub

	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#initVisualization()
	 */
	@Override
	protected void initVisualization() {
		if(instanceParser == null){
			if(hasParameter("trainPath")){
				initTraining();
			} else if(hasParameter("devPath")){
				initTuning();
			} else if(hasParameter("testPath")){
				initTesting();
			} else {
				throw LOGGER.throwing(Level.ERROR, new RuntimeException("["+getCurrentTask()+"]Visualization requires one of --trainPath, --devPath, or --testPath to be specified."));
			}
		}
		initGlobalNetworkParam();

		initAndSetInstanceParser();

		if(hasParameter("trainPath")){
			getInstancesForTraining();
		} else if(hasParameter("devPath")){
			getInstancesForTuning();
		} else if(hasParameter("testPath")){
			getInstancesForTesting();
		}

		initAndSetNetworkCompiler();
		initAndSetFeatureManager();

		initNetworkModel();

	}

	public Instance[] getInstancesForTraining(){
		if(hasParameter("trainInstances")){
			return getParameter("trainInstances");
		}
		if(!hasParameter("trainPath")){
			throw LOGGER.throwing(Level.ERROR, new RuntimeException(String.format("["+getCurrentTask()+"]The task %s requires --trainPath to be specified.", getCurrentTask())));
		}
		try {
			Instance[] trainInstances = instanceParser.buildInstances((String)getParameter("trainPath"));
			if(hasParameter("numTrain")){
				int numTrain = getParameter("numTrain");
				if(numTrain > 0){
					numTrain = Math.min(trainInstances.length, numTrain);
					trainInstances = Arrays.copyOfRange(trainInstances, 0, numTrain);
				}
			}
			setParameter("trainInstances", trainInstances);
			return trainInstances;
		} catch (FileNotFoundException e) {
			throw new RuntimeException(e);
		}
	}

	public Instance[] getInstancesForTuning(){
		if(hasParameter("devInstances")){
			return getParameter("devInstances");
		}
		if(!hasParameter("devPath")){
			throw LOGGER.throwing(Level.ERROR, new RuntimeException(String.format("["+getCurrentTask()+"]The task %s requires --devPath to be specified.", getCurrentTask())));
		}
		try {
			Instance[] devInstances = instanceParser.buildInstances((String)getParameter("devPath"));
			if(hasParameter("numDev")){
				int numDev = getParameter("numDev");
				if(numDev > 0){
					numDev = Math.min(devInstances.length, numDev);
					devInstances = Arrays.copyOfRange(devInstances, 0, numDev);
				}
			}
			setParameter("devInstances", devInstances);
			return devInstances;
		} catch (FileNotFoundException e) {
			throw new RuntimeException(e);
		}
	}

	public Instance[] getInstancesForTesting(){
		if(hasParameter("testInstances")){
			return getParameter("testInstances");
		}
		if(!hasParameter("testPath")){
			throw LOGGER.throwing(Level.ERROR, new RuntimeException(String.format("["+getCurrentTask()+"]The task %s requires --testPath to be specified.", getCurrentTask())));
		}
		try {
			Instance[] testInstances = instanceParser.buildInstances((String)getParameter("testPath"));
			if(hasParameter("numTest")){
				int numTest = getParameter("numTest");
				if(numTest > 0){
					numTest = Math.min(testInstances.length, numTest);
					testInstances = Arrays.copyOfRange(testInstances, 0, numTest);
				}
			}
			setParameter("testInstances", testInstances);
			return testInstances;
		} catch (FileNotFoundException e) {
			throw new RuntimeException(e);
		}
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#getInstancesForEvaluation()
	 */
	@Override
	public Instance[] getInstancesForEvaluation() {
		return getInstancesForTesting();
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#getInstancesForEvaluation()
	 */
	@Override
	public Instance[] getInstancesForVisualization() {
		Instance[] result = getParameter("trainInstances");
		if(result == null){
			result = getParameter("devInstances");
		}
		if(result == null){
			result = getParameter("testInstances");
		}
		if(result == null){
			throw LOGGER.throwing(new RuntimeException("["+getCurrentTask()+"]Cannot find instances to be visualized. "
					+ "Please specify them through --trainPath, --devPath, or --testPath."));
		}
		return result;
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#tune()
	 */
	@Override
	protected void train(Instance[] trainInstances) {
		if(hasParameter("evaluateEvery")){
			int evaluateEvery = getParameter("evaluateEvery");
			if(evaluateEvery > 0){
				networkModel.setEndOfIterCallback(new Consumer<TrainingIterationInformation>(){

					@Override
					public void accept(TrainingIterationInformation t) {
						int iterNum = t.iterNum;
						if((iterNum+1) % evaluateEvery == 0){
							Instance[] instances = getInstancesForTuning();
							for(int k = 0; k < instances.length; k++){
								instances[k].setUnlabeled();
							}

							try {
								instances = networkModel.decode(instances, true);
							} catch (InterruptedException e) {}
							evaluate(instances);
						}
					}

				});
			}
		}
		long duration = System.nanoTime();
		try {
			networkModel.train(trainInstances, getParameter("maxIter"));
			duration = System.nanoTime() - duration;
		} catch (InterruptedException e) {
			throw LOGGER.throwing(new RuntimeException(e));
		}
		LOGGER.info("[%s]Total training time: %.3fs\n", getCurrentTask(), duration/1.0e9);
	}
	
	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#tune()
	 */
	@Override
	protected void tune(Instance[] devInstances) {

	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#test()
	 */
	@Override
	protected void test(Instance[] testInstances) {
		long duration = System.nanoTime();
		try {
			Instance[] instanceWithPredictions = networkModel.decode(testInstances, (int)getParameter("predictTopK"));
			duration = System.nanoTime() - duration;
			setParameter("testInstances", instanceWithPredictions);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		LOGGER.info("[%s]Total testing time: %.3fs\n", getCurrentTask(), duration/1.0e9);		
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#evaluationResult(com.statnlp.commons.types.Instance[])
	 */
	@Override
	protected void evaluate(Instance[] instancesWithPrediction) {
		int corr = 0;
		int total = 0;
		for(Instance instance: instancesWithPrediction){
			LinearInstance<?> linInstance = (LinearInstance<?>)instance;
			try{
				corr += linInstance.countNumCorrectlyPredicted();
			} catch (IndexOutOfBoundsException e){
				throw new RuntimeException("IndexOutOfBoundsException occurred. "
						+ "This is usually caused by different number of predictions "
						+ "compared to gold. The default evaluation procedure assumes tagging task, "
						+ "with the same number of predictions. You can create custom evaluation procedure "
						+ "by either overriding the evaluate(Instance[]) function in a subclass of GenericPipeline, "
						+ "or supplying the evaluateCallback function through setEvaluateCallback.", e);
			}
			total += linInstance.size();
		}
		LOGGER.info("[%s]Correct/Total: %d/%d", getCurrentTask(), corr, total);
		LOGGER.info("[%s]Accuracy: %.2f%%", getCurrentTask(), 100.0*corr/total);
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#visualize(com.statnlp.commons.types.Instance[])
	 */
	@SuppressWarnings("unchecked")
	@Override
	protected void visualize(Instance[] instances) {
		if(visualizerClass == null){
			String visualizerModelName = getParameter("linearModelClass")+"Viewer";
			try{
				visualizerClass = (Class<VisualizationViewerEngine>)Class.forName(visualizerModelName);
			} catch (ClassNotFoundException e) {
				LOGGER.warn("[%s]Cannot automatically find viewer class for model name %s", getCurrentTask(), (String)getParameter("linearModelClass"));
				LOGGER.throwing(Level.WARN, e);
				return;
			}
		}
		try {
			networkModel.visualize(visualizerClass, instances);
		} catch (InterruptedException e) {
			LOGGER.info("[%s]Visualizer was interrupted.", getCurrentTask());
		}     
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#savePredictions()
	 */
	@Override
	protected void savePredictions() {
		if(!hasParameter("resultPath")){
			LOGGER.warn("[%s]Task savePredictions requires --resultPath to be specified.", getCurrentTask());
			return;
		}
		String resultPath = getParameter("resultPath");
		try{
			PrintWriter printer = new PrintWriter(new File(resultPath));
			Instance[] instances = getParameter("testInstances");
			for(Instance instance: instances){
				printer.println(instance.toString());
			}
			printer.close();
		} catch (FileNotFoundException e){
			LOGGER.warn("[%s]Cannot find file %s for storing prediction results.", getCurrentTask(), resultPath);
		}
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#extractFeatures(com.statnlp.commons.types.Instance[])
	 */
	@Override
	protected void extractFeatures(Instance[] instances) {
		// TODO Auto-generated method stub
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#writeFeatures(com.statnlp.commons.types.Instance[])
	 */
	@Override
	protected void writeFeatures(Instance[] instances) {
		// TODO Auto-generated method stub

	}

	public void initExecute(){
		boolean hasWarning = false;
		if(taskList.contains(TASK_TRAIN)){
			if(!hasParameter("trainPath")){
				LOGGER.warn("train task is specified but --trainPath is missing.");
				hasWarning = true;
			}
			if(!hasParameter("modelPath")){
				LOGGER.warn("Trained model will not be saved since --modelPath is missing.");
				hasWarning = true;
			}
		}
		if(taskList.contains(TASK_TUNE)){
			if(!hasParameter("devPath")){
				LOGGER.warn("tune task is specified but --devPath is missing.");
				hasWarning = true;
			}
		}
		if(taskList.contains(TASK_TEST)){
			if(!hasParameter("testPath")){
				LOGGER.warn("test task is specified but --testPath is missing.");
				hasWarning = true;
			}
		}
		if(hasWarning){
			try{Thread.sleep(2000);}catch(InterruptedException e){}
		}
		super.initExecute();
	}

}
