package org.statnlp.util.instance_parser;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

import org.statnlp.commons.types.Label;
import org.statnlp.commons.types.LinearInstance;
import org.statnlp.util.Pipeline;

/**
 * The instance parser to build instances from files in the format similar to CSV or TSV.<br>
 * CoNLL data format can be parsed with this instance parser.
 */
public class DelimiterBasedInstanceParser extends InstanceParser implements Serializable {
	
	private static final long serialVersionUID = -4113323166917321677L;
	
	private boolean locked = false;
	public final Map<String, Label> LABELS = new HashMap<String, Label>();
	public final Map<Integer, Label> LABELS_INDEX = new HashMap<Integer, Label>();
	
	public Label getLabel(String form){
		if(!LABELS.containsKey(form)){
			if(locked){
				throw new IllegalArgumentException("Label "+form+" is not one of the registered labels: "+LABELS.values());
			}
			Label label = new Label(form, LABELS.size());
			LABELS.put(form, label);
			LABELS_INDEX.put(label.getId(), label);
		}
		return LABELS.get(form);
	}
	
	public Label getLabel(int id){
		return LABELS_INDEX.get(id);
	}
	
	public void lock(){
		locked = true;
	}
	
	public boolean locked(){
		return locked;
	}
	
	public void reset(){
		LABELS.clear();
		LABELS_INDEX.clear();
	}
	
	/** The column index which should be regarded as the output label */
	public int labelColumnIndex;
	/** The delimiter when reading the input. This will be parsed as a regex. */
	public String regexDelimiter;
	
	public DelimiterBasedInstanceParser(Pipeline<?> pipeline){
		this(pipeline, "[ \t]+", -1);
	}

	public DelimiterBasedInstanceParser(Pipeline<?> pipeline, String regexDelimiter, int labelColumnIndex) {
		super(pipeline);
		this.regexDelimiter = regexDelimiter;
		this.labelColumnIndex = labelColumnIndex;
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public LinearInstance<Label>[] buildInstances(String... sources) throws FileNotFoundException {
		ArrayList<LinearInstance<Label>> instancesList = new ArrayList<LinearInstance<Label>>();
		int id = 1;
		
		for(String filenameInput: sources){
			ArrayList<String[]> inputArrList = new ArrayList<String[]>();
			ArrayList<Label> labelList = new ArrayList<Label>();
			
			Scanner sc = new Scanner(new File(filenameInput));
			String line;
			boolean hasSeenToken = false;
			while(sc.hasNextLine()){
				line = sc.nextLine().trim();
				if(line.length() == 0){
					instancesList.add(new LinearInstance<Label>(id, 1.0, inputArrList, labelList));
					id += 1;
					inputArrList = new ArrayList<String[]>();
					labelList = new ArrayList<Label>();
					hasSeenToken = false;
				} else {
					hasSeenToken = true;
					String[] tokens = line.split(regexDelimiter);
					String[] inputArr = new String[tokens.length-1];
					int inputArrIdx = 0;
					for(int i=0; i<tokens.length; i++){
						if(i == (labelColumnIndex+tokens.length)%tokens.length){
							Label label = getLabel(tokens[i]);
							labelList.add(label);
						} else {
							inputArr[inputArrIdx] = tokens[i];
							inputArrIdx += 1;
						}
					}
					inputArrList.add(inputArr);
				}
			}
			sc.close();
			if(hasSeenToken){
				instancesList.add(new LinearInstance<Label>(id, 1.0, inputArrList, labelList));
				id += 1;
			}
		}
		return instancesList.toArray(new LinearInstance[instancesList.size()]);
	}

}
