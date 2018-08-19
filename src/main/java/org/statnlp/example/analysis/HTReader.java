package org.statnlp.example.analysis;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.statnlp.commons.io.RAWF;

public class HTReader {

	
	public static List<ResInstance> readResult(String file) throws IOException {
		List<ResInstance> res = new ArrayList<>();
		BufferedReader br = RAWF.reader(file);
		String line = null;
		String prevLine = null;
		while ((line = br.readLine()) != null) {
			if (line.equals("=INPUT=")) {
				ResInstance inst = new ResInstance();
				String input = br.readLine();
				inst.instanceId=Integer.parseInt(prevLine.split(":\t")[0]);
				inst.input = input;
				line = br.readLine(); //=OUTPUT=
				inst.outputTree = "";
				inst.predictionTree = "";
				line = br.readLine(); //start of output
				while (!line.equals("]")) {
					inst.outputTree += line + "\n";
					line = br.readLine();
				}
				inst.outputTree += line + "\n"; //last ]
				
				line = br.readLine(); //empty line;
				line = br.readLine(); //=PREDICTION=
				line = br.readLine(); //start of prediction
				while (!line.equals("]")) {
					inst.predictionTree += line + "\n";
					line = br.readLine();
				}
				inst.predictionTree += line + "\n";
				line = br.readLine(); //empty line;
				line = br.readLine(); //output;
//				System.out.println(line);
				inst.outputMRL = line.split(":	")[1];
//				System.out.println(inst.outputMRL);
				line = br.readLine(); //prediction;
				inst.predictMRL = line.split(":	")[1];
//				System.out.println(inst.predictMRL);
				res.add(inst);
			}
			prevLine = line;
		}
		br.close();
		Collections.sort(res, Comparator.comparing(ResInstance::getInstanceId));
		System.out.println(res.size());
		for (ResInstance inst : res) {
			System.out.println(inst.instanceId);
		}
		return res;
	}
	
	public static List<ResInstance> readDepResult(String file) throws IOException {
		List<ResInstance> res = new ArrayList<>();
		BufferedReader br = RAWF.reader(file);
		String line = null;
		String prevLine = null;
		while ((line = br.readLine()) != null) {
			if (line.equals("=INPUT=")) {
				ResInstance inst = new ResInstance();
				String input = br.readLine();
				inst.instanceId=Integer.parseInt(prevLine.split(":\t")[0]);
				inst.input = input;
				line = br.readLine(); //=OUTPUT=
				inst.outputTree = "";
				inst.predictionTree = "";
				line = br.readLine(); //start of output
				while (!line.equals("]")) {
					inst.outputTree += line + "\n";
					line = br.readLine();
				}
				inst.outputTree += line + "\n"; //last ]
				
				line = br.readLine(); //empty line;
				line = br.readLine(); //=PREDICTION=
				line = br.readLine(); //start of prediction
				while (!line.equals("]")) {
					inst.predictionTree += line + "\n";
					line = br.readLine();
				}
				inst.predictionTree += line + "\n";
				line = br.readLine(); //empty line;
				line = br.readLine(); //=Dependencies=;
				line = br.readLine(); //start of Dependencies
				inst.dependency = "";
				while (!line.equals("]")) {
					inst.dependency += line + "\n";
					line = br.readLine();
				}
				inst.dependency += line + "\n";
				line = br.readLine(); //output;
//				System.out.println(line);
				inst.outputMRL = line.split(":	")[1];
//				System.out.println(inst.outputMRL);
				line = br.readLine(); //prediction;
				inst.predictMRL = line.split(":	")[1];
//				System.out.println(inst.predictMRL);
				res.add(inst);
			}
			prevLine = line;
		}
		br.close();
		Collections.sort(res, Comparator.comparing(ResInstance::getInstanceId));
		System.out.println(res.size());
		for (ResInstance inst : res) {
			System.out.println(inst.dependency);
		}
		return res;
	}
	
	public static void main(String[] args) throws IOException {
//		readResult("F:\\Dropbox\\SUTD\\Research\\dephybridtree\\results\\compare\\hybridtree_id.log");
		readDepResult("F:\\Dropbox\\SUTD\\Research\\dephybridtree\\results\\dep_baseline_id_test_test_bow_em_0.01.log");
	}
}
