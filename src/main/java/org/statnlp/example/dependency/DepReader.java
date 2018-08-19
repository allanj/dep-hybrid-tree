package org.statnlp.example.dependency;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.statnlp.commons.io.RAWF;
import org.statnlp.commons.types.Sentence;
import org.statnlp.commons.types.WordToken;

public class DepReader {

	public static String ROOT_WORD = "ROOT";
	public static String ROOT_TAG = "ROOT";

	/**
	 * Reading the CoNLL-X format data
	 * @param fileName
	 * @param isLabeled
	 * @param number
	 * @param checkProjective: check the projectiveness or not
	 * @param lenOne: means if we restrict all segments are with length 1. The label is still without IOBES encoding
	 *               also it select the head for the span starting from the left portion.
	 * @return
	 * @throws IOException
	 */
	public static DepInstance[] readCoNLLXData(String fileName, boolean isLabeled, int number, boolean checkProjective) throws IOException{
		BufferedReader br = RAWF.reader(fileName);
		ArrayList<DepInstance> result = new ArrayList<DepInstance>();
		ArrayList<WordToken> words = new ArrayList<WordToken>();
		List<Integer> originalHeads = new ArrayList<>();
		int maxSentenceLength = -1;
		words.add(new WordToken(ROOT_WORD, ROOT_TAG));
		originalHeads.add(-1);
		int instanceId = 1;
		int numOfNotTree = 0;
		int nonProjective = 0;
		while(br.ready()){
			String line = br.readLine().trim();
			if(line.length() == 0){
				WordToken[] wtArr = new WordToken[words.size()];
				int[] heads = new int[originalHeads.size()];
				for (int i = 0; i < originalHeads.size(); i++) heads[i] = originalHeads.get(i);
				DepInstance instance = new DepInstance(instanceId, 1.0, new Sentence(words.toArray(wtArr)));
				instance.output = heads;
				
				boolean projectiveness = DataChecker.checkProjective(originalHeads);

				boolean isTree = DataChecker.checkIsTree(originalHeads);
				if (isLabeled) {
					if(!isTree || (checkProjective && !projectiveness)) {
						words = new ArrayList<WordToken>();
						words.add(new WordToken(ROOT_WORD, ROOT_TAG));
						originalHeads = new ArrayList<>();
						originalHeads.add(-1);
						if (!isTree) numOfNotTree++;
						if (checkProjective && !projectiveness) nonProjective++;
						continue;
					}
				}
				if(isLabeled){
					instance.setLabeled(); // Important!
				} else {
					instance.setUnlabeled();
				}
				instanceId++;
				result.add(instance);
				maxSentenceLength = Math.max(maxSentenceLength, instance.size());
				words = new ArrayList<WordToken>();
				words.add(new WordToken(ROOT_WORD, ROOT_TAG));
				originalHeads = new ArrayList<>();
				originalHeads.add(-1);
				if(result.size()==number)
					break;
			} else {
				String[] values = line.split("[\t ]");
				String word = values[1];
				String pos = values[4];
				int headIdx = Integer.parseInt(values[6]);
				String depLabel = values[7];
				if(depLabel.contains("|")) throw new RuntimeException("Mutiple label?");
				words.add(new WordToken(word, pos));
				originalHeads.add(headIdx);
			}
		}
		br.close();
		String type = isLabeled? "train":"test";
		System.err.println("[Info] number of "+type+" instances:"+ result.size());
		System.err.println("[Info] max sentence length: " + maxSentenceLength);
		if (isLabeled) {
			System.err.println("[Info] #not tree: " + numOfNotTree);
			System.err.println("[Info] #non projective: " + nonProjective);
		}
		return result.toArray(new DepInstance[result.size()]);
	}
	
}
