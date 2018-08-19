package org.statnlp.example.dependency;

import java.util.ArrayList;

import org.statnlp.commons.types.Sentence;
import org.statnlp.example.dependency.DepConfig.Dir;
import org.statnlp.example.dependency.DepNetworkCompiler.NodeType;
import org.statnlp.hypergraph.FeatureArray;
import org.statnlp.hypergraph.FeatureBox;
import org.statnlp.hypergraph.FeatureManager;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkIDMapper;

public class DepFeatureManager extends FeatureManager {


	private static final long serialVersionUID = -8984938159601419422L;
	
	
	public enum FeaType {
		word, tag, shape, prev_word, prev_tag, prev_shape, next_word, next_tag, next_shape,
		 surround_tags, surround_shapes, word_n_gram, left_4, right_4, entity_prefix, entity_suffix,transition,
		unigram, bigram, contextual, inbetween, prefix, joint1, joint2, joint3, joint4
	}
	
	public DepFeatureManager(GlobalNetworkParam param_g) {
		super(param_g);
	}


	@Override
	protected FeatureArray extract_helper(Network network, int parent_k, int[] children_k, int children_k_index) {
		DepInstance inst = (DepInstance)network.getInstance();
		Sentence sent = inst.getInput();
		long node = network.getNode(parent_k);
		int[] nodeArr = NetworkIDMapper.toHybridNodeArray(node);
		int threadId = network.getThreadId();
		
		int pos = nodeArr[0];
		if (pos == 0 || pos == sent.length())
			return FeatureArray.EMPTY;
		FeatureArray start = new FeatureArray(FeatureBox.getFeatureBox(new int[]{}, this.getParams_L()[threadId]));
		FeatureArray fa = start;
		if (nodeArr[4] == NodeType.dep.ordinal()) {
			int leftIndex = nodeArr[0] - nodeArr[1];
			int rightIndex = nodeArr[0];
			int direction = nodeArr[3];
			int completeness = nodeArr[2];
			addDepFeatures(fa, network, sent, leftIndex, rightIndex, completeness, direction, threadId);
		}
		return start;
	}
	
	

	private void addDepFeatures(FeatureArray orgFa, Network network, Sentence sent, int leftIndex, int rightIndex, int completeness, int direction, int threadId) {
		
		ArrayList<Integer> headList = new ArrayList<Integer>();
		ArrayList<Integer> headDistList = new ArrayList<Integer>();
		ArrayList<Integer> modifierList = new ArrayList<Integer>();
		ArrayList<Integer> modifierDistList = new ArrayList<Integer>();
		ArrayList<Integer> pairList = new ArrayList<Integer>();
		ArrayList<Integer> pairDistList = new ArrayList<Integer>();
		
		ArrayList<Integer> bound1 = new ArrayList<Integer>();
		ArrayList<Integer> bound1Dist = new ArrayList<Integer>();
		ArrayList<Integer> bound2 = new ArrayList<Integer>();
		ArrayList<Integer> bound2Dist = new ArrayList<Integer>();
		ArrayList<Integer> bound3 = new ArrayList<Integer>();
		ArrayList<Integer> bound3Dist = new ArrayList<Integer>();
		ArrayList<Integer> bound4 = new ArrayList<Integer>();
		ArrayList<Integer> bound4Dist = new ArrayList<Integer>();
		ArrayList<Integer> inDistList = new ArrayList<Integer>();
		ArrayList<Integer> inList = new ArrayList<Integer>();
		
		String leftTag = sent.get(leftIndex).getTag();
		String rightTag = sent.get(rightIndex).getTag();
		String leftA = leftTag.substring(0, 1);
		String rightA = rightTag.substring(0, 1);
		
		//dist in a span sense sense...
		/**Dist is over span now.**/
		int dist = Math.abs(rightIndex - leftIndex);
		String att = direction == 1 ? "RA" : "LA";
		String distBool = "0";
		if(dist > 1)  distBool = "1";
		if(dist > 2)  distBool = "2";
		if(dist > 3)  distBool = "3";
		if(dist > 4)  distBool = "4";
		if(dist > 5)  distBool = "5";
		if(dist > 10) distBool = "10";
		String attDist = "&" + att + "&" + distBool;
		//maybe we can concatenate the whole span.
		
		if (completeness == 0) {
			int headIndex = direction == Dir.right.ordinal()? leftIndex: rightIndex;
			int modifierIndex = direction == Dir.right.ordinal()? rightIndex : leftIndex;
			String headWord = sent.get(headIndex).getForm();
			String headTag = sent.get(headIndex).getTag();
			String modifierWord = sent.get(modifierIndex).getForm();
			String modifierTag = sent.get(modifierIndex).getTag();
			
			if(headWord.length()>5 || modifierWord.length()>5){
				int hL = headWord.length();
				int mL = modifierWord.length();
				String preHead = hL>5? headWord.substring(0,5):headWord;
				String preModifier = mL>5?modifierWord.substring(0,5):modifierWord;
				pairDistList.add(this._param_g.toFeature(network, FeaType.prefix.name(), "bigram-prefix-all-dist", preHead+","+headTag+","+preModifier+","+modifierTag+attDist));
				pairDistList.add(this._param_g.toFeature(network, FeaType.prefix.name(), "bigram-prefix-word-dist", preHead+","+preModifier+attDist));
				pairList.add(this._param_g.toFeature(network, FeaType.prefix.name(), "bigram-prefix-all", preHead+","+headTag+","+preModifier+","+modifierTag));
				pairList.add(this._param_g.toFeature(network, FeaType.prefix.name(), "bigram-prefix-word", preHead+","+preModifier));
				if(mL>5){
					pairDistList.add(this._param_g.toFeature(network, FeaType.prefix.name(), "modi-prefix-hTmomoT-dist", headTag+","+preModifier+","+modifierTag+attDist));
					pairDistList.add(this._param_g.toFeature(network, FeaType.prefix.name(), "modi-prefix-hTmo-dist", headTag+","+preModifier+attDist));
					pairDistList.add(this._param_g.toFeature(network, FeaType.prefix.name(), "modi-prefix-modiall-dist", preModifier+","+modifierTag+attDist));
					pairDistList.add(this._param_g.toFeature(network, FeaType.prefix.name(), "modi-prefix-modi-dist", preModifier+attDist));
					pairList.add(this._param_g.toFeature(network, FeaType.prefix.name(), "modi-prefix-hTmomoT", headTag+","+preModifier+","+modifierTag));
					pairList.add(this._param_g.toFeature(network, FeaType.prefix.name(), "modi-prefix-hTmo", headTag+","+preModifier));
					pairList.add(this._param_g.toFeature(network, FeaType.prefix.name(), "modi-prefix-modiall", preModifier+","+modifierTag));
					pairList.add(this._param_g.toFeature(network, FeaType.prefix.name(), "modi-prefix-modiall", preModifier));
				}
				if(hL>5){
					pairDistList.add(this._param_g.toFeature(network, FeaType.prefix.name(), "head-prefix-hhTmoT-dist", preHead+","+headTag+","+modifierTag+attDist));
					pairDistList.add(this._param_g.toFeature(network, FeaType.prefix.name(), "head-prefix-hmoT-dist", preHead+","+modifierTag+attDist));
					pairDistList.add(this._param_g.toFeature(network, FeaType.prefix.name(), "head-prefix-hall-dist", preHead+","+headTag+attDist));
					pairDistList.add(this._param_g.toFeature(network, FeaType.prefix.name(), "head-prefix-h-dist", preHead+attDist));
					pairList.add(this._param_g.toFeature(network, FeaType.prefix.name(), "head-prefix-hhTmoT", preHead+","+headTag+","+modifierTag));
					pairList.add(this._param_g.toFeature(network, FeaType.prefix.name(), "head-prefix-hmoT", preHead+","+modifierTag));
					pairList.add(this._param_g.toFeature(network, FeaType.prefix.name(), "head-prefix-hall", preHead+","+headTag));
					pairList.add(this._param_g.toFeature(network, FeaType.prefix.name(), "head-prefix-h", preHead));
				}
			}
			
			/**Unigram feature without dist info**/
			headList.add(this._param_g.toFeature(network, FeaType.unigram.name(), "headword", headWord));
			headList.add(this._param_g.toFeature(network, FeaType.unigram.name(), "headtag", headTag));
			modifierList.add(this._param_g.toFeature(network, FeaType.unigram.name(), "modifierword", modifierWord));
			modifierList.add(this._param_g.toFeature(network, FeaType.unigram.name(), "modifiertag", modifierTag));
			headList.add(this._param_g.toFeature(network, FeaType.unigram.name(), "headwordtag", headWord+","+headTag));
			modifierList.add(this._param_g.toFeature(network, FeaType.unigram.name(), "modifierwordtag", modifierWord+","+modifierTag));
			
			/**Unigram feature with dist info**/
			headDistList.add(this._param_g.toFeature(network, FeaType.unigram.name(), "headword-dist", headWord+attDist));
			headDistList.add(this._param_g.toFeature(network, FeaType.unigram.name(), "headtag-dist", headTag+attDist));
			modifierDistList.add(this._param_g.toFeature(network, FeaType.unigram.name(), "modifierword-dist", modifierWord+attDist));
			modifierDistList.add(this._param_g.toFeature(network, FeaType.unigram.name(), "modifiertag-dist", modifierTag+attDist));
			headDistList.add(this._param_g.toFeature(network, FeaType.unigram.name(), "headwordtag-dist", headWord+","+headTag+attDist));
			modifierDistList.add(this._param_g.toFeature(network, FeaType.unigram.name(), "modifierwordtag-dist", modifierWord+","+modifierTag+attDist));
			
			/****Bigram features without dist info******/
			pairList.add(this._param_g.toFeature(network, FeaType.bigram.name(), "bigramword", headWord+","+modifierWord));
			pairList.add(this._param_g.toFeature(network, FeaType.bigram.name(),"bigramtag", headTag+","+modifierTag));
			pairList.add(this._param_g.toFeature(network, FeaType.bigram.name(), "bigramnametag",  headWord+","+headTag+","+modifierWord+","+modifierTag));
			pairList.add(this._param_g.toFeature(network, FeaType.bigram.name(), "headallmoditag", headWord+","+headTag+","+modifierTag));
			pairList.add(this._param_g.toFeature(network, FeaType.bigram.name(), "headallmodiword", headWord+","+headTag+","+modifierWord));
			pairList.add(this._param_g.toFeature(network, FeaType.bigram.name(), "headtagmodiall", headTag+","+modifierWord+","+modifierTag));
			pairList.add(this._param_g.toFeature(network, FeaType.bigram.name(), "headwordmodiall", headWord+","+modifierWord+","+modifierTag));
			
			/****Bigram features with dist info******/
			pairDistList.add(this._param_g.toFeature(network, FeaType.bigram.name(), "bigramword-dist", headWord+","+modifierWord+attDist));
			pairDistList.add(this._param_g.toFeature(network, FeaType.bigram.name(),"bigramtag-dist", headTag+","+modifierTag+attDist));
			pairDistList.add(this._param_g.toFeature(network, FeaType.bigram.name(), "bigramnametag-dist",  headWord+","+headTag+","+modifierWord+","+modifierTag+attDist));
			pairDistList.add(this._param_g.toFeature(network, FeaType.bigram.name(), "headallmoditag-dist", headWord+","+headTag+","+modifierTag+attDist));
			pairDistList.add(this._param_g.toFeature(network, FeaType.bigram.name(), "headallmodiword-dist", headWord+","+headTag+","+modifierWord+attDist));
			pairDistList.add(this._param_g.toFeature(network, FeaType.bigram.name(), "headtagmodiall-dist", headTag+","+modifierWord+","+modifierTag+attDist));
			pairDistList.add(this._param_g.toFeature(network, FeaType.bigram.name(), "headwordmodiall-dist", headWord+","+modifierWord+","+modifierTag+attDist));
		
			
			
			String leftMinusTag = leftIndex>0? sent.get(leftIndex-1).getTag(): "STR";
			String rightPlusTag = rightIndex<sent.length()-1? sent.get(rightIndex+1).getTag():"END";
			String leftPlusTag = leftIndex<rightIndex-1? sent.get(leftIndex+1).getTag():"MID";
			String rightMinusTag = rightIndex-1 > leftIndex? sent.get(rightIndex-1).getTag():"MID";
			
			String leftMinusA = leftIndex>0? sent.get(leftIndex-1).getATag(): "STR";
			String rightPlusA = rightIndex<sent.length()-1? sent.get(rightIndex+1).getATag():"END";
			String leftPlusA = leftIndex<rightIndex-1? sent.get(leftIndex+1).getATag():"MID";
			String rightMinusA = rightIndex-1 > leftIndex?sent.get(rightIndex-1).getATag():"MID";
			
			//l-1,l,r,r+1
			bound1Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-1-1-dist", leftMinusTag+","+leftTag+","+rightTag+","+rightPlusTag+attDist));
			bound1Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-1-2-dist", leftMinusTag+","+leftTag+","+rightTag+attDist));
			bound1Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-1-3-dist", leftMinusTag+","+rightTag+","+rightPlusTag+attDist));
			bound1Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-1-4-dist", leftMinusTag+","+leftTag+","+rightPlusTag+attDist));
			bound1Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-1-5-dist", leftTag+","+rightTag+","+rightPlusTag+attDist));
			
			bound1.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-1-1", leftMinusTag+","+leftTag+","+rightTag+","+rightPlusTag));
			bound1.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-1-2", leftMinusTag+","+leftTag+","+rightTag));
			bound1.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-1-3", leftMinusTag+","+rightTag+","+rightPlusTag));
			bound1.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-1-4", leftMinusTag+","+leftTag+","+rightPlusTag));
			bound1.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-1-5", leftTag+","+rightTag+","+rightPlusTag));
			
			
			bound1Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-1-1a-dist", leftMinusA+","+leftA+","+rightA+","+rightPlusA+attDist));
			bound1Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-1-2a-dist", leftMinusA+","+leftA+","+rightA+attDist));
			bound1Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-1-3a-dist", leftMinusA+","+rightA+","+rightA+attDist));
			bound1Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-1-4a-dist", leftMinusA+","+leftA+","+rightPlusA+attDist));
			bound1Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-1-5a-dist", leftA+","+rightA+","+rightPlusA+attDist));
			
			bound1.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-1-1a", leftMinusA+","+leftA+","+rightA+","+rightPlusA));
			bound1.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-1-2a", leftMinusA+","+leftA+","+rightA));
			bound1.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-1-3a", leftMinusA+","+rightA+","+rightA));
			bound1.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-1-4a", leftMinusA+","+leftA+","+rightPlusA));
			bound1.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-1-5a", leftA+","+rightA+","+rightPlusA));
			
			//l,l+1,r-1,r
			bound2Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-2-1-dist", leftTag+","+leftPlusTag+","+rightMinusTag+","+rightTag+attDist));
			bound2Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-2-2-dist", leftTag+","+rightMinusTag+","+rightTag+attDist));
			bound2Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-2-3-dist", leftTag+","+leftPlusTag+","+rightTag+attDist));
			bound2Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-2-4-dist", leftPlusTag+","+rightMinusTag+","+rightTag+attDist));
			bound2Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-2-5-dist", leftTag+","+leftPlusTag+","+rightMinusTag+attDist));
			
			bound2.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-2-1", leftTag+","+leftPlusTag+","+rightMinusTag+","+rightTag));
			bound2.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-2-2", leftTag+","+rightMinusTag+","+rightTag));
			bound2.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-2-3", leftTag+","+leftPlusTag+","+rightTag));
			bound2.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-2-4", leftPlusTag+","+rightMinusTag+","+rightTag));
			bound2.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-2-5", leftTag+","+leftPlusTag+","+rightMinusTag));
			
			bound2Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-2-1a-dist", leftA+","+leftPlusA+","+rightMinusA+","+rightA+attDist));
			bound2Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-2-2a-dist", leftA+","+rightMinusA+","+rightA+attDist));
			bound2Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-2-3a-dist", leftA+","+leftPlusA+","+rightA+attDist));
			bound2Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-2-4a-dist", leftPlusA+","+rightMinusA+","+rightA+attDist));
			bound2Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-2-5a-dist", leftA+","+leftPlusA+","+rightMinusA+attDist));
			
			bound2.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-2-1a", leftA+","+leftPlusA+","+rightMinusA+","+rightA));
			bound2.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-2-2a", leftA+","+rightMinusA+","+rightA));
			bound2.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-2-3a", leftA+","+leftPlusA+","+rightA));
			bound2.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-2-4a", leftPlusA+","+rightMinusA+","+rightA));
			bound2.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-2-5a", leftA+","+leftPlusA+","+rightMinusA));
			
			//l-1,l,r-1,r
			//l,l+1,r,r+1
			bound3Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-3-1-dist", leftMinusTag+","+leftTag+","+rightMinusTag+","+rightTag+attDist));
			bound3.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-3-1", leftMinusTag+","+leftTag+","+rightMinusTag+","+rightTag));
			bound3Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-3-1a-dist", leftMinusA+","+leftA+","+rightMinusA+","+rightA+attDist));
			bound3.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-3-1a", leftMinusA+","+leftA+","+rightMinusA+","+rightA));
			
			bound4Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-4-1-dist", leftTag+","+leftPlusTag+","+rightTag+","+rightPlusTag+attDist));
			bound4.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-4-1", leftTag+","+leftPlusTag+","+rightTag+","+rightPlusTag));
			bound4Dist.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-4-1a-dist", leftA+","+leftPlusA+","+rightA+","+rightPlusA+attDist));
			bound4.add(this._param_g.toFeature(network, FeaType.contextual.name(), "contextual-4-1a", leftA+","+leftPlusA+","+rightA+","+rightPlusA));
			
			
			for(int i = leftIndex + 1; i < rightIndex; i++){
				inDistList.add(this._param_g.toFeature(network, FeaType.inbetween.name(), "inbetween-1", leftTag+","+sent.get(i).getTag()+","+rightTag+attDist));
				inList.add(this._param_g.toFeature(network, FeaType.inbetween.name(), "inbetween-2", leftTag+","+sent.get(i).getTag()+","+rightTag));
				inDistList.add(this._param_g.toFeature(network, FeaType.inbetween.name(), "inbetween-3", leftA+","+sent.get(i).getATag()+","+rightA+attDist));
				inList.add(this._param_g.toFeature(network, FeaType.inbetween.name(), "inbetween-4", leftA+","+sent.get(i).getATag()+","+rightA));
			}
		}
		
		ArrayList<ArrayList<Integer>> bigList = new ArrayList<ArrayList<Integer>>();
		bigList.add(headDistList); bigList.add(modifierList);
		bigList.add(modifierDistList); bigList.add(pairList); bigList.add(pairDistList);
		bigList.add(bound1); bigList.add(bound1Dist);
		bigList.add(bound2); bigList.add(bound2Dist);
		bigList.add(bound3); bigList.add(bound3Dist);
		bigList.add(bound4); bigList.add(bound4Dist);
		bigList.add(inList); bigList.add(inDistList);
		FeatureArray fa = orgFa;
		for (int i = 0; i < bigList.size(); i++) {
			FeatureArray curr = addNext(fa, bigList.get(i), threadId);
			fa = curr;
		}
	}
	
	/**
	 * return the next feature array, not the curr
	 * @param fa
	 * @param featureList
	 * @param threadId
	 * @return
	 */
	private FeatureArray addNext(FeatureArray fa, ArrayList<Integer> featureList, int threadId) {
		ArrayList<Integer> finalList = new ArrayList<Integer>();
		for(int i=0;i<featureList.size();i++){
			if(featureList.get(i)!=-1)
				finalList.add( featureList.get(i) );
		}
		if(finalList.size()==0) return fa;
		else {
			int[] fs = new int[finalList.size()];
			for(int i = 0; i < fs.length; i++) fs[i] = finalList.get(i);
			FeatureArray curr = new FeatureArray(FeatureBox.getFeatureBox(fs, this.getParams_L()[threadId]));
			fa.next(curr);
			return curr;
		}
	}
	
	
}
