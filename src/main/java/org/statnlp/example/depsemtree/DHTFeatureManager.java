package org.statnlp.example.depsemtree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.AbstractMap.SimpleImmutableEntry;

import org.statnlp.commons.types.Sentence;
import org.statnlp.example.base.BaseNetwork;
import org.statnlp.example.depsemtree.DHTNetworkCompiler.Dir;
import org.statnlp.example.depsemtree.DHTNetworkCompiler.nt;
import org.statnlp.example.depsemtree.DepHybridTree.NeuralType;
import org.statnlp.example.depsemtree.emb.WordEmbedding;
import org.statnlp.hypergraph.FeatureArray;
import org.statnlp.hypergraph.FeatureManager;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkConfig;

public class DHTFeatureManager extends FeatureManager {

	private static final long serialVersionUID = 1L;

	private HybridGrammar _g;
	protected SemTextDataManager _dm;
	
	private enum FT {emission, transition, pattern, head, modifier, bow,headEmb, emitEmb, cheat};
	
	private boolean CHEAT = false;
	private NeuralType nn = NeuralType.none;
	private transient WordEmbedding emb = null; //means using the embedding as feature values
	private boolean bowFeats = true;
	private boolean hmFeats = true;
	
	public DHTFeatureManager(GlobalNetworkParam param_g, HybridGrammar g, SemTextDataManager dm,
			NeuralType nn, WordEmbedding emb,
			boolean bowFeats, boolean hmFeats) {
		super(param_g);
		this._g = g;
		this._dm = dm;
		this.nn = nn;
		this.emb = emb;
		this.bowFeats = bowFeats;
		this.hmFeats = hmFeats;
	}
	
	//eIndex, eIndex - bIndex, direction, hIndex, wIndex, pId, nodeType.ordinal(), k
	@Override
	protected FeatureArray extract_helper(Network network, int parent_k, int[] children_k, int children_k_index) {
		BaseNetwork stNetwork = (BaseNetwork)network;
		SemTextInstance inst = (SemTextInstance)stNetwork.getInstance();
		Sentence sent = inst.getInput();
		int[] ids_parent = network.getNodeArray(parent_k);
		//if it is a leaf, but the pattern is not W.
		if (ids_parent[5] != this._g.getw().getId() && children_k.length == 0) {
			throw new RuntimeException("xxx:"+Arrays.toString(ids_parent));
		}
		
		if (CHEAT) {
			int[] fd = new int[1];
			fd[0] = this._param_g.toFeature(stNetwork, FT.cheat.name(), ids_parent[4] + "", "");
			return this.createFeatureArray(stNetwork, fd);
		}
		
		int paStart = ids_parent[0] - ids_parent[1];
		int paEnd = ids_parent[0];
		Dir paDir = Dir.values()[ids_parent[2]];
		int hIdx = ids_parent[3];
		int wIdx = ids_parent[4];
		int parentPatternId = ids_parent[5];
		nt paNodeType = nt.values()[ids_parent[6]];
//		int paMid =  ids_parent[ids_parent[7]];
		
		if(ids_parent[0]==0){ //end index is 0;
			return FeatureArray.EMPTY;
		}
		if (paDir == Dir.right && 
				hIdx == DHTConfig._SEMANTIC_FOREST_MAX_DEPTH && wIdx == this._dm.getAllUnits().size()) {
			//root node
			if(children_k.length==0){
				return FeatureArray.EMPTY;
			}
			int[] ids_child = stNetwork.getNodeArray(children_k[0]);
			SemanticUnit c_unit = this._dm.getAllUnits().get(ids_child[4]);
			int f = this._param_g.toFeature(network, FT.transition.name(), "ROOT", c_unit.toString());
			int[] fs = new int[]{f};
			return this.createFeatureArray(stNetwork, fs);
		} else {
			HybridPattern pattern_parent = this._g.getPatternById(parentPatternId);
			Dir direction = Dir.values()[ids_parent[2]];
			SemanticUnit p_unit = this._dm.getAllUnits().get(ids_parent[4]);
			SemanticUnit c_units[] = new SemanticUnit[children_k.length];
			HybridPattern pattern_children[] = new HybridPattern[children_k.length];
			nt[] childrenNodeType = new nt[children_k.length];
			int cIndex = -1;
			for(int k = 0; k<children_k.length; k++){
				int[] ids_child = stNetwork.getNodeArray(children_k[k]);
				pattern_children[k] = this._g.getPatternById(ids_child[5]);
				childrenNodeType[k] = nt.values()[ids_child[6]];
				if (ids_child[6] == nt.arc.ordinal()) {
					cIndex = ids_child[7];
				}
				c_units[k] = this._dm.getAllUnits().get(ids_child[4]);
			}
			return this.extract_helper(network, p_unit, c_units, pattern_parent, pattern_children, direction,
					sent, paStart, cIndex, paEnd, parent_k, children_k_index, ids_parent[4], paNodeType, childrenNodeType);
		}
	}

	private FeatureArray extract_helper(Network network, SemanticUnit p_unit, SemanticUnit[] c_units, 
			HybridPattern pattern_parent, HybridPattern[] pattern_children, Dir direction,
			Sentence sent, int bIndex, int cIndex, int eIndex, int parent_k, int children_k_index, int p_unit_id,
			nt parentNodeType, nt[] childrenNodeType){
		
		if (pattern_parent.isw()) {
			int[] fs = new int[1];
			if (bIndex != eIndex) throw new RuntimeException("not equal if pattern is w");
			String word = sent.get(bIndex).getForm();
			fs[0] = this._param_g.toFeature(network,  FT.emission.name(),  p_unit.toString(), word);
//			if (NetworkConfig.USE_NEURAL_FEATURES) {
//				if (this.nn == NeuralType.mlp) {
//					StringBuilder sb = new StringBuilder();
//					for (int rel = -this.mlpHalfWindowSize; rel<=this.mlpHalfWindowSize; rel++) {
//						String w = this.getWord(sent, bIndex + rel);
//						sb.append(rel == -this.mlpHalfWindowSize? w : " " + w);
//					}
//					this.addNeural(network, 0, parent_k, children_k_index, sb.toString(), p_unit_id);
//				} else if (this.nn == NeuralType.lstm) {
//					StringBuilder sb = new StringBuilder();
//					for (int i = 0; i < sent.length(); i++) {
//						String currWord = this.getWord(sent, i);
//						sb.append(i == 0 ? currWord : " " + currWord);
//					}
//					SimpleImmutableEntry<String, Integer> edgeInput = 
//							new SimpleImmutableEntry<String, Integer>(sb.toString(), bIndex);
//					this.addNeural(network, 0, parent_k, children_k_index, edgeInput, p_unit_id);
//				}
//			}
			FeatureArray fa = this.createFeatureArray(network, fs);
//			FeatureArray curr = fa;
//			if (this.emb != null) {
//				String embWord = this.getWord(sent, bIndex);
//				//first try average
//				double[] emitEmb = this.emb.getEmbedding(embWord);
//				double[] fvs = emitEmb;
//				int[] femb = new int[emitEmb.length];
//				for (int p = 0; p < emitEmb.length; p++) {
//					femb[p] = this._param_g.toFeature(network, FT.emitEmb.name(), p_unit.toString(), (p+1)+ "");
//				}
//				curr = curr.addNext(this.createFeatureArray(network, femb, fvs));
//			}
			return fa;
		} else if (pattern_parent.isA() || pattern_parent.isB() || pattern_parent.isC()) {
			if(pattern_children.length!=1){
				throw new RuntimeException("The pattern_children has size "+pattern_children.length);
			}
			int headIdx = direction == Dir.right ? bIndex : eIndex;
			int modIdx = cIndex;
			if (p_unit.isContextIndependent()) {
				String wordPhrase = sent.get(modIdx).getForm().trim();
				ArrayList<String> phrases = this._dm.getPriorUnitToPhrases(p_unit);
				boolean contained = false;
				for (String phrase : phrases) {
					String[] vals = phrase.split(" ");
					if (vals.length == 1) {
						if (phrase.equals(wordPhrase)) contained = true;
					} else {
						for (int i = modIdx - vals.length + 1 >= 1? modIdx - vals.length + 1 : 1; i<= modIdx; i++) {
							int start = i;
							int end = i + vals.length - 1;
							if (!(end < sent.length())) continue;
							StringBuilder candidate_phrase = new StringBuilder();
							for (int w = start;  w<= end; w++) {
								candidate_phrase.append(w == start ? sent.get(w).getForm() : " " + sent.get(w).getForm());
							}
							if (phrase.equals(candidate_phrase.toString())) {
								contained = true;
							}
						}
					}
				}
				if (contained) return FeatureArray.EMPTY;
				else return FeatureArray.NEGATIVE_INFINITY;
			}
			int f = this._param_g.toFeature(network, FT.pattern.name(), p_unit.toString(), pattern_children[0].toString());
			int[] fs = new int[]{f};
			FeatureArray fa = this.createFeatureArray(network, fs);
			FeatureArray curr = fa;
			if (NetworkConfig.USE_NEURAL_FEATURES) {
				if (this.nn == NeuralType.mlp) {
					String headWord = this.getWord(sent, headIdx);
					String moWord = this.getWord(sent, modIdx);
					this.addNeural(network, 0, parent_k, children_k_index, headWord + " " + moWord, p_unit_id);
				} else if (this.nn == NeuralType.lstm) {
					StringBuilder sb = new StringBuilder();
					for (int i = 0; i < sent.length(); i++) {
						String currWord = this.getWord(sent, i);
						sb.append(i == 0 ? currWord : " " + currWord);
					}
					SimpleImmutableEntry<String, Integer> edgeInput = 
							new SimpleImmutableEntry<String, Integer>(sb.toString(), modIdx);
					this.addNeural(network, 0, parent_k, children_k_index, edgeInput, p_unit_id);
				}
			}
			if (this.emb != null) {
				String headWord = this.getWord(sent, headIdx);
				String modWord = this.getWord(sent, modIdx);
				//first try average
				double[] headEmb = this.emb.getEmbedding(headWord);
				double[] modEmb = this.emb.getEmbedding(modWord);
				double[] fvs = new double[headEmb.length];
				int[] femb = new int[headEmb.length];
				for (int p = 0; p < headEmb.length; p++) {
					femb[p] = this._param_g.toFeature(network, FT.headEmb.name(), p_unit.toString(), (p+1)+ "");
					fvs[p] = (headEmb[p] + modEmb[p]) / 2 ;
				}
				curr = curr.addNext(this.createFeatureArray(network, femb, fvs));
			}

			
			List<Integer> fstr = new ArrayList<>();
			int min = Math.min(headIdx, modIdx);
			int max = Math.max(headIdx, modIdx);
			if (this.bowFeats) {
				for (int p = min; p <= max; p++) {
					fstr.add(this._param_g.toFeature(network, FT.bow.name(), p_unit.toString(), sent.get(p).getForm()));
				}
			}
			if (this.hmFeats) {
				fstr.add(this._param_g.toFeature(network, FT.head.name(), p_unit.toString(), sent.get(headIdx).getForm()));
				fstr.add(this._param_g.toFeature(network, FT.modifier.name(), p_unit.toString(), sent.get(modIdx).getForm()));
			}
			curr = curr.addNext(this.createFeatureArray(network, fstr));
			
			return fa;
		} else if (pattern_parent.isX() && (pattern_children[0].isA() || pattern_children[0].isB() || pattern_children[0].isC())) {
			if(c_units[0].isContextIndependent()){
				return FeatureArray.EMPTY;
			}
			int f = this._param_g.toFeature(network, FT.transition.name(), p_unit.toString()+":0", c_units[0].toString());
//			System.out.println(p_unit.toString() + " ,  " + pattern_children[0].getForm() + "  " + c_units[0].toString() );
			return this.createFeatureArray(network, new int[] {f});
		} else if (pattern_parent.isY() && (pattern_children[0].isA() || pattern_children[0].isB() || pattern_children[0].isC()) ) {
			if(c_units[0].isContextIndependent()){
				return FeatureArray.EMPTY;
			}
			int f = this._param_g.toFeature(network, FT.transition.name(), p_unit.toString()+":1", c_units[0].toString());
			return this.createFeatureArray(network, new int[] {f});
		} else if(pattern_children.length == 1){
			//W - > w  usual to usual,
			//W -> W arc to usuual
			//X-> X  arc to usual
			//followings are code to check.
//			if (pattern_parent.getForm().equals("W")) {
//				if (pattern_children[0].getForm().equals("W") && parentNodeType != nt.arc)
//					System.err.println(pattern_parent+"\t"+Arrays.toString(pattern_children));
//				if (!pattern_children[0].getForm().equals("W") && !pattern_children[0].getForm().equals("w")) {
//					System.err.println(pattern_parent+"\t"+Arrays.toString(pattern_children));
//					System.err.println(parentNodeType+"\t"+Arrays.toString(childrenNodeType));
//				}
//			} else if (pattern_parent.getForm().equals("X")) {
//				if (!pattern_children[0].getForm().equals("X"))
//					System.err.println(pattern_parent+"\t"+Arrays.toString(pattern_children));
//			} else {
//				System.err.println(pattern_parent+"\t"+Arrays.toString(pattern_children));
//			}
			
			return FeatureArray.EMPTY;
		} else {
			//mix or arc node. decomposition
			// WW -> W W
			// WX -> W X   XW - > X W, XY - > X Y
			// W - > w w W
			//followings are code to check
//			if (!pattern_parent.getForm().equals("XY") && !pattern_parent.getForm().equals("YX") && !pattern_parent.getForm().equals("XW") && !pattern_parent.getForm().equals("WX") &&
//					!pattern_parent.getForm().equals("WW") && !pattern_parent.getForm().equals("W") )
//				System.err.println(pattern_parent+"\t"+Arrays.toString(pattern_children));
//			if (pattern_parent.isW() && parentNodeType != nt.usual)
//				System.err.println(pattern_parent+"\t"+Arrays.toString(pattern_children));
			//emit word here, at usual w word.
//			List<Integer> fs = new ArrayList<>();
//			if (pattern_parent.isW()) {
//				int wordIdx = direction == Dir.right ? bIndex + 1 : eIndex - 1;
//				if (wordIdx > eIndex || wordIdx < bIndex) return FeatureArray.EMPTY;
//				if (NetworkConfig.USE_NEURAL_FEATURES) {
//					if (this.nn == NeuralType.mlp) {
//						StringBuilder sb = new StringBuilder();
//						for (int p = bIndex - this.mlpHalfWindowSize; p <= bIndex + this.mlpHalfWindowSize; p++) {
//							String currWord = getWord(sent, p);
//							sb.append(p == bIndex - this.mlpHalfWindowSize ? currWord : " " + currWord);
//						}
//						this.addNeural(network, 0, parent_k, children_k_index, sb.toString(), p_unit_id);
//					}
//				} 
//				fs.add(this._param_g.toFeature(network, FT.emission.name(), p_unit.toString(), sent.get(wordIdx).getForm()));
//				return this.createFeatureArray(network, fs);
//			} 
			return FeatureArray.EMPTY;
		}
	}
	
	private String getWord(Sentence sent, int index) {
		int target = index;
		if(target == 0) {
			return "<S>";
		} else if(target == sent.length()) {
			return "</S>";
		} else if(target >= 0 && target < sent.length()) {
			if(sent.get(target).getForm().equals("")) { // for multiple whitespaces..
				return "<UNK>";
			}
			return sent.get(target).getForm();
		} else {
			return "<PAD>";
		}
	}
}
