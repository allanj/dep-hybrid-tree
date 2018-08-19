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

import java.util.ArrayList;
import java.util.Arrays;

import org.statnlp.commons.types.Sentence;
import org.statnlp.example.base.BaseNetwork;
import org.statnlp.hypergraph.FeatureArray;
import org.statnlp.hypergraph.FeatureManager;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkIDMapper;

/**
 * @author wei_lu
 *
 */
public class SemTextFeatureManager_Discriminative extends FeatureManager{
	
	private static final long serialVersionUID = -1424454809943983877L;
	
	private HybridGrammar _g;
	private SemTextDataManager _dm;
	
//	private int _prevWord;
//	private int historySize;
	
	private enum FEATURE_TYPE {emission, transition, pattern};
	
	public SemTextFeatureManager_Discriminative(GlobalNetworkParam param_g, HybridGrammar g, SemTextDataManager dm) {
		super(param_g);
		this._g = g;
		this._dm = dm;
//		historySize = NetworkConfig._SEMANTIC_PARSING_NGRAM-1;
//		if(historySize == 0){
//			_prevWord = 1;
//		} else {
//			_prevWord = historySize;
//		}
	}
	
	@Override
	protected FeatureArray extract_helper(Network network, int parent_k, int[] children_k, int children_k_index) {
		
		BaseNetwork stNetwork = (BaseNetwork)network;
		SemTextInstance inst = (SemTextInstance)stNetwork.getInstance();
		Sentence sent = inst.getInput();
		
		long parent = stNetwork.getNode(parent_k);
		int[] ids_parent = NetworkIDMapper.toHybridNodeArray(parent);
		
		//if it is a leaf, but the pattern is not w.
		if(ids_parent[4]!=0 && children_k.length==0){
			throw new RuntimeException("xxx:"+Arrays.toString(ids_parent));
//			return FeatureArray.NEGATIVE_INFINITY;
		}
		
		if(ids_parent[0]==0){
			return FeatureArray.EMPTY;
		}
		
		//it is the root...
		if(ids_parent[1]==0 && ids_parent[2]==0 && ids_parent[3]==0 && ids_parent[4]==0){
			
			if(children_k.length==0){
				return FeatureArray.EMPTY;
			}
			
			long child = stNetwork.getNode(children_k[0]);
			int[] ids_child = NetworkIDMapper.toHybridNodeArray(child);
			SemanticUnit c_unit = this._dm.getAllUnits().get(ids_child[3]);
			
			int f = this._param_g.toFeature(network, FEATURE_TYPE.transition.name(), "ROOT", c_unit.toString());
			int[] fs = new int[]{f};
			return new FeatureArray(fs);
			
		}
		
		else {
			
			HybridPattern pattern_parent = this._g.getPatternById(ids_parent[4]);

			int eIndex = ids_parent[0];
			int bIndex = ids_parent[0] - ids_parent[1];
			int cIndex = -1;
			
			SemanticUnit p_unit = this._dm.getAllUnits().get(ids_parent[3]);
			SemanticUnit c_units[] = new SemanticUnit[children_k.length];
			
			HybridPattern pattern_children[] = new HybridPattern[children_k.length];
			for(int k = 0; k<children_k.length; k++){
				long child = stNetwork.getNode(children_k[k]);
				int[] ids_child = NetworkIDMapper.toHybridNodeArray(child);
				pattern_children[k] = this._g.getPatternById(ids_child[4]);
				if(k==1){
					cIndex = ids_child[0]-ids_child[1];
				}
				c_units[k] = this._dm.getAllUnits().get(ids_child[3]);
			}
			
			return this.extract_helper(network, p_unit, c_units, pattern_parent, pattern_children, sent, bIndex, cIndex, eIndex);
			
		}
		
	}
	
	private FeatureArray extract_helper(Network network, SemanticUnit p_unit, SemanticUnit[] c_units, 
			HybridPattern pattern_parent, HybridPattern[] pattern_children, 
			Sentence sent, int bIndex, int cIndex, int eIndex){
		
		if(pattern_parent.isw()){
			return FeatureArray.EMPTY;
		}
		
		else if(pattern_parent.isA() || pattern_parent.isB() || pattern_parent.isC()){
			
			if(pattern_children.length!=1){
				throw new RuntimeException("The pattern_children has size "+pattern_children.length);
			}
			
			FeatureArray fa = null;
			
			if(p_unit.isContextIndependent()){
				StringBuilder sb_phrase = new StringBuilder();
				for(int index = bIndex; index<eIndex; index++){
					sb_phrase.append(sent.get(index).getForm());
					sb_phrase.append(' ');
				}
				String phrase = sb_phrase.toString().trim();
				ArrayList<String> phrases = this._dm.getPriorUnitToPhrases(p_unit);
				if(!phrases.contains(phrase)){
//					System.err.println("xx["+phrase+"]");
					return FeatureArray.NEGATIVE_INFINITY;
				} else {
//					System.err.println("YES["+phrase+"]");
					return FeatureArray.EMPTY;
				}
			}
			
			for(int historySize = 0; historySize<=DHTConfig._SEMANTIC_PARSING_NGRAM-1; historySize++)
			{
				int prevWord;
				if(historySize == 0){
					prevWord = 1;
				} else {
					prevWord = historySize;
				}
				
				int[] fs = new int[1+1+prevWord];

				int t = 0;
				
				fs[t++] = this._param_g.toFeature(network, FEATURE_TYPE.pattern.name(), p_unit.toString(), pattern_children[0].toString());
				
				String output, input;
				
				ArrayList<String> wordsInWindow;
				
				wordsInWindow = new ArrayList<String>();
				for(int k = 0; k<historySize; k++){
					String word = this.getForm(pattern_children[0], k-historySize, sent, eIndex-(historySize-k));
					wordsInWindow.add(word);
				}
				
				//single last word.
				output = p_unit.toString();
				for(int k = 0; k<historySize; k++){
					output += wordsInWindow.get(k)+"|||";
				}
				input = "[END]";
				fs[t++] = this._param_g.toFeature(network, FEATURE_TYPE.emission.name(), output, input);

				wordsInWindow = new ArrayList<String>();
				for(int k = 0; k<historySize; k++){
					wordsInWindow.add("[BEGIN]");
				}
				for(int w = 0; w<prevWord; w++){
					//the first _prevWord words
					output = p_unit.toString();
					for(int k = 0; k<historySize; k++){
						output += wordsInWindow.get(k)+"|||";
					}
					input = this.getForm(pattern_children[0], w, sent, bIndex+w);
					fs[t++] = this._param_g.toFeature(network, FEATURE_TYPE.emission.name(), output, input);
					
					wordsInWindow.add(input);
					wordsInWindow.remove(0);
				}
				
				fa = new FeatureArray(fs, fa);
			}
			
			return fa;
		}
		
		else if(pattern_parent.isX()){
			if(c_units[0].isContextIndependent()){
				return FeatureArray.EMPTY;
			}
			int f = this._param_g.toFeature(network, FEATURE_TYPE.transition.name(), p_unit.toString()+":0", c_units[0].toString().toString());
			int[] fs = new int[]{f};
			return new FeatureArray(fs);
		}
		
		else if(pattern_parent.isY()){
			if(c_units[0].isContextIndependent()){
				return FeatureArray.EMPTY;
			}
			int f = this._param_g.toFeature(network, FEATURE_TYPE.transition.name(), p_unit.toString()+":1", c_units[0].toString().toString());
			int[] fs = new int[]{f};
			return new FeatureArray(fs);
		}
		
		else if(pattern_children.length==1){
//			System.err.println(pattern_parent+"\t"+Arrays.toString(pattern_children));
			return FeatureArray.EMPTY;
		}
		
		else {
			
			FeatureArray fa = null;
			
			for(int historySize = 0; historySize<=DHTConfig._SEMANTIC_PARSING_NGRAM-1; historySize++) 
			{
				int prevWord;
				if(historySize == 0){
					prevWord = 1;
				} else {
					prevWord = historySize;
				}
				
				int[] fs = new int[prevWord];
				String output, input;
				
				ArrayList<String> wordsInWindow = new ArrayList<String>();
				for(int k = 0; k<historySize; k++){
//					System.err.println(pattern_parent+"\t"+pattern_children[0]+"\t"+(k-_prevWord)+"\t"+(cIndex-(_prevWord-k))+"\t"+k);
					String word = this.getForm(pattern_children[0], k-historySize, sent, cIndex-(historySize-k));
					wordsInWindow.add(word);
				}
				
				for(int w = 0; w<prevWord; w++){
					//the first _prevWord words
					output = p_unit.toString();
					for(int k = 0; k<historySize; k++){
						output += wordsInWindow.get(k)+"|||";
					}
					
					input = this.getForm(pattern_children[1], w, sent, cIndex+w);
					fs[w] = this._param_g.toFeature(FEATURE_TYPE.emission.name(), output, input);
					
					wordsInWindow.add(input);
					wordsInWindow.remove(0);
				}
				fa = new FeatureArray(fs, fa);
			}
			return fa;
			
		}
		
	}
	
	private String getForm(HybridPattern p, int offset, Sentence sent, int index){
		char c = p.getFormat(offset);
		if(c=='w' || c=='W'){
			return sent.get(index).getForm();
		}
		if(c!='X' && c!='Y'){
			throw new RuntimeException("Invalid:"+p+"\t"+c);
		}
		return "["+c+"]";
	}

}
