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

import org.statnlp.commons.types.Instance;
import org.statnlp.commons.types.Sentence;
import org.statnlp.example.base.BaseNetwork;
import org.statnlp.example.base.BaseNetwork.NetworkBuilder;
import org.statnlp.hypergraph.LocalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkCompiler;
import org.statnlp.hypergraph.NetworkIDMapper;

/**
 * @author wei_lu
 *
 */
public class SemTextNetworkCompiler extends NetworkCompiler{
	
	private static final long serialVersionUID = -2627518568805384558L;
	
	private HybridGrammar _g;
	private SemTextDataManager _dm;
	private SemanticForest _forest;//the global forest.
	
	private int _maxSentLen = 24;
	private long[] _nodes;
	private int[][][] _children;
	private int[] _numNodesInSubStructure = new int[this._maxSentLen+1];
	
	public SemTextNetworkCompiler(HybridGrammar g, SemanticForest forest, SemTextDataManager dm){
		
		this._g = g;
		this._dm = dm;
		this._forest = forest;
		System.out.println("total forest node:" + this._forest.getAllNodes().size());
		int[] capacity = new int[]{300, 300, 1000, 1000, 1000, 2};
		NetworkIDMapper.setCapacity(capacity);
		
	}
	
	
	public BaseNetwork compileLabeled(int networkId, Instance ins, LocalNetworkParam param){
		SemTextInstance inst = (SemTextInstance)ins;
		NetworkBuilder<BaseNetwork> builder = NetworkBuilder.builder();
		Sentence sent = inst.getInput();
		SemanticForest tree = inst.getOutput();
		return this.compile(builder, sent, tree, networkId, inst,param);
		
	}
	
	private BaseNetwork compile(NetworkBuilder<BaseNetwork> builder, Sentence sent, SemanticForest tree,
			int networkId, SemTextInstance inst, LocalNetworkParam param){
		
		System.err.print('+');
		
//		for(SemanticForestNode forestNode : tree.getAllNodes()){
//			System.out.println(forestNode.isRoot());
//			System.out.println(forestNode);
//		}
		for(int eIndex=1; eIndex<=sent.length(); eIndex++){
			
			for(int L = 1; L<=eIndex; L++){
				int bIndex = eIndex - L;
				//[bIndex, eIndex)
				for(SemanticForestNode forestNode : tree.getAllNodes()){
					
					if(forestNode.isRoot())
						continue;
					
					if(eIndex == bIndex+1){
						long node = this.toNode(bIndex, eIndex, forestNode, this._g.getw());
						builder.addNode(node);
					}
					
					if(forestNode.arity()==1){
						SemanticForestNode[] childTreeNodes0 = forestNode.getChildren()[0];
						
						long node_X = this.toNode(bIndex, eIndex, forestNode, this._g.getX());
						boolean added = false;
						for(SemanticForestNode childForestNode : childTreeNodes0){
							long node_child = this.toNode(bIndex, eIndex, childForestNode, this._g.getRootPatternByArity(childForestNode.arity()));
							if(builder.contains(node_child)){
								if(!added){
									builder.addNode(node_X);
									added = true;
								}
								builder.addEdge(node_X, new long[]{node_child});
							}
						}
					} else if(forestNode.arity()==2){
						SemanticForestNode[] childTreeNodes0 = forestNode.getChildren()[0];
						SemanticForestNode[] childTreeNodes1 = forestNode.getChildren()[1];
						
						long node_X = this.toNode(bIndex, eIndex, forestNode, this._g.getX());
						boolean addedX = false;
						for(SemanticForestNode childForestNode : childTreeNodes0){
							long node_child = this.toNode(bIndex, eIndex, childForestNode, this._g.getRootPatternByArity(childForestNode.arity()));
//							System.out.println(this._g.getRootPatternByArity(childForestNode.arity()));
							if(builder.contains(node_child)){
								if(!addedX){
									builder.addNode(node_X);
									addedX = true;
								}
								builder.addEdge(node_X, new long[]{node_child});
							}
						}
						
						long node_Y = this.toNode(bIndex, eIndex, forestNode, this._g.getY());
						boolean addedY = false;
						for(SemanticForestNode childForestNode : childTreeNodes1){
							long node_child = this.toNode(bIndex, eIndex, childForestNode, this._g.getRootPatternByArity(childForestNode.arity()));
							if(builder.contains(node_child)){
								if(!addedY){
									builder.addNode(node_Y);
									addedY = true;
								}
								builder.addEdge(node_Y, new long[]{node_child});
							}
						}
					}
					
					for(HybridPattern lhs : this.getValidHybridPatterns(forestNode)){
						
						long node = this.toNode(bIndex, eIndex, forestNode, lhs);
						boolean added = false;
						
						ArrayList<HybridPattern[]> RHS = this._g.getRHS(forestNode.arity(), lhs);
						//no edges to add for this pattern.
						if(lhs.isw()){
							continue;
						}
						if(lhs.isX()){
							continue;
						}
						if(lhs.isY()){
							continue;
						}
						for(HybridPattern[] rhs : RHS){
							if(rhs.length == 1){
								long node_c1 = this.toNode(bIndex, eIndex, forestNode, rhs[0]);
								if(builder.contains(node_c1)){
									if(!added){
										builder.addNode(node);
										added = true;
									}
									builder.addEdge(node, new long[]{node_c1});
								}
							} else if(rhs.length == 2){
								for(int cIndex=bIndex+1; cIndex<eIndex; cIndex++){
									long node_c1 = this.toNode(bIndex, cIndex, forestNode, rhs[0]);
									long node_c2 = this.toNode(cIndex, eIndex, forestNode, rhs[1]);
									if(builder.contains(node_c1) && builder.contains(node_c2)){
										if(!added){
											builder.addNode(node);
											added = true;
										}
										builder.addEdge(node, new long[]{node_c1, node_c2});
									}
								}
							} else {
								throw new RuntimeException("# rhs="+Arrays.toString(rhs));
							}
						}
					}
				}
			}
		}
		
		long root = this.toNode_root(sent.length());
		builder.addNode(root);
		
		SemanticForestNode[][] children_of_root = tree.getRoot().getChildren();
		
		if(children_of_root.length!=1)
			throw new RuntimeException("The root should have arity 1...");
		
		SemanticForestNode[] child_of_root = children_of_root[0];
		
		for(int k = 0; k<child_of_root.length; k++){
			long preroot = this.toNode(0, sent.length(), child_of_root[k], this._g.getRootPatternByArity(child_of_root[0].arity()));
			builder.addEdge(root, new long[]{preroot});
		}
		
		BaseNetwork network = builder.build(networkId, inst, param, this);
		
		return network;
	}
	
	
	public BaseNetwork compileUnlabeled(int networkId, Instance ins, LocalNetworkParam param){
		SemTextInstance inst = (SemTextInstance)ins;
//		if(NetworkConfig.TRAIN_MODE_IS_GENERATIVE){
//			throw new RuntimeException("Why do you care about this?");
//		}
		
//		boolean compileSpecificUnlabled = true;
//		
//		if(compileSpecificUnlabled){
//			return this.compile(inst.getInput().length());
//		}
		
		if(this._nodes==null){
			this.compile();
		}
		BaseNetwork network = NetworkBuilder.quickBuild(networkId, inst, this._nodes, this._children, this._numNodesInSubStructure[inst.getInput().length()], param, this);
		return network;
		
	}
	
	private synchronized void compile(){
		
		//done already
		if(this._nodes!=null){
			return;
		}
		
		NetworkBuilder<BaseNetwork> builder = NetworkBuilder.builder();
		
		for(int eIndex=1; eIndex<=this._maxSentLen; eIndex++){
			System.err.println("eIndex="+eIndex);
			
			for(int L = 1; L<=eIndex; L++){
				int bIndex = eIndex - L;
				//[bIndex, eIndex)
				for(SemanticForestNode forestNode : this._forest.getAllNodes()){
					if(forestNode.isRoot())
						continue;
					
					if(eIndex == bIndex+1){
						long node = this.toNode(bIndex, eIndex, forestNode, this._g.getw());
						builder.addNode(node);
					}
					
					if(forestNode.arity()==1){
						SemanticForestNode[] childTreeNodes0 = forestNode.getChildren()[0];
						
						long node_X = this.toNode(bIndex, eIndex, forestNode, this._g.getX());
						boolean added = false;
						for(SemanticForestNode childForestNode : childTreeNodes0){
							long node_child = this.toNode(bIndex, eIndex, childForestNode, this._g.getRootPatternByArity(childForestNode.arity()));
							
							if(builder.contains(node_child)){
								if(!added){
									builder.addNode(node_X);
									added = true;
								}
								builder.addEdge(node_X, new long[]{node_child});
							}
						}
					} else if(forestNode.arity()==2){
						SemanticForestNode[] childTreeNodes0 = forestNode.getChildren()[0];
						SemanticForestNode[] childTreeNodes1 = forestNode.getChildren()[1];
						
						long node_X = this.toNode(bIndex, eIndex, forestNode, this._g.getX());
						boolean addedX = false;
						for(SemanticForestNode childForestNode : childTreeNodes0){
							long node_child = this.toNode(bIndex, eIndex, childForestNode, this._g.getRootPatternByArity(childForestNode.arity()));
							if(builder.contains(node_child)){
								if(!addedX){
									builder.addNode(node_X);
									addedX = true;
								}
								builder.addEdge(node_X, new long[]{node_child});
							}
						}
						
						long node_Y = this.toNode(bIndex, eIndex, forestNode, this._g.getY());
						boolean addedY = false;
						for(SemanticForestNode childForestNode : childTreeNodes1){
							long node_child = this.toNode(bIndex, eIndex, childForestNode, this._g.getRootPatternByArity(childForestNode.arity()));
							if(builder.contains(node_child)){
								if(!addedY){
									builder.addNode(node_Y);
									addedY = true;
								}
								builder.addEdge(node_Y, new long[]{node_child});
							}
						}
					}
					
//					System.err.println(forestNode.getUnit()+"\t"+bIndex+"\t"+eIndex);
					
					for(HybridPattern lhs : this.getValidHybridPatterns(forestNode)){
						
						long node = this.toNode(bIndex, eIndex, forestNode, lhs);
						boolean added = false;
						
						ArrayList<HybridPattern[]> RHS = this._g.getRHS(forestNode.arity(), lhs);
						//no edges to add for this pattern.
						if(lhs.isw()){
							continue;
						}
						if(lhs.isX()){
							continue;
						}
						if(lhs.isY()){
							continue;
						}
						for(HybridPattern[] rhs : RHS){
							
							if(rhs.length == 1){
//								System.err.println(Arrays.toString(rhs));
								long node_c1 = this.toNode(bIndex, eIndex, forestNode, rhs[0]);
								if(builder.contains(node_c1)){
									if(!added){
										builder.addNode(node);
										added = true;
									}
									builder.addEdge(node, new long[]{node_c1});
								}
							} else if(rhs.length == 2){
								for(int cIndex=bIndex+1; cIndex<eIndex; cIndex++){
									long node_c1 = this.toNode(bIndex, cIndex, forestNode, rhs[0]);
									long node_c2 = this.toNode(cIndex, eIndex, forestNode, rhs[1]);
									if(builder.contains(node_c1) && builder.contains(node_c2)){
										if(!added){
											builder.addNode(node);
											added = true;
										}
										builder.addEdge(node, new long[]{node_c1, node_c2});
									}
								}
							} else {
								throw new RuntimeException("# rhs="+Arrays.toString(rhs));
							}
							
						}
					}
				}
			}
			
			long root = this.toNode_root(eIndex);
			builder.addNode(root);
			
			int numNodes = builder.numNodes_tmp();
			this._numNodesInSubStructure[eIndex] = numNodes;
			
			SemanticForestNode[][] children_of_root = this._forest.getRoot().getChildren();
			
			if(children_of_root.length!=1)
				throw new RuntimeException("The root should have arity 1...");
			
			SemanticForestNode[] child_of_root = children_of_root[0];
			
			for(int k = 0; k<child_of_root.length; k++){
				long preroot = this.toNode(0, eIndex, child_of_root[k], this._g.getRootPatternByArity(child_of_root[0].arity()));
//				System.err.println(Arrays.toString(NetworkIDMapper.toHybridNodeArray(root))+"\t"+child_of_root[0].arity()+"\t"+this._g.getRootPatternByArity(child_of_root[0].arity()));
//				System.err.println(Arrays.toString(NetworkIDMapper.toHybridNodeArray(preroot)));
				if(builder.contains(preroot)){
					builder.addEdge(root, new long[]{preroot});
				}
			}
			
		}
		
		BaseNetwork network = builder.buildRudimentaryNetwork();
		this._nodes = network.getAllNodes();
		this._children = network.getAllChildren();
		
		System.err.println(network.countNodes()+" nodes..");
		
	}
	
	private HybridPattern[] getValidHybridPatterns(SemanticForestNode forestNode){
		
		HybridPattern[] ps = this._g.getPatternsByArity(forestNode.arity());
		
//		System.err.println(forestNode.arity());
//		System.err.println(Arrays.toString(ps));
		
		return ps;
	}
	
	@Override
	public SemTextInstance decompile(Network network) {
		
		BaseNetwork stNetwork = (BaseNetwork)network;
		
		SemTextInstance inst = (SemTextInstance)stNetwork.getInstance();
		inst = inst.duplicate();
		
		//if the value is -inf, it means there is no prediction.
		if(stNetwork.getMax()==Double.NEGATIVE_INFINITY){
			return inst;
		}
		
		SemanticForest forest = this.toTree(stNetwork);
		inst.setPrediction(forest);
		
		return inst;
		
	}
	
	private SemanticForest toTree(BaseNetwork network){
		
		SemanticForestNode root = SemanticForestNode.createRootNode(DHTConfig._SEMANTIC_FOREST_MAX_DEPTH);
		this.toTree_helper(network, network.countNodes()-1, root);
		return new SemanticForest(root);
		
	}
	
	private void toTree_helper(BaseNetwork network, int node_k, SemanticForestNode currNode){
		
		long node = network.getNode(node_k);
		int[] ids_node = NetworkIDMapper.toHybridNodeArray(node);
//		System.err.println(">>>"+Arrays.toString(ids_node)+"<<<"+"\t"+node+"\t"+network.getRoot());
//		System.exit(1);
		int[] children_k = network.getMaxPath(node_k);
		double score = network.getMax(node_k);
		if(currNode.getScore()==Double.NEGATIVE_INFINITY){
			currNode.setScore(score);
			currNode.setInfo("info:"+Arrays.toString(NetworkIDMapper.toHybridNodeArray(node)));
		}
		
		for(int child_k : children_k){
			long child = network.getNode(child_k);
			int[] ids_child = NetworkIDMapper.toHybridNodeArray(child);
			
			if(node_k==network.countNodes()-1){
				SemanticUnit unit = this._dm.getAllUnits().get(ids_child[3]);
				SemanticForestNode childNode = new SemanticForestNode(unit, currNode.getHIndex()-1);
				
//				System.err.println("1:"+currNode.getUnit()+"\t"+currNode.getUnit().arity());
				currNode.setChildren(0, new SemanticForestNode[]{childNode});
				
				this.toTree_helper(network, child_k, childNode);
			} else if(this._g.getX().getId()==ids_node[4]){
				
//				if(currNode.arity()==0){
//					throw new RuntimeException("joke?"+"\t"+currNode.getUnit()+"\t"+Arrays.toString(ids_node));
//				}
				
				SemanticUnit unit = this._dm.getAllUnits().get(ids_child[3]);
				SemanticForestNode childNode = new SemanticForestNode(unit, currNode.getHIndex()-1);
				
//				System.err.println("yy:"+currNode.getUnit()+"\t"+childNode.getUnit()+"\t"+currNode.arity());
//				System.err.println("2:"+currNode.getUnit()+"\t"+currNode.getUnit().arity());
				currNode.setChildren(0, new SemanticForestNode[]{childNode});
				
				this.toTree_helper(network, child_k, childNode);
			} else if(this._g.getY().getId()==ids_node[4]){
				SemanticUnit unit = this._dm.getAllUnits().get(ids_child[3]);
				SemanticForestNode childNode = new SemanticForestNode(unit, currNode.getHIndex()-1);
				
//				System.err.println("3:"+currNode.getUnit()+"\t"+currNode.getUnit().arity());
				currNode.setChildren(1, new SemanticForestNode[]{childNode});
				
				this.toTree_helper(network, child_k, childNode);
			} else {
				this.toTree_helper(network, child_k, currNode);
			}
		}
		
	}
	
	private long toNode_root(int sent_len){
		return NetworkIDMapper.toHybridNodeID(new int[]{sent_len+1, 0, 0, 0, 0, Network.NODE_TYPE.max.ordinal()});
	}
	
	private long toNode(int bIndex, int eIndex, SemanticForestNode node, HybridPattern p){
		//okay, the weird problem is now fixed. due to the fact that there are some new units which only appear in the test set.
//		if(12==eIndex && 12==eIndex-bIndex && 16==node.getHIndex() && 0==node.getWIndex() && 2==p.getId()){
//			System.err.println("0-th unit:"+this._dm.getAllUnits().get(0)+"\t"+this._dm.getAllUnits().get(0).getId());
//			System.err.println("1-st unit:"+this._dm.getAllUnits().get(1)+"\t"+this._dm.getAllUnits().get(1).getId());
//			System.err.println("curr unit:"+node.getUnit().getId());
//			throw new RuntimeException("ah??"+"\t"+node.getUnit()+"\t"+node.arity());
//		}
		return NetworkIDMapper.toHybridNodeID(new int[]{eIndex, eIndex-bIndex, node.getHIndex(), node.getWIndex(), 
				p.getId(), Network.NODE_TYPE.max.ordinal()});
	}

}