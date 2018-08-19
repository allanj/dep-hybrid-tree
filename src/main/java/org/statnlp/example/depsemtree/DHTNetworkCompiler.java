package org.statnlp.example.depsemtree;

import java.util.ArrayList;
import java.util.Arrays;

import org.statnlp.commons.types.Dependency;
import org.statnlp.commons.types.Instance;
import org.statnlp.commons.types.Sentence;
import org.statnlp.example.base.BaseNetwork;
import org.statnlp.example.base.BaseNetwork.NetworkBuilder;
import org.statnlp.hypergraph.LocalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkCompiler;
import org.statnlp.hypergraph.NetworkIDMapper;

public class DHTNetworkCompiler extends NetworkCompiler {

	private static final long serialVersionUID = 4835182393678428500L;

	private HybridGrammar _g;
	private SemTextDataManager _dm;
	private SemanticForest _forest;
	
	private int _maxSentLen = 24;
	private long[] _nodes;
	private int[][][] _children;
	private int[] _numNodesInSubStructure = new int[this._maxSentLen + 1];
	
	private boolean DEBUG = false;

	public static enum Dir {
		left,
		right
	}
	
	public static enum nt {
		usual,
		arc
	}
	
	static {
		NetworkIDMapper.setCapacity(new int[]{300, 300, 3, 1000, 1000, 1000, 2, 100});
	}
	
	public DHTNetworkCompiler(HybridGrammar g, SemanticForest forest, SemTextDataManager dm){
		this._g = g;
		this._dm = dm;
		this._forest = forest;
		System.out.println("Compiling generic network");
		this.compileGeneric();
	}
	
	private long toNodeRoot (int sentLen) {
		return toNode(0, sentLen - 1, Dir.right.ordinal(), DHTConfig._SEMANTIC_FOREST_MAX_DEPTH, this._dm.getAllUnits().size(), 0 , nt.usual,0);
	}
	
	private long toNodeUsual(int bIndex, int eIndex, Dir dir, SemanticForestNode node, HybridPattern p) {
		return this.toNode(bIndex, eIndex, dir, node, p, nt.usual, 0); // k is eIndex +1, to be higher than the arc ndoe.
	}
	
	private long toNodeArc (int bIndex, int eIndex, Dir dir, SemanticForestNode node, HybridPattern p, int k) {
		return 
			this.toNode(bIndex, eIndex, 
					dir.ordinal(), node.getHIndex(), node.getWIndex(), p.getId(), nt.arc, k);
	}
	
	private long toNode(int bIndex, int eIndex,Dir dir, 
			SemanticForestNode node, HybridPattern p, nt nodeType, int k) {
		//endIndex, span len, 
		return 
		this.toNode(bIndex, eIndex, dir.ordinal(), node.getHIndex(), node.getWIndex(), p.getId(), nodeType, k);
	}
	
	private long toNode(int bIndex, int eIndex, int direction, int hIndex, int wIndex, int pId, nt nodeType, int k) {
		return NetworkIDMapper.toHybridNodeID(new int[] {
				//the direction becomes the mid index for the full span
				eIndex, eIndex - bIndex, direction, hIndex, wIndex, pId, nodeType.ordinal(), k
		});
	}
	
	/**
	 * For debug purpose
	 * @param node
	 * @return
	 */
	@SuppressWarnings("unused")
	private String node2Str(long node) {
		StringBuilder sb = new StringBuilder();
		int[] arr = NetworkIDMapper.toHybridNodeArray(node);
		if (arr[4] == this._dm.getAllUnits().size()) return "ROOT";
		sb.append("[");
		sb.append((arr[0] - arr[1]) + "," + (arr[0]) + 
				" mIdx:" + arr[7] + "  " +nt.values()[arr[6]] + " " + 
				Dir.values()[arr[2]].toString() + 
				" h:" + arr[3] + 
				" unit:" + this._dm.getAllUnits().get(arr[4]) + " " + 
				this._g.getPatternById(arr[5]));
		return sb.toString();
	}
	
	@Override
	public BaseNetwork compileLabeled(int networkId, Instance ins, LocalNetworkParam param) {
		SemTextInstance inst = (SemTextInstance)ins;
		NetworkBuilder<BaseNetwork> builder = NetworkBuilder.builder();
		Sentence sent = inst.getInput();
		SemanticForest tree = inst.getOutput();
		
		for(int eIndex = 1; eIndex < sent.length(); eIndex++){
			for(int L = 0; L<=eIndex; L++){ //The length can be 0.
				int bIndex = eIndex - L;
				//[bIndex, eIndex]
				for (Dir direction : Dir.values()) {
					if (bIndex == 0 && direction == Dir.left) continue;
					if (bIndex == eIndex && bIndex == 0) continue; //there is no 0,0 span for both left or right dierection.
					for (SemanticForestNode forestNode : tree.getAllNodes()) {
						if (forestNode.isRoot()) continue; //dummy node.
						
						if (bIndex == eIndex) {
							long w = this.toNodeUsual(eIndex, eIndex, direction, forestNode,_g.getw());
							builder.addNode(w);
						}
						if (forestNode.arity() >= 1) {
							SemanticForestNode[] childTreeNodes0 = forestNode.getChildren()[0];
							long node_X = this.toNodeUsual(bIndex, eIndex, direction, forestNode, this._g.getX());
							for(SemanticForestNode childForestNode : childTreeNodes0){
								long node_child = this.toNodeUsual(bIndex, eIndex, direction, childForestNode, this._g.getRootPatternByArity(childForestNode.arity()));
								if (builder.contains(node_child)) {
									builder.addNode(node_X);
									builder.addEdge(node_X, new long[] {node_child});
								}
							}
						}
						if (forestNode.arity() == 2) {
							SemanticForestNode[] childTreeNodes1 = forestNode.getChildren()[1];
							long node_Y = this.toNodeUsual(bIndex, eIndex, direction, forestNode, this._g.getY());
							for(SemanticForestNode childForestNode : childTreeNodes1){
								long node_child = this.toNodeUsual(bIndex, eIndex, direction, childForestNode, this._g.getRootPatternByArity(childForestNode.arity()));
								if (builder.contains(node_child)) {
									 builder.addNode(node_Y);
									 builder.addEdge(node_Y, new long[] {node_child});
								}
							}
						}
						
						for (HybridPattern lhs : this.getValidHybridPatterns(forestNode)) {
							if(lhs.isw() || lhs.isY()){
								continue;
							}
							long node = this.toNodeUsual(bIndex, eIndex, direction, forestNode, lhs);
							ArrayList<HybridPattern[]> RHS = this._g.getRHS(forestNode.arity(), lhs);
							if (lhs.isA() || lhs.isB() || lhs.isC() ) {
								for(HybridPattern[] rhs : RHS){
									if (rhs.length != 1) throw new RuntimeException("right handside is not len of 1");
									int beginIndex = direction == Dir.right ? bIndex + 1 : bIndex;
									int endIndex = direction == Dir.right ? eIndex : eIndex - 1;
									for (int k = beginIndex; k <= endIndex; k++) {
										long arcNode = this.toNodeArc(bIndex, eIndex, direction, forestNode, rhs[0], k);
										if (builder.contains(arcNode)) {
											builder.addNode(node);
											builder.addEdge(node, new long[] {arcNode});
										}
									}
									//constraint on only right direction can point to the left most index.
									//and it must be X/W., can relax the constraint to left direction as well.
									if ((rhs[0].isX() || rhs[0].isW()) && direction == Dir.right && bIndex != 0) { 
										long arcNode = this.toNodeArc(bIndex, eIndex, direction, forestNode, rhs[0], bIndex);
										if (builder.contains(arcNode)) {
											builder.addNode(node);
											builder.addEdge(node, new long[] {arcNode});
										}
									}
								}
							} else {
								if (lhs.isW()) {
									//build the usual W node.
									long usualWNode = this.toNodeUsual(bIndex, eIndex, direction, forestNode, lhs);
									for(HybridPattern[] rhs : RHS){
										if (rhs.length == 1 && rhs[0].isw() && bIndex == eIndex) { //should be W.
											long child = this.toNodeUsual(bIndex, eIndex, direction, forestNode, rhs[0]);
											if (builder.contains(child)) {
												builder.addNode(usualWNode);
												builder.addEdge(usualWNode, new long[] {child});
											}
										}
										if (rhs.length == 3) { //should only be W - > w w W
											int c1Idx = direction == Dir.right ? bIndex : eIndex;
											int c2Idx = direction == Dir.right ? bIndex + 1 : eIndex - 1;
											int c3Start = direction == Dir.right ? bIndex + 1 : bIndex;
											int c3End = direction == Dir.right ? eIndex : eIndex - 1;
											long node_c1 = this.toNodeUsual(c1Idx, c1Idx, direction, forestNode, rhs[0]); //make the last one the W.
											long node_c2 = this.toNodeUsual(c2Idx, c2Idx, Dir.values()[Dir.values().length - direction.ordinal() - 1], forestNode, rhs[1]);
											long node_c3 = this.toNodeUsual(c3Start, c3End, direction, forestNode, rhs[2]);
											if (builder.contains(node_c1) && builder.contains(node_c2) && builder.contains(node_c3)) {
												builder.addNode(usualWNode);
												builder.addEdge(usualWNode, new long[] {node_c1, node_c2, node_c3});
											}
										}
									}
								}
								//build arc node first.  (W, WW, WX, XW, XY, YX)
								//first W. //constraint on W and X. right direction also
								if ((lhs.isW() || lhs.isX()) && direction == Dir.right) {
									long wArcNode = this.toNodeArc(bIndex, eIndex, direction, forestNode, lhs, bIndex);
									long child = this.toNodeUsual(bIndex, eIndex, direction, forestNode, lhs);
									if (builder.contains(child)) {
										builder.addNode(wArcNode);
										builder.addEdge(wArcNode, new long[] {child});
									}
								}
								if (!lhs.isW() && !lhs.isX()) {
									int beginIndex = direction == Dir.right ? bIndex + 1 : bIndex;
									int endIndex = direction == Dir.right ? eIndex : eIndex - 1;
									for (int k = beginIndex; k <= endIndex; k++) {
										long arcNode = this.toNodeArc(bIndex, eIndex, direction, forestNode, lhs, k);
										for(HybridPattern[] rhs : RHS){
											if (rhs.length == 2) { //should be WW, XW, WX, XY, YX
												long node_c1 = this.toNodeUsual(beginIndex, k, Dir.left, forestNode, rhs[0]);
												long node_c2 = this.toNodeUsual(k, endIndex, Dir.right, forestNode, rhs[1]);
												if (builder.contains(node_c1) && builder.contains(node_c2)) {
													builder.addNode(arcNode);
													builder.addEdge(arcNode, new long[] {node_c1, node_c2});
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		
//		for (long node : builder.getNodes_tmp()) {
//			System.out.println(node2Str(node));
//		}
		
		long root = this.toNodeRoot(sent.length());
		builder.addNode(root);
		SemanticForestNode[][] children_of_root = tree.getRoot().getChildren();
		if(children_of_root.length!=1)
			throw new RuntimeException("The root should have arity 1...");
		SemanticForestNode[] child_of_root = children_of_root[0];
		for(int k = 0; k<child_of_root.length; k++){
			long preroot = this.toNodeUsual(0, sent.length() - 1, Dir.right, child_of_root[k], this._g.getRootPatternByArity(child_of_root[k].arity()));
//			System.out.println(inst.isLabeled());
//			System.out.println(sent.toString());
//			System.out.println(node2Str(preroot));
			builder.addEdge(root, new long[]{preroot});
		}
		BaseNetwork network = builder.build(networkId, inst, param, this);
		if (DEBUG) {
			BaseNetwork unlabeled = this.compileUnlabeled(networkId, inst, param);
			if (!unlabeled.contains(network)) {
				System.out.println(inst.size());
				throw new RuntimeException("not contained");
			}
			else System.out.println("contain");
		}
		return network;
	}

	@Override
	public BaseNetwork compileUnlabeled(int networkId, Instance ins, LocalNetworkParam param) {
		SemTextInstance inst = (SemTextInstance)ins;
//		return this.compileLabeled(networkId, inst, param);
		BaseNetwork network = NetworkBuilder.quickBuild(networkId, inst, this._nodes, this._children, this._numNodesInSubStructure[inst.getInput().length()-1], param, this);
		return network;
	}
	
	/**
	 * Build the generic network, only run once.
	 */
	private void compileGeneric(){
//		System.out.println(this._forest);
		NetworkBuilder<BaseNetwork> builder = NetworkBuilder.builder();
		for(int eIndex = 1; eIndex <= this._maxSentLen; eIndex++){
			System.err.println("eIndex: " + eIndex);
			for(int L = 0; L<=eIndex; L++){ //The length can be 0.
				int bIndex = eIndex - L;
				//[bIndex, eIndex]
				for (Dir direction : Dir.values()) {
					if (bIndex == 0 && direction == Dir.left) continue;
					if (bIndex == eIndex && bIndex == 0) continue;
					for (SemanticForestNode forestNode : this._forest.getAllNodes()) {
						if (forestNode.isRoot()) continue; //dummy node.
						
						if (bIndex == eIndex) {
							long w = this.toNodeUsual(eIndex, eIndex, direction, forestNode,_g.getw());
							builder.addNode(w);
						}
						if (forestNode.arity() >= 1) {
							SemanticForestNode[] childTreeNodes0 = forestNode.getChildren()[0];
							long node_X = this.toNodeUsual(bIndex, eIndex, direction, forestNode, this._g.getX());
							for(SemanticForestNode childForestNode : childTreeNodes0){
								long node_child = this.toNodeUsual(bIndex, eIndex, direction, childForestNode, this._g.getRootPatternByArity(childForestNode.arity()));
								if (builder.contains(node_child)) {
									builder.addNode(node_X);
									builder.addEdge(node_X, new long[] {node_child});
								}
							}
						}
						if (forestNode.arity() == 2) {
							SemanticForestNode[] childTreeNodes1 = forestNode.getChildren()[1];
							long node_Y = this.toNodeUsual(bIndex, eIndex, direction, forestNode, this._g.getY());
							for(SemanticForestNode childForestNode : childTreeNodes1){
								long node_child = this.toNodeUsual(bIndex, eIndex, direction, childForestNode, this._g.getRootPatternByArity(childForestNode.arity()));
								if (builder.contains(node_child)) {
									 builder.addNode(node_Y);
									 builder.addEdge(node_Y, new long[] {node_child});
								}
							}
						}
						
						for (HybridPattern lhs : this.getValidHybridPatterns(forestNode)) {
							if(lhs.isw() || lhs.isY()){
								continue;
							}
							long node = this.toNodeUsual(bIndex, eIndex, direction, forestNode, lhs);
							ArrayList<HybridPattern[]> RHS = this._g.getRHS(forestNode.arity(), lhs);
							if (lhs.isA() || lhs.isB() || lhs.isC() ) {
								for(HybridPattern[] rhs : RHS){
									if (rhs.length != 1) throw new RuntimeException("right handside is not len of 1");
									int beginIndex = direction == Dir.right ? bIndex + 1 : bIndex;
									int endIndex = direction == Dir.right ? eIndex : eIndex - 1;
									for (int k = beginIndex; k <= endIndex; k++) {
										long arcNode = this.toNodeArc(bIndex, eIndex, direction, forestNode, rhs[0], k);
										if (builder.contains(arcNode)) {
											builder.addNode(node);
											builder.addEdge(node, new long[] {arcNode});
										}
									}
									//constraint on only right direction can point to the left most index.
									//and it must be X/W., can relax the constraint to left direction as well.
									if ((rhs[0].isX() || rhs[0].isW()) && direction == Dir.right && bIndex != 0) { 
										//so from A -> W, only allows when the middle index is same as the left index. otherwise should be A -> WW.
										//from B->X, also only allows when the middle index is same as the left index, otherwise should be either WX/XW
										long arcNode = this.toNodeArc(bIndex, eIndex, direction, forestNode, rhs[0], bIndex);
										if (builder.contains(arcNode)) {
											builder.addNode(node);
											builder.addEdge(node, new long[] {arcNode});
										}
									}
								}
							} else {
								if (lhs.isW()) {
									//build the usual W node.
									long usualWNode = this.toNodeUsual(bIndex, eIndex, direction, forestNode, lhs);
									for(HybridPattern[] rhs : RHS){
										if (rhs.length == 1 && rhs[0].isw() && bIndex == eIndex) { //should be W.
//											if (bIndex != eIndex) throw new RuntimeException("left right not equal?");
											long child = this.toNodeUsual(bIndex, eIndex, direction, forestNode, rhs[0]);
											if (builder.contains(child)) {
												builder.addNode(usualWNode);
												builder.addEdge(usualWNode, new long[] {child});
											}
										}
										if (rhs.length == 3) { //should only be W - > w w W
											int c1Idx = direction == Dir.right ? bIndex : eIndex;
											int c2Idx = direction == Dir.right ? bIndex + 1 : eIndex - 1;
											int c3Start = direction == Dir.right ? bIndex + 1 : bIndex;
											int c3End = direction == Dir.right ? eIndex : eIndex - 1;
											long node_c1 = this.toNodeUsual(c1Idx, c1Idx, direction, forestNode, rhs[0]); //make the last one the W.
											long node_c2 = this.toNodeUsual(c2Idx, c2Idx, Dir.values()[Dir.values().length - direction.ordinal() - 1], forestNode, rhs[1]);
											long node_c3 = this.toNodeUsual(c3Start, c3End, direction, forestNode, rhs[2]);
											if (builder.contains(node_c1) && builder.contains(node_c2) && builder.contains(node_c3)) {
												builder.addNode(usualWNode);
												builder.addEdge(usualWNode, new long[] {node_c1, node_c2, node_c3});
											}
										}
									}
								}
								//build arc node first.  (W, WW, WX, XW, XY, YX)
								//first W. //constraint on W and X. right direction also
								if ((lhs.isW() || lhs.isX()) && direction == Dir.right) {
									long wArcNode = this.toNodeArc(bIndex, eIndex, direction, forestNode, lhs, bIndex);
									long child = this.toNodeUsual(bIndex, eIndex, direction, forestNode, lhs);
									if (builder.contains(child)) {
										builder.addNode(wArcNode);
										builder.addEdge(wArcNode, new long[] {child});
									}
								}
								if (!lhs.isW() && !lhs.isX()) {
									int beginIndex = direction == Dir.right ? bIndex + 1 : bIndex;
									int endIndex = direction == Dir.right ? eIndex : eIndex - 1;
									for (int k = beginIndex; k <= endIndex; k++) {
										long arcNode = this.toNodeArc(bIndex, eIndex, direction, forestNode, lhs, k);
										for(HybridPattern[] rhs : RHS){
											if (rhs.length == 2) { //should be WW, XW, WX, XY, YX
												long node_c1 = this.toNodeUsual(beginIndex, k, Dir.left, forestNode, rhs[0]);
												long node_c2 = this.toNodeUsual(k, endIndex, Dir.right, forestNode, rhs[1]);
												if (builder.contains(node_c1) && builder.contains(node_c2)) {
													builder.addNode(arcNode);
													builder.addEdge(arcNode, new long[] {node_c1, node_c2});
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
			long root = this.toNodeRoot(eIndex + 1);
			builder.addNode(root);
			int numNodes = builder.numNodes_tmp();
			this._numNodesInSubStructure[eIndex] = numNodes;
			SemanticForestNode[][] children_of_root = this._forest.getRoot().getChildren();
			if(children_of_root.length!=1)
				throw new RuntimeException("The root should have arity 1...");
			SemanticForestNode[] child_of_root = children_of_root[0];
			boolean rootAdded = false;
			for(int k = 0; k<child_of_root.length; k++){
//				System.out.println(child_of_root[k]);
				long preroot = this.toNodeUsual(0, eIndex, Dir.right, child_of_root[k], this._g.getRootPatternByArity(child_of_root[k].arity()));
				if(builder.contains(preroot)){
					rootAdded = true;
					builder.addEdge(root, new long[]{preroot});
				}
			}
			if (!rootAdded) {
				System.out.println("root not added for index: " + eIndex);
			}
		}
		
		BaseNetwork network = builder.buildRudimentaryNetwork();
		
		this._nodes = network.getAllNodes();
//		for (long node : _nodes) {
//			System.out.println(this.node2Str(node));
//		}
		this._children = network.getAllChildren();
		
		System.err.println(network.countNodes()+" nodes..");
	}
	
	private HybridPattern[] getValidHybridPatterns(SemanticForestNode forestNode){
		HybridPattern[] ps = this._g.getPatternsByArity(forestNode.arity());
		return ps;
	}

	@Override
	public Instance decompile(Network network) {
		BaseNetwork stNetwork = (BaseNetwork)network;
		SemTextInstance inst = (SemTextInstance)stNetwork.getInstance();
		System.err.println(stNetwork.getMax() + " id: " + inst.getInstanceId() + " " + inst.getInput().toString());
		//if the value is -inf, it means there is no prediction.
		if(stNetwork.getMax()==Double.NEGATIVE_INFINITY){
			return inst;
		}
		SemanticForest forest = this.toTree(stNetwork);
//		System.out.println(forest);
		inst.setPrediction(forest);
		return inst;
	}
	
	private SemanticForest toTree(BaseNetwork network){
		SemanticForestNode root = SemanticForestNode.createRootNode(DHTConfig._SEMANTIC_FOREST_MAX_DEPTH);
		SemTextInstance inst  = (SemTextInstance)network.getInstance();
		inst.input.deps = null; //clear the dependency.
		this.toTree_helper(network, network.countNodes()-1, root);
		return new SemanticForest(root);
	}
	
	//eIndex, eIndex - bIndex, direction, hIndex, wIndex, pId, nodeType.ordinal(), k
	private void toTree_helper(BaseNetwork network, int node_k, SemanticForestNode currNode){
		int[] ids_node = network.getNodeArray(node_k);
		int[] children_k = network.getMaxPath(node_k);
		double score = network.getMax(node_k);
		SemTextInstance inst  = (SemTextInstance)network.getInstance();
		if(currNode.getScore()==Double.NEGATIVE_INFINITY){
			currNode.setScore(score);
//			System.out.println("neg");
			currNode.setInfo("info:"+Arrays.toString(ids_node));
		}
		for (int child_k : children_k) {
			int[] ids_child = network.getNodeArray(child_k);
//			System.out.println("unit: " + node2Str(network.getNode(child_k)));
			if (node_k == network.countNodes() - 1) {
				SemanticUnit unit = this._dm.getAllUnits().get(ids_child[4]);
//				System.out.println("root unit: " + unit);
				SemanticForestNode childNode = new SemanticForestNode(unit, currNode.getHIndex() -1);
				currNode.setChildren(0, new SemanticForestNode[] {childNode});
				this.toTree_helper(network, child_k, childNode);
			} else if (this._g.getX().getId() == ids_node[5] && ids_node[6] == nt.usual.ordinal() ) {
				SemanticUnit unit = this._dm.getAllUnits().get(ids_child[4]);
//				System.out.println("c1: " + unit);
				SemanticForestNode childNode = new SemanticForestNode(unit, currNode.getHIndex() - 1);
				currNode.setChildren(0, new SemanticForestNode[] {childNode});
				this.toTree_helper(network, child_k, childNode);
			} else if (this._g.getY().getId() == ids_node[5] && ids_node[6] == nt.usual.ordinal()) {
				SemanticUnit unit = this._dm.getAllUnits().get(ids_child[4]);
//				System.out.println("c2: " + unit);
				SemanticForestNode childNode = new SemanticForestNode(unit, currNode.getHIndex() - 1);
				currNode.setChildren(1, new SemanticForestNode[] {childNode});
				this.toTree_helper(network, child_k, childNode);
			} else if (this._g.getPatternById(ids_node[5]).isA() ||
					this._g.getPatternById(ids_node[5]).isB() ||
					this._g.getPatternById(ids_node[5]).isC()){
				Dir direction = Dir.values()[ids_child[2]];
				int eIdx = ids_child[0];
				int bIdx = ids_child[0] - ids_child[1];
				int k = ids_child[7];
				SemanticUnit unit = this._dm.getAllUnits().get(ids_child[4]);
				if (direction == Dir.right) {
					inst.getInput().addDep(new Dependency(bIdx, k, unit));
				} else {
					inst.getInput().addDep(new Dependency(eIdx, k, unit));
				}
				this.toTree_helper(network, child_k, currNode);
			} else {
				this.toTree_helper(network, child_k, currNode);
			}
		}
	}
	
}
