package org.statnlp.example.sempar_bak;

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
	private int[] _numNodesInSubStructure = new int[this._maxSentLen+1];

	public static enum Dir {
		left,
		right
	}
	
	public static enum Comp {
		incomplete,
		complete
	}
	
	static {
		NetworkIDMapper.setCapacity(new int[]{300, 300, 1000, 1000, 1000, 2});
	}
	
	public DHTNetworkCompiler(HybridGrammar g, SemanticForest forest, SemTextDataManager dm){
		this._g = g;
		this._dm = dm;
		this._forest = forest;
	}
	
	private long toNodeRoot (int sentLen) {
		return toNode(sentLen+1, 0, 0, 0, 0, 0 ,0);
	}
	
	private long toNodeIncomplete (int bIndex, int eIndex, Dir dir, SemanticForestNode node) {
		return 
			this.toNode(bIndex, eIndex, Comp.incomplete.ordinal(), 
					dir.ordinal(), node.getHIndex(), node.getWIndex(), 0);
	}
	
	private long toNode(int bIndex, int eIndex, Comp comp ,Dir dir, 
			SemanticForestNode node, HybridPattern p) {
		//endIndex, span len, 
		return 
		this.toNode(bIndex, eIndex, comp.ordinal(), dir.ordinal(), node.getHIndex(), node.getWIndex(), p.getId());
	}
	
	private long toNode(int bIndex, int eIndex, int comp, int direction, int hIndex, int wIndex, int pId) {
		return NetworkIDMapper.toHybridNodeID(new int[] {
				//the direction becomes the mid index for the full span
				eIndex, eIndex - bIndex, comp, direction, hIndex, wIndex, pId
		});
	}
	
	@Override
	public Network compileLabeled(int networkId, Instance ins, LocalNetworkParam param) {
		SemTextInstance inst = (SemTextInstance)ins;
		NetworkBuilder<BaseNetwork> builder = NetworkBuilder.builder();
		Sentence sent = inst.getInput();
		SemanticForest tree = inst.getOutput();
		
		for(int eIndex = 1; eIndex <= sent.length(); eIndex++){
			for (SemanticForestNode forestNode : tree.getAllNodes()) {
				long wl = this.toNode(eIndex, eIndex, Comp.complete, Dir.left, forestNode,_g.getw());
				long wr = this.toNode(eIndex, eIndex, Comp.complete, Dir.right, forestNode, _g.getw());
				builder.addNode(wl);
				builder.addNode(wr);
			}
			for(int L = 1; L<=eIndex; L++){
				int bIndex = eIndex - L;
				//[bIndex, eIndex]
				for (Comp comp : Comp.values()) {
					for (Dir direction : Dir.values()) {
						if (bIndex == 0 && direction == Dir.left) continue;
						for (SemanticForestNode forestNode : tree.getAllNodes()) {
							if (forestNode.isRoot()) continue; //dummy node.
							if (comp == Comp.incomplete) {
								long uniNode = this.toNodeIncomplete(bIndex, eIndex, direction, forestNode);
								builder.addNode(uniNode);
							} else {
								//complete node.
								if (direction == Dir.right) {
									if (forestNode.arity() >= 1) {
										SemanticForestNode[] childTreeNodes0 = forestNode.getChildren()[0];
										long node_X = this.toNode(bIndex, eIndex, comp, direction, forestNode, this._g.getX());
										for (int k = bIndex + 1; k <= eIndex; k++) {
											for(SemanticForestNode childForestNode : childTreeNodes0){
												 long uniNode = this.toNodeIncomplete(bIndex, k, direction, childForestNode);
												 long fullNode = this.toNode(bIndex + 1, eIndex, Comp.complete.ordinal(), k, 
														 childForestNode.getHIndex(), childForestNode.getWIndex(), 
														 this._g.getRootPatternByArity(childForestNode.arity()).getId());
												 if (builder.contains(uniNode) && builder.contains(fullNode)) {
													 builder.addNode(node_X);
													 builder.addEdge(node_X, new long[] {uniNode, fullNode});
												 }
											}
										}
									}
									if (forestNode.arity() == 2) {
										SemanticForestNode[] childTreeNodes1 = forestNode.getChildren()[1];
										long node_Y = this.toNode(bIndex, eIndex, comp, direction, forestNode, this._g.getY());
										for (int k = bIndex + 1; k <= eIndex; k++) {
											for(SemanticForestNode childForestNode : childTreeNodes1){
												long uniNode = this.toNodeIncomplete(bIndex, k, direction, childForestNode);
												long fullNode = this.toNode(bIndex + 1, eIndex, Comp.complete.ordinal(), k, 
														 childForestNode.getHIndex(), childForestNode.getWIndex(), 
														 this._g.getRootPatternByArity(childForestNode.arity()).getId());
												if (builder.contains(uniNode) && builder.contains(fullNode)) {
													 builder.addNode(node_Y);
													 builder.addEdge(node_Y, new long[] {uniNode, fullNode});
												}
											}
										}
									}
								} else {
									//left Direction.
									if (forestNode.arity() >= 1) {
										SemanticForestNode[] childTreeNodes0 = forestNode.getChildren()[0];
										long node_X = this.toNode(bIndex, eIndex, comp, direction, forestNode, this._g.getX());
										for (int k = bIndex ; k <= eIndex - 1; k++) {
											for(SemanticForestNode childForestNode : childTreeNodes0){
												 long uniNode = this.toNodeIncomplete(k, eIndex, direction, childForestNode);
												 long fullNode = this.toNode(bIndex, eIndex - 1, Comp.complete.ordinal(), k, 
														 childForestNode.getHIndex(), childForestNode.getWIndex(), 
														 this._g.getRootPatternByArity(childForestNode.arity()).getId());
												 if (builder.contains(uniNode) && builder.contains(fullNode)) {
													 builder.addNode(node_X);
													 builder.addEdge(node_X, new long[] {uniNode, fullNode});
												 }
											}
										}
									}
									if (forestNode.arity() == 2) {
										SemanticForestNode[] childTreeNodes1 = forestNode.getChildren()[1];
										long node_Y = this.toNode(bIndex, eIndex, comp, direction, forestNode, this._g.getY());
										for (int k = bIndex; k <= eIndex - 1; k++) {
											for(SemanticForestNode childForestNode : childTreeNodes1){
												long uniNode = this.toNodeIncomplete(k, eIndex, direction, childForestNode);
												long fullNode = this.toNode(bIndex, eIndex - 1, Comp.complete.ordinal(), k, 
														 childForestNode.getHIndex(), childForestNode.getWIndex(), 
														 this._g.getRootPatternByArity(childForestNode.arity()).getId());
												if (builder.contains(uniNode) && builder.contains(fullNode)) {
													 builder.addNode(node_Y);
													 builder.addEdge(node_Y, new long[] {uniNode, fullNode});
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
		return null;
	}

	@Override
	public Network compileUnlabeled(int networkId, Instance inst, LocalNetworkParam param) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Instance decompile(Network network) {
		// TODO Auto-generated method stub
		return null;
	}
	
}
