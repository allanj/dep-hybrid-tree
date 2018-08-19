package org.statnlp.example.dependency;

import java.util.Arrays;

import org.statnlp.commons.types.Instance;
import org.statnlp.example.base.BaseNetwork;
import org.statnlp.example.base.BaseNetwork.NetworkBuilder;
import org.statnlp.example.dependency.DepConfig.Comp;
import org.statnlp.example.dependency.DepConfig.Dir;
import org.statnlp.hypergraph.LocalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkCompiler;
import org.statnlp.hypergraph.NetworkIDMapper;

public class DepNetworkCompiler extends NetworkCompiler {

	private static final long serialVersionUID = -938028087951999377L;

	private int maxSentLen = 150; //including the root node at 0 idx
	private long[] _nodes;
	private int[][][] _children;
	private boolean DEBUG = false;
	public BaseNetwork genericUnlabeledNetwork;
	
	public enum NodeType {
		dep,
		root,
	}
	
	private int rightDir = Dir.right.ordinal();
	private int leftDir = Dir.left.ordinal();
	
	static {
		/***for dependency parsing: rightIdx, rightIdx-leftIdx, complete, direction, nodeType**/
		NetworkIDMapper.setCapacity(new int[]{200, 200, 5, 5, 10});
	}
	
	public DepNetworkCompiler(int maxSize){
		this.maxSentLen = maxSize;
		this.compileUnlabeledInstancesGeneric();
		
	}

	/**
	 * Obtain a root node, the sentence length should consider the root(pos=0) as well.
	 * @param len: 
	 * @return
	 */
	private long toNode_JointRoot(int len) {
		return NetworkIDMapper.toHybridNodeID(new int[]{len, 0, 0, 0, NodeType.root.ordinal()});
	}
	
	
	private long toNode_DepRoot(int sentLen){
		int endIndex = sentLen - 1;
		return NetworkIDMapper.toHybridNodeID(new int[]{endIndex, endIndex - 0, Comp.comp.ordinal(), Dir.right.ordinal(), NodeType.dep.ordinal()});
	}
	
	private long toNodeIncomp(int leftIndex, int rightIndex, int direction){
		return NetworkIDMapper.toHybridNodeID(new int[]{rightIndex, rightIndex-leftIndex, Comp.incomp.ordinal(), direction, NodeType.dep.ordinal()});
	}
	
	private long toNodeComp(int leftIndex, int rightIndex, int direction){
		return NetworkIDMapper.toHybridNodeID(new int[]{rightIndex, rightIndex-leftIndex, Comp.comp.ordinal(), direction, NodeType.dep.ordinal()});
	}
	
	@Override
	public BaseNetwork compileLabeled(int networkId, Instance inst, LocalNetworkParam param) {
		NetworkBuilder<BaseNetwork> builder = NetworkBuilder.builder();
		DepInstance mfjInst = (DepInstance)inst;
		long jointRoot = this.toNode_JointRoot(inst.size());
		builder.addNode(jointRoot);
		int size = mfjInst.size();
		
		int[] heads = mfjInst.getOutput();
		this.buildDepNetwork(builder, size, heads);
		BaseNetwork network = builder.build(networkId, inst, param, this);
		if (DEBUG) {
			BaseNetwork generic = (BaseNetwork) compileUnlabeled(networkId, inst, param);
			if (!generic.contains(network)) {
				System.err.println("wrong");
			}
		}
		return network;
	}
	
	@Override
	public Network compileUnlabeled(int networkId, Instance inst, LocalNetworkParam param) {
		int size = inst.size();
		long root = toNode_JointRoot(size);
		int root_k = Arrays.binarySearch(this._nodes, root);
		int numNodes = root_k + 1;
		return NetworkBuilder.quickBuild(networkId, inst, this._nodes, this._children, numNodes, param, this);
	}
	
	private void compileUnlabeledInstancesGeneric() {
		if (this._nodes != null) return;
		genericUnlabeledNetwork = this.buildDepNetwork(NetworkBuilder.builder(), maxSentLen, null);
		this._nodes = genericUnlabeledNetwork.getAllNodes();
		this._children = genericUnlabeledNetwork.getAllChildren();
	}
	
	private BaseNetwork buildDepNetwork(NetworkBuilder<BaseNetwork> builder, int maxLength, int[] heads) {
		long rootRight = this.toNodeComp(0, 0, rightDir);
		builder.addNode(rootRight);
		long depRoot = this.toNode_DepRoot(maxLength);
		builder.addNode(depRoot);
		long jointRoot = this.toNode_JointRoot(maxLength);
		if (heads != null) {
			builder.addNode(jointRoot);
			builder.addEdge(jointRoot, new long[]{depRoot});
		}
		for(int rightIndex = 1; rightIndex <= maxLength - 1; rightIndex++){
			//eIndex: 1,2,3,4,5,..n
			long wordLeftNode = this.toNodeComp(rightIndex, rightIndex, leftDir);
			long wordRightNode = this.toNodeComp(rightIndex, rightIndex, rightDir);
			builder.addNode(wordLeftNode);
			builder.addNode(wordRightNode);
			
			for(int L = 1;L <= rightIndex;L++){
				//L:(1),(1,2),(1,2,3),(1,2,3,4),(1,2,3,4,5),...(1..n)
				//bIndex:(0),(1,0),(2,1,0),(3,2,1,0),(4,3,2,1,0),...(n-1..0)
				//span: {(0,1)},{(1,2),(0,2)}
				int leftIndex = rightIndex - L;
				//span:[bIndex, rightIndex]
				for(int complete = 0; complete <= 1; complete++){
					for (int direction = 0; direction <= 1; direction++) {
						if (leftIndex == 0 && direction == 0) continue;
						if (complete == 0) {
							long parent = this.toNodeIncomp(leftIndex, rightIndex, direction);
							if (heads != null) {
								if (direction == rightDir && heads[rightIndex] != leftIndex) continue;
								if (direction == leftDir && heads[leftIndex] != rightIndex) continue;
							}
							for (int m = leftIndex; m < rightIndex; m++) {
								long child_1 = this.toNodeComp(leftIndex, m, rightDir);
								long child_2 = this.toNodeComp(m + 1, rightIndex, leftDir);
								if (builder.contains(child_1) && builder.contains(child_2)) {
									builder.addNode(parent);
									builder.addEdge(parent, new long[]{child_1,child_2});
								}
							}
						}
						
						if (complete == Comp.comp.ordinal() && direction == leftDir) {
							long parent = this.toNodeComp(leftIndex, rightIndex, leftDir);
							for (int m = leftIndex; m < rightIndex; m++) {
								long child_1 = this.toNodeComp(leftIndex, m, leftDir);
								long child_2 = this.toNodeIncomp(m, rightIndex, leftDir);
								if(builder.contains(child_1) && builder.contains(child_2)){
									builder.addNode(parent);
									builder.addEdge(parent, new long[]{child_1,child_2});
								}
							}
						}
						
						if (complete == Comp.comp.ordinal() && direction == rightDir) {
							long parent = this.toNodeComp(leftIndex, rightIndex, rightDir);
							for (int m = leftIndex + 1; m <= rightIndex; m++) {
								long child_1 = this.toNodeIncomp(leftIndex, m, rightDir);
								long child_2 = this.toNodeComp(m, rightIndex, rightDir);
								if (builder.contains(child_1) && builder.contains(child_2)) {
									builder.addNode(parent);
									builder.addEdge(parent, new long[] { child_1, child_2 });
								}
							}
							if (heads == null && leftIndex == 0 && builder.contains(parent)) {
								jointRoot = this.toNode_JointRoot(rightIndex + 1);
								builder.addNode(jointRoot);
								builder.addEdge(jointRoot, new long[]{parent});
							}
						}
					}
				}
			}
		}
		return builder.buildRudimentaryNetwork();
	}
	
	@Override
	public Instance decompile(Network network) {
		BaseNetwork mfjNetwork = (BaseNetwork)network;
		DepInstance mfjInst = (DepInstance)network.getInstance();
		int[] prediction = this.toOutput(mfjNetwork, mfjInst);
		mfjInst.setPrediction(prediction);
		return mfjInst;
	}
	
	
	private int[] toOutput(BaseNetwork network, DepInstance inst) {
		int[] prediction = new int[inst.size()];
		prediction[0] = -1;  //no head for the leftmost root node
		long root = this.toNode_JointRoot(inst.size());
		int rootIdx = Arrays.binarySearch(network.getAllNodes(), root);
		findBest(network, inst, rootIdx, prediction);
		return prediction;
	}
	
	private void findBest(BaseNetwork network, DepInstance inst, int parent_k, int[] prediction) {
		int[] children_k = network.getMaxPath( parent_k);
		for (int child_k: children_k) {
			long node = network.getNode(child_k);
			int[] nodeArr = NetworkIDMapper.toHybridNodeArray(node);
			int rightIndex = nodeArr[0];
			int leftIndex = nodeArr[0] - nodeArr[1];
			int comp = nodeArr[2];
			int direction = nodeArr[3];
			if (comp == Comp.incomp.ordinal()) {
				if (direction == leftDir) {
					prediction[leftIndex] = rightIndex;
				} else {
					prediction[rightIndex] = leftIndex;
				}
			}
			findBest(network, inst, child_k, prediction);
 		}
	}
}
