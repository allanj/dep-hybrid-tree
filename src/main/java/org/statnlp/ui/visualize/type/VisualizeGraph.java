package org.statnlp.ui.visualize.type;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;

import org.statnlp.hypergraph.NetworkIDMapper;

import edu.uci.ics.jung.graph.DirectedGraph;
import edu.uci.ics.jung.graph.DirectedSparseMultigraph;
import edu.uci.ics.jung.graph.util.EdgeType;


public class VisualizeGraph {
	//static int edgeCount = 0;
	
	//static int nodeCount = 0;
	
	//static int hyperEdgeCount = 0;
	
	public DirectedGraph<VNode, VLink> g = null;
	
	//Forest<VNode, VLink> g = null;
	
	ArrayList<VNode> nodes = new ArrayList<VNode>();
	
	ArrayList<HyperLink> hyperlinks = new ArrayList<HyperLink>();
	
	VNode root = null;
	
	VNode sink = null;
	
	public static boolean hideEdgetoSink = true;
	
	long nodes_arr[] = null;
	
	int childrens_arr[][][] = null;
	
	ArrayList<String> words = null;
	
	
	public void setWords(ArrayList<String> words){
		this.words = words;
	}
	
	public VisualizeGraph(){
		clear();
	}
	
	public void setArray(long nodes_arr[], int childrens_arr[][][]){
		this.nodes_arr = nodes_arr;
		this.childrens_arr = childrens_arr;	
	}
	
	public long[] getNodesArray(){
		return nodes_arr;
	}
	
	public int[][][] getChildrensArray(){
		return childrens_arr;
	}
	
	public ArrayList<VNode> getNodes(){
		return nodes;
	}
	
	public void buildArrayToGraph(long[] nodes_arr, int[][][] childrens_arr){
		for (int i = 0; i < nodes_arr.length; i++) {
			VNode node = this.addNode(nodes_arr[i], i);
			int[] ids = NetworkIDMapper.toHybridNodeArray(nodes_arr[i]);
			node.ids = ids;
		}

		for (int k = 0; k < nodes_arr.length; k++) {
			// long parent = nodes[k];
			if (childrens_arr[k] == null){
				continue;
			}
			
			int[][] childrenList = childrens_arr[k];
			
			for (int i = 0; i < childrenList.length; i++) {
				int[] children = childrenList[i];
				this.addEdge(k, children);
			}
		}
	}

	public void buildArrayToGraph(){
		for (int i = 0; i < nodes_arr.length; i++) {
			VNode node = this.addNode(nodes_arr[i], i);

			int[] node_array = NetworkIDMapper.toHybridNodeArray(nodes_arr[node.index]);
			node.label = Arrays.toString(node_array);
			node.type = Itemtype.getType(node_array[4]);
			node.bIndex = node_array[0];
			if (words != null && node.bIndex > 0 && node.type == Itemtype.ROOT){
				node.label = words.get(node.bIndex - 1) + node.label;
			}
		}

		for (int k = 0; k < nodes_arr.length; k++) {
			// long parent = nodes[k];
			int[][] childrenList = childrens_arr[k];
			for (int i = 0; i < childrenList.length; i++) {
				int[] children = childrenList[i];
				this.addEdge(k, children);
			}
		}
	}
	
	
	public void buildGraphToArray(){
		this.nodes_arr = new long[nodes.size()];
		
		for(VNode node : nodes){
			nodes_arr[node.index] = node.id;
		}
		
		this.childrens_arr = new int[nodes_arr.length][][];
		
		for(int k = 0; k < nodes_arr.length; k++){
			VNode node = nodes.get(k);
			
			int num_hyperlinks = node.hyperlinks.size();
			
			childrens_arr[k] = new int[num_hyperlinks][];
			
			for(int i = 0; i < childrens_arr[k].length; i++){
				HyperLink hyperlink = node.hyperlinks.get(i);
				
				int num_links = hyperlink.links.size();
				childrens_arr[k][i] = new int[num_links];
				
				for(int j = 0; j < childrens_arr[k][i].length; j++){
					VNode Dest = hyperlink.links.get(j).Dest;
					childrens_arr[k][i][j] = Dest.index;
				}
			}
		}
	}
	
	public VNode addNode(long id, int index){
		VNode node = new VNode(id, index);
		nodes.add(node);
		g.addVertex(node);
		if (id == 0){
			sink = node;
		}
		return node;
	}
	
	public void addNode(VNode node){
		this.nodes.add(node);
	}
	
	public void addEdge(HyperLink hyperlink){
		this.hyperlinks.add(hyperlink);
	}
	
	public void addEdge(int parent_index, int[] children){	
		VNode parent;
		parent = (parent_index == -1) ? root : nodes.get(parent_index);
		
		ArrayList<VNode> children_nodes = new ArrayList<VNode>();
		for(int i = 0; i < children.length; i++){
			children_nodes.add(nodes.get(children[i]));
		}
		
		HyperLink hyperlink = new HyperLink(parent, children_nodes);
		hyperlinks.add(hyperlink);
		
		for (VLink link : hyperlink.links){
			//if (hideEdgetoSink && link.Dest.type == Itemtype.X) continue;
		
			g.addEdge(link, parent, link.Dest, EdgeType.DIRECTED);
			if (parent.isTypeParentOf(link.Dest)){
				link.Dest.setParent(parent);
			}
		}
	}
	
	public void addRootEdge(int[] children){
		int root_index = nodes.size();
		root = addNode(-1, root_index);
		addEdge(root_index, children);
	}
	
	public void removeEdge(VLink link){
		HyperLink hyperlink = link.hyperlink;
		hyperlink.removeLink(link);
		g.removeEdge(link);
	}
	
	public void removeEdges(Set<VLink> links){
		for(VLink link : links){
			this.removeEdge(link);
		}
	}
	
	public void removeNode(VNode node){
		node.beRemoved();
		g.removeVertex(node);	
	}
	
	public void removeNodes(Set<VNode> nodes){
		for(VNode node : nodes){
			this.removeNode(node);
		}
	}
	
	public void clear(){
		g = new DirectedSparseMultigraph<VNode, VLink>();
		//g = new DelegateForest<VNode, VLink>();
		nodes.clear();
		hyperlinks.clear();
		VLink.edgeCount = 0;
		HyperLink.hyperEdgeCount = 0;
		VNode.nodeCount = 0;
		this.nodes_arr = null;
		this.childrens_arr = null;
		this.words = null;		
	}
	
	public void updateAllNodeID(){
		for(VNode node : this.nodes){
			node.updateNodeIDfromNodeIDArray();
		}
	}
	
	public String[] getWords(){
		ArrayList<VNode> nodes = new ArrayList<VNode>();
		
		for(VNode node : this.nodes){
			if (node.type == Itemtype.A){
				nodes.add(node);
			}
		}
		
		String[] words = new String[nodes.size()];
		for(int i = 0; i < words.length; i++){
			words[i] = nodes.get(i).getContentString();
		}
		
		return words;
	}
	
	public String[] getTags(){
		ArrayList<VNode> nodes = new ArrayList<VNode>();
		
		for(VNode node : this.nodes){
			if (node.type == Itemtype.E){
				nodes.add(node);
			}
		}
		
		String[] tags = new String[nodes.size()];
		for(int i = 0; i < tags.length; i++){
			tags[i] = nodes.get(i).getContentString();
		}
		
		return tags;
	}
	
	public enum Itemtype {
		X, I, T, E, A, ROOT, Edge, UNKNOWN;
		
		@Override
	    public String toString() {
	            return this.name();
	    }
		
		public static Itemtype getType(int typeID){
			switch(typeID){
			case -1:
				return UNKNOWN;
			case 0:
				return X;
			case 1:
				return I;
			case 2:
				return T;
			case 3:
				return E;
			case 4:
				return A;
			case 5:
				return ROOT;
			case 6:
				return Edge;
				
			}
			return UNKNOWN;
		}
		
		public boolean isTypeParentOf(Itemtype type){
			if (this.ordinal() > ROOT.ordinal() || this.ordinal() <= X.ordinal())
				return false;
			
			return (this.ordinal() - type.ordinal() == 1);
		}
	}

}
