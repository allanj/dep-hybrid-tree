package org.statnlp.ui.visualize.type;

import java.util.ArrayList;

/**
 * Represents a hyperedge in the visualization graph
 */
public class HyperLink {
	public static int hyperEdgeCount = 0;
	
	public ArrayList<VLink> links = new ArrayList<VLink>();
	
	public VNode parent;
	
	public int id;
	
	public int hyperid;
	
	public HyperLink(VNode parent, ArrayList<VNode> children){
		this.id = hyperEdgeCount++;
		this.parent = parent;
		this.hyperid = parent.hyperlinks.size();
		parent.hyperlinks.add(this);
		
		for(int i = 0; i < children.size(); i++){
			links.add(new VLink(this.hyperid, children.get(i), this));
		}
		
	}
	
	public HyperLink(VNode parent){
		this.id = hyperEdgeCount++;
		this.parent = parent;
		this.hyperid = parent.hyperlinks.size();
		parent.hyperlinks.add(this);
		
	}
	
	public void addLink(VLink link){
		links.add(link);
	}
	
	public String toString(){
		StringBuffer sb = new StringBuffer();
		sb.append("[" + hyperid + "|" + parent.index + "]\n");
		for(VLink link : links){
			sb.append("\t" + link);
		}
		
		return sb.toString();
	}
	
	public int getLinkCount(){
		return this.links.size();
	}
	
	public boolean removeLink(VLink link){
		if (links.contains(link)){
			links.remove(link);
			return true;
		}
		return false;
	}
	
	public ArrayList<VLink> getLinks(){
		return this.links;
	}
	
}