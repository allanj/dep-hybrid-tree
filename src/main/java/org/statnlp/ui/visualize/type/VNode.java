package org.statnlp.ui.visualize.type;

import java.awt.Color;
import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.Arrays;

import org.statnlp.hypergraph.NetworkIDMapper;
import org.statnlp.ui.visualize.type.VisualizeGraph.Itemtype;

public class VNode {
	public static int nodeCount = 0;
	
	public static int UNDEFINED = -1;

	public ArrayList<HyperLink> hyperlinks = new ArrayList<HyperLink>();

	public Itemtype type = Itemtype.UNKNOWN;

	public long id = UNDEFINED;
	
	public int[] ids = null;
	
	public int index;
	
	public int bIndex = UNDEFINED;

	public Point2D point = null;
	
	public String label = "";
	
	public Color color = Color.WHITE;
	
	public int tag_length = 0;
	
	public int tag_ID = 0;
	
	VNode parent = null;
	
	Object content = null;
	
	boolean picked = false;
	
	public int typeID = 0;

	public VNode(long id, int index) {
		this.id = id;
		this.index = index;
	}

	VNode() {
		this.index = nodeCount++;
	}

	public static VNode createNode() {
		return new VNode();
	}
	
	public void setParent(VNode node){
		this.parent = node;
		if (this.type == Itemtype.T){
			this.parent.tag_length++;
			this.tag_ID = this.parent.tag_length;
		}
		
		if (this.type == Itemtype.I){
			this.tag_ID = this.parent.tag_ID;
		}
	}
	
	
	public void update_bIndex_downwards(){
		for(HyperLink hyperlink : this.hyperlinks){
			for(VLink link : hyperlink.links){
				VNode child = link.Dest;
				if (child.type != Itemtype.X && this.isTypeParentOf(child)){
					child.bIndex = this.bIndex;
					child.update_bIndex_downwards();
				}
			}
		}
	}
	
	public void update_tag_length(){
		if (this.type == Itemtype.E){
			update_tag_length_upwards();
			update_tag_length_downwards();
		}
	}
	
	public void update_tag_length_upwards(){
		if (this.parent != null){
			this.parent.tag_length = this.tag_length;
			this.parent.update_tag_length_upwards();
		}
	}
	
	public void update_tag_length_downwards(){
		for(HyperLink hyperlink : this.hyperlinks){
			for(VLink link : hyperlink.links){
				VNode child = link.Dest;
				if (child.type != Itemtype.X && this.isTypeParentOf(child)){
					child.tag_length = this.tag_length;
					child.update_tag_length_downwards();
				}
			}
		}
	}
	
	public VNode getParent(){
		return this.parent;
	}

	public String toString() {
		return "V: " + Arrays.toString(this.getNodeIDArray());
	}

	public void setType(int typeID) {
		type = Itemtype.getType(typeID);
		if (type == Itemtype.X){
			this.id = 0;
			this.bIndex = 0;
		}
	}
	
	public Itemtype getType() {
		return type;
	}
	
	public void setbIndex(int bIndex){
		this.bIndex = bIndex;
	}
	
	public HyperLink getHyperLinkInLinearMode(){
		if (hyperlinks.isEmpty()){
			HyperLink hyperlink = new HyperLink(this);
			hyperlinks.add(hyperlink);
		}
		
		return hyperlinks.get(0);
	}
	
	public int getChildrenCount(){
		int count = 0;
		for(HyperLink hyperlink : this.hyperlinks){
			count += hyperlink.getLinkCount();
		}
		
		return count;
	}
	
	public void setNodeID(long id){
		this.id = id;
	}
	
	public void setNodeID(int[] nodeID_arr){
		if (nodeID_arr == null){
			this.id = UNDEFINED;
		} else {
			this.id = NetworkIDMapper.toHybridNodeID(nodeID_arr);
		}
	}
	

	
	public boolean isTypeParentOf(VNode node){
		return this.type.isTypeParentOf(node.type);
	}
	
	
	public int[] getNodeIDArray(){
		int srcHeight = this.bIndex;
		int srcWidth = 255;
		int tgtHeight = srcHeight;
		int tgtWidth = UNDEFINED;
		if (this.tag_length != UNDEFINED){
			switch (type){
			case ROOT:
				tgtWidth = this.tag_length + 4;
				break;
			case A:
			case E:
				tgtWidth = this.tag_length + 3;
				break;
			case T:
			case I:
				tgtWidth = this.tag_ID;
				break;
			case X:
				return new int[]{0, 0, 0, 0, 0};
			case UNKNOWN:
			case Edge:
				break;
			}
		}
		int hybridType = type.ordinal();
		
		return new int[]{srcHeight, srcWidth, tgtHeight, tgtWidth, hybridType};
	}
	
	public void updateNodeIDfromNodeIDArray(){
		int[] nodeIDArray = this.getNodeIDArray();
		this.setNodeID(nodeIDArray);
		this.label = this.getContentString() + Arrays.toString(nodeIDArray);
	}
	
	public void setContent(String content){
		String nodeIDArray = Arrays.toString(this.getNodeIDArray());
		
		switch (type){
		case ROOT:
		case A:
		case E:
			this.content = content;
			break;
		case T:
			this.content = content;
			break;
		case I:
			this.content = content;
			break;
		case X:
			this.content = "";
			break;
		case UNKNOWN:
		case Edge:
			break;
		}
		
		this.label = content + " " + nodeIDArray;
		
	}
	
	public Object getContent(){
		return this.content;
	}
	
	public String getContentString(){
		if (content == null)
			return "";
		
		switch (type){
		case ROOT:
		case A:
		case E:
			return (String)content;

		case T:
			return content.toString();

		case I:
			return (String)content;

		case X:
			return "";
		case UNKNOWN:
		case Edge:
			break;
		}
		
		return "<UNKNOWN TYPE>";
	}
	
	
	public Color getColor(){
		if (picked)
			return Color.PINK;
		
		switch (type) {
		case ROOT:
			return Color.RED;
		case A:
			return Color.ORANGE;
		case E:
			return Color.YELLOW;
		case T:
			return Color.GREEN;
		case I:
			return Color.BLUE;
		case X:
			return Color.LIGHT_GRAY;
		case UNKNOWN:
			return Color.WHITE;
		case Edge:
			break;
		}

		return Color.WHITE;
	}
	
	public void Pick(boolean picked){
		this.picked = picked;
	}
	
	public void beRemoved(){
		if (this.type == Itemtype.T){
			if (this.parent != null){
				this.parent.tag_length--;
				this.parent.update_tag_length_downwards();
				this.parent.update_tag_length_upwards();
			}
		}
	}
}
