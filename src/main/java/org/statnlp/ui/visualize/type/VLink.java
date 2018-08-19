package org.statnlp.ui.visualize.type;

import java.awt.Color;




public class VLink {
	
	public static int edgeCount = 0;
	
	public static Color[] commoncolor = new Color[]{Color.CYAN, Color.YELLOW, Color.ORANGE, Color.PINK, Color.MAGENTA};
	
    int id;
    int hyperid = -1;
    public VNode Dest = null;
    
    public HyperLink hyperlink = null;
    
    public VLink(int hyperid, VNode Dest, HyperLink hyperlink) {
        this.id = edgeCount++;
        this.hyperid = hyperid;
        this.Dest = Dest;
        this.hyperlink = hyperlink;
    } 
    
    VLink() {
    	this.id = edgeCount++;
    }
    
    public static VLink createEdge(VNode start, VNode end)
    {

    	if (start != null && end != null)
    	{
    		HyperLink hyperlink = start.getHyperLinkInLinearMode();
    		VLink link = new VLink(hyperlink.hyperid, end, hyperlink);
    		hyperlink.addLink(link);
    		return link;
    	}
    	
    	return null;
    }

    public String toString() {
        return "E "+hyperid + "|" + Dest.index ;
    }
    
    public int getHyperID()
    {
    	return this.hyperid;
    }
    
}