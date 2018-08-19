package org.statnlp.ui.visualize.type;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Paint;
import java.awt.Shape;
import java.awt.Stroke;
import java.awt.event.MouseEvent;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Properties;

import javax.imageio.ImageIO;
import javax.swing.JFrame;

import org.apache.commons.collections15.Transformer;
import org.freehep.graphics2d.VectorGraphics;
import org.freehep.graphicsio.pdf.PDFGraphics2D;
import org.statnlp.commons.types.Instance;
import org.statnlp.hypergraph.TableLookupNetwork;
import org.statnlp.util.instance_parser.InstanceParser;

import edu.uci.ics.jung.algorithms.layout.FRLayout;
import edu.uci.ics.jung.algorithms.layout.Layout;
import edu.uci.ics.jung.algorithms.layout.SpringLayout;
import edu.uci.ics.jung.algorithms.layout.StaticLayout;
import edu.uci.ics.jung.visualization.RenderContext;
import edu.uci.ics.jung.visualization.VisualizationImageServer;
import edu.uci.ics.jung.visualization.VisualizationViewer;
import edu.uci.ics.jung.visualization.control.CrossoverScalingControl;
import edu.uci.ics.jung.visualization.control.DefaultModalGraphMouse;
import edu.uci.ics.jung.visualization.control.ScalingGraphMousePlugin;
import edu.uci.ics.jung.visualization.control.TranslatingGraphMousePlugin;
import edu.uci.ics.jung.visualization.renderers.Renderer;
import edu.uci.ics.jung.visualization.renderers.Renderer.VertexLabel.Position;

public class VisualizationViewerEngine {

	public static boolean DEBUG = true;
	
	static int layout_width = 1024;

	static int layout_height = 768;
	
	static int margin_width = 50;
	
	static int margin_height = 50;

	static double span_width = 50;

	static double span_height = 50.0;
	
	static double offset_width = 100;
	
	static double offset_height = 100;
	
	static double node_size = 30;
	
	static double scale = 1;
	
	protected Layout<VNode, VLink> layout;
	
	protected VisualizationViewer<VNode, VLink> vv;
	
	protected VisualizeGraph vg = new VisualizeGraph();
	
	//node_id to label
	protected HashMap<Long, String> labelMap = new HashMap<Long, String>();
	
	//node_type/node_id to color
	protected Color[] colorMap = null;
	
	//node_id to coordinate
	protected HashMap<Long, Point2D> coordinateMap = new HashMap<Long, Point2D>();
	
	protected InstanceParser instanceParser;
	
	protected TableLookupNetwork network;
	
	protected Instance instance;
	
	protected Object inputs;
	
	protected Object outputs;
	
	public VisualizationViewerEngine(InstanceParser instanceParser) {
		this.instanceParser =  instanceParser;
		initTypeColorMapping();
	}
	
	protected void initTypeColorMapping() {
		colorMap = new Color[9];
		colorMap[0] = Color.WHITE;
		colorMap[1] = Color.MAGENTA;
		colorMap[2] = Color.PINK;
		colorMap[3] = Color.YELLOW;
		colorMap[4] = Color.GREEN;
		colorMap[5] = Color.LIGHT_GRAY;
		colorMap[6] = Color.CYAN;
		colorMap[7] = Color.WHITE;
		colorMap[8] = Color.ORANGE;
	}
	
	protected void initNodeColor(VisualizeGraph vg){
		if (colorMap != null)
		for(VNode node : vg.getNodes()){
			node.typeID = node.ids[4];
			node.color = colorMap[node.typeID];
		}
	}
	
	protected double x_mapping(double x){
		return x;
	}
	
	protected double y_mapping(double y){
		return y;
	}
	
	protected void initNodeCoordinate(VisualizeGraph vg){
		for(VNode node : vg.getNodes()){
			int[] ids = node.ids;
			node.point = new Point2D.Double(x_mapping(ids[0]) * span_width + offset_width, y_mapping(ids[1]) * span_height + offset_height);
			layout.setLocation(node, node.point);
			layout.lock(node, true);
		}
	}
	
	protected String label_mapping(VNode node){
		return Arrays.toString(node.ids);
	}
	
	protected void initNodeLabel(VisualizeGraph vg){
		for(VNode node : vg.getNodes()){
			node.label = label_mapping(node);
		}
	}
	
	protected void initData(){
		
	}
	
	/**
	 * Update the viewer to visualize the given network
	 * @param network The network to be visualized
	 */
	public void visualizeNetwork(TableLookupNetwork network){
		this.network = network;
		long[] nodes_arr = Arrays.copyOfRange(network.getAllNodes(), 0, network.countNodes());
		int[][][] childrens_arr = Arrays.copyOfRange(network.getAllChildren(), 0, network.countNodes());
		instance = network.getInstance();
		inputs = network.getInstance().getInput();
		outputs = network.getInstance().getOutput();
		
		initData();
		
		vg.clear();
		vg.buildArrayToGraph(nodes_arr, childrens_arr);
		
		initNodeColor(vg);
		initNodeLabel(vg);
		
		layout = getLayout("static");
		vv.setGraphLayout(layout);
		
		initNodeCoordinate(vg);
	}
	
	public VisualizationViewer<VNode, VLink> initVisualizationViewer(TableLookupNetwork network, JFrame frame, String title){
		this.network = network;
		long[] nodes_arr = Arrays.copyOfRange(network.getAllNodes(), 0, network.countNodes());
		int[][][] childrens_arr = Arrays.copyOfRange(network.getAllChildren(), 0, network.countNodes());
		instance = network.getInstance();
		inputs = network.getInstance().getInput();
		outputs = network.getInstance().getOutput();
		
		initData();
		
		vg.clear();
		vg.buildArrayToGraph(nodes_arr, childrens_arr);
		
		initNodeColor(vg);
		initNodeLabel(vg);
	
		initVisualizationViewer(layout_width, layout_height, margin_width, margin_height, "static");
		
		initNodeCoordinate(vg);
		
		DefaultModalGraphMouse<VNode, VLink> gm = new DefaultModalGraphMouse<VNode, VLink>();
//		gm.setMode(ModalGraphMouse.Mode.TRANSFORMING);
		gm.add(new TranslatingGraphMousePlugin(MouseEvent.BUTTON1_MASK));
		gm.add(new ScalingGraphMousePlugin(new CrossoverScalingControl(), 0, 1.075f, 1/1.075f));
		vv.setGraphMouse(gm);
		return vv;
	}

	protected void setSize(int layout_width, int layout_height, int margin_width, int margin_height){
		layout.setSize(new Dimension(layout_width, layout_height)); // sets the initial size of the layout space
		vv.setPreferredSize(new Dimension(layout_width - margin_width, layout_height - margin_height)); // Sets the viewing area size
	}
	
	protected Layout<VNode, VLink> getLayout(String layout_type){
		if (layout_type.equals("FR")){
			return new FRLayout<VNode, VLink>(vg.g);
		} else if (layout_type.equals("Spring")){
			return new SpringLayout<VNode, VLink>(vg.g);
		} else {
			return new StaticLayout<VNode, VLink>(vg.g);
		}
	}
	
	protected void initVisualizationViewer(int layout_width, int layout_height, int margin_width, int margin_height, String layout_type){
		layout = getLayout(layout_type);
		
		vv = new VisualizationViewer<VNode, VLink>(layout);
		
		this.setSize(layout_width, layout_height, margin_width, margin_height);
		vv.setBackground(Color.WHITE);

		vv.getRenderContext().setVertexFillPaintTransformer(vertexPaint);
		vv.getRenderContext().setVertexLabelTransformer(vertexLabel);
		vv.getRenderContext().setVertexShapeTransformer(vertexShape);
		vv.getRenderContext().setVertexFontTransformer(vertexFont);
		vv.getRenderContext().setVertexDrawPaintTransformer(vertexBorder);
		
		vv.getRenderContext().setEdgeFillPaintTransformer(edgePaint);
		// vv.getRenderContext().setEdgeLabelTransformer(edgeLabel);
		vv.getRenderContext().setEdgeStrokeTransformer(edgeStroke);
		vv.getRenderer().getVertexLabelRenderer().setPosition(Position.CNTR);
//		vv.getRenderer().setVertexRenderer(vertexRender);
	}
	
	public Renderer.Vertex<VNode, VLink> vertexRender = new Renderer.Vertex<VNode, VLink>(){
		@Override
		public void paintVertex(RenderContext<VNode, VLink> rc, Layout<VNode, VLink> arg1, VNode node) {
			Shape shape = new Ellipse2D.Double(node.point.getX(), node.point.getY(), 40, 40);
			rc.getGraphicsContext().fill(shape);
			rc.getGraphicsContext().setPaint(node.color);
		}
	};
	
	public Transformer<VNode, Shape> vertexShape = new Transformer<VNode, Shape>(){
		@Override
		public Shape transform(VNode node) {
			return new Ellipse2D.Double(-15, -15, node_size, node_size);
		}
	};
	
	public Transformer<VNode, Font> vertexFont = new Transformer<VNode, Font>(){
		@Override
		public Font transform(VNode node) {
			return new Font("Calibri", Font.BOLD, 8);
		}
	};

	public Transformer<VNode, String> vertexLabel = new Transformer<VNode, String>() {
		@Override
		public String transform(VNode node) {
			return node.label;
		}
	};
	
	public Transformer<VNode, Paint> vertexPaint = new Transformer<VNode, Paint>() {
		@Override
		public Paint transform(VNode node) {
			return node.color;
		}
	};
	
	public Transformer<VNode, Paint> vertexBorder = new Transformer<VNode, Paint>() {
		@Override
		public Paint transform(VNode node) {
			return Color.BLACK;
		}
	};

	public static Transformer<VLink, Paint> edgePaint = new Transformer<VLink, Paint>() {
		@Override
		public Paint transform(VLink i) {
			return VLink.commoncolor[i.hyperlink.id % VLink.commoncolor.length];
		}
	};

	public static Transformer<VLink, String> edgeLabel = new Transformer<VLink, String>() {
		@Override
		public String transform(VLink i) {
			return "Hyper ID: " + i.getHyperID();
		}
	};

	public static Transformer<VLink, Stroke> edgeStroke = new Transformer<VLink, Stroke>() {
//		float dash[] = { 10.0f };

		@Override
		public Stroke transform(VLink i) {
			return new BasicStroke(0.6f);
			 //return new BasicStroke(1.0f, BasicStroke.CAP_BUTT,BasicStroke.JOIN_MITER, 10.0f, dash, 0.0f);
		}
	};
	
	public void saveImage(String filename, Dimension d){
		saveAsImage(filename, d);
		saveAsPDF(filename);                  
	}

	private void saveAsImage(String filename, Dimension d) {
		if (d == null){
			d = vv.getGraphLayout().getSize();
		}
		
		// Create the VisualizationImageServer
		VisualizationImageServer<VNode, VLink> vis = new VisualizationImageServer<VNode, VLink>(vv.getGraphLayout(), d);
		vis.setRenderContext(vv.getRenderContext());
		vis.getRenderer().getVertexLabelRenderer().setPosition(Position.CNTR);
		
		// Create the buffered image
		BufferedImage image = (BufferedImage) vis.getImage(
				new Point2D.Double(d.getWidth() / 2, d.getHeight() /2 ), d);

		// Write image to a png file
		File outputfile = new File(filename + ".png");

		try {
		    ImageIO.write(image, "png", outputfile);
		    System.out.println(filename + ".png is saved");
		} catch (IOException e) {
		    // Exception handling
		}
	}

	private void saveAsPDF(String filename) {
		Properties p = new Properties(); 
		p.setProperty("PageSize","A4"); 

		VectorGraphics g;
		try {
			g = new PDFGraphics2D(new File(filename + ".pdf"), vv);
			g.setProperties(p); 
			g.startExport(); 
			vv.print(g); 
			g.endExport();
			System.out.println(filename + ".pdf is saved");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

}
