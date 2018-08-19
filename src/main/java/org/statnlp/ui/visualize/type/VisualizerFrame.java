/**
 * 
 */
package org.statnlp.ui.visualize.type;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.WindowEvent;

import javax.swing.JCheckBoxMenuItem;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.KeyStroke;

import org.statnlp.hypergraph.NetworkModel;
import org.statnlp.hypergraph.TableLookupNetwork;

/**
 * The frame containing the UI elements for network visualization
 */
public class VisualizerFrame extends JFrame{

	private static final long serialVersionUID = 1980330636235848277L;
	
	private NetworkModel model;
	private VisualizationViewerEngine viewer;
	private int instanceId;
	private boolean showLabeledNetwork;

	public VisualizerFrame(NetworkModel model) {
		this(model, new VisualizationViewerEngine(model.getInstanceParser()));
	}
	
	public VisualizerFrame(NetworkModel model, VisualizationViewerEngine viewer){
		this.model = model;
		this.viewer = viewer;
		init();
	}
	
	private void init(){
		instanceId = 1;
		showLabeledNetwork = true;
		setTitle(makeTitle());
		JFrame frame = this;
		JMenuBar menuBar = new JMenuBar();
		
		// File-related menus
		JMenu file = new JMenu("File");
		JMenuItem save = new JMenuItem("Save");
		save.addActionListener(new ActionListener(){

			@Override
			public void actionPerformed(ActionEvent e) {
				viewer.saveImage(frame.getTitle(), frame.getContentPane().getSize());
			}
			
		});
		save.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_S, ActionEvent.META_MASK));
		JMenuItem exit = new JMenuItem("Exit");
		exit.addActionListener(new ActionListener(){
			@Override
			public void actionPerformed(ActionEvent e) {
				frame.dispose();
				frame.dispatchEvent(new WindowEvent(frame, WindowEvent.WINDOW_CLOSING));
			}
		});
		exit.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_W, ActionEvent.META_MASK));
		file.add(save);
		file.add(exit);
		
		menuBar.add(file);
		
		// Navigation-related menus
		JMenu navigate = new JMenu("Navigate");
		JMenuItem next = new JMenuItem("Next");
		next.addActionListener(new ActionListener(){
			@Override
			public void actionPerformed(ActionEvent e){
				nextNetwork();
			}
		});
		next.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_N, ActionEvent.META_MASK));
		navigate.add(next);
		JMenuItem prev = new JMenuItem("Prev");
		prev.addActionListener(new ActionListener(){
			@Override
			public void actionPerformed(ActionEvent e){
				prevNetwork();
			}
		});
		prev.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_P, ActionEvent.META_MASK));
		navigate.add(prev);
		menuBar.add(navigate);
		
		// Settings menu
		JMenu settings = new JMenu("Settings");
		JCheckBoxMenuItem showLabeled = new JCheckBoxMenuItem("Show Labeled", showLabeledNetwork);
		showLabeled.addActionListener(new ActionListener(){
			@Override
			public void actionPerformed(ActionEvent e) {
				showLabeledNetwork = showLabeled.isSelected();
				refresh();
			}
		});
		showLabeled.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_L, ActionEvent.META_MASK));
		settings.add(showLabeled);
		menuBar.add(settings);
		
		frame.setJMenuBar(menuBar);
		KeyListener keyListener = new KeyListener(){
			@Override
			public void keyTyped(KeyEvent e) {}

			@Override
			public void keyPressed(KeyEvent e) {
				if(e.getKeyCode() == KeyEvent.VK_RIGHT){
					nextNetwork();
				} else if (e.getKeyCode() == KeyEvent.VK_LEFT){
					prevNetwork();
				}
			}

			@Override
			public void keyReleased(KeyEvent e) {}
		};
		
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		JPanel vv = viewer.initVisualizationViewer((TableLookupNetwork)model.getLabeledNetwork(instanceId), frame, getTitle());
		vv.addKeyListener(keyListener);
		frame.add(vv);
		frame.validate();
		frame.pack();
		
		frame.setVisible(true);
	}
	
	private String makeTitle(){
		return String.format("Visualization for %s: Network %d (%s)",
				viewer.getClass().getName(),
				instanceId,
				showLabeledNetwork ? "Labeled" : "Unlabeled");
	}
	
	/**
	 * Redraw the visualization based on the current instance ID and settings.
	 */
	public void refresh(){
		TableLookupNetwork network = null;
		setTitle(makeTitle());
		if(showLabeledNetwork){
			network = (TableLookupNetwork)model.getLabeledNetwork(instanceId);
		} else {
			network = (TableLookupNetwork)model.getUnlabeledNetwork(instanceId);
		}
		viewer.visualizeNetwork(network);
	}
	
	public void prevNetwork(){
		instanceId -= 1;
		if(instanceId <= 0){
			instanceId = model.getInstances().length;
		}
		refresh();
	}
	
	public void nextNetwork(){
		instanceId += 1;
		if(instanceId > model.getInstances().length){
			instanceId = 1;
		}
		refresh();
	}

}
