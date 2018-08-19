package org.statnlp.commons.types;

import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.stack.TIntStack;

public class DependencyTree {

private TIntObjectMap<TIntList> tree;
	
	/**
	 * The sentence must be with head index and it's 0-indexed. 
	 * In this case, the root should be indexed by -1.
	 * @param sent
	 */
	public DependencyTree (Sentence sent) {
		this.build(sent);
	}
	
	public void build(Sentence sent) {
		tree = new TIntObjectHashMap<>();
		for (int i = 0; i < sent.length(); i++) {
			this.add(sent.get(i).getHeadIndex(), i);
		}
	}
	
	private void add (int parent, int child) {
		if (tree.containsKey(parent)) {
			TIntList list = tree.get(parent);
			list.add(child);
		} else {
			TIntList list = new TIntArrayList();
			list.add(child);
			tree.put(parent, list);
		}
	}
	
	
	public boolean dfsFind (int val, TIntStack stack) {
		return dfsFind(-1, val, stack);
	}
	
	private boolean dfsFind(int curr, int val, TIntStack stack) {
		stack.push(curr);
		if (curr == val) {
			return true;
		} else {
			TIntList children = tree.get(curr);
			if (children != null) {
				for (int i = 0; i < children.size(); i++) {
					boolean found = dfsFind(children.get(i), val, stack);
					if (found) {
						return true;
					} 
				}
			}
			stack.pop();
			return false;
		}
	}
	
}
