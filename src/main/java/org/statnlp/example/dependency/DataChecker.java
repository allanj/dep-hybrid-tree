package org.statnlp.example.dependency;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;


public class DataChecker {
	
	/**
	 * Using the heads only to check the projectiveness
	 * @param heads
	 * @return
	 */
	public static boolean checkProjective(List<Integer> heads){
		for (int i = 0; i < heads.size(); i++) {
			int ihead = heads.get(i);
			if (ihead == -1) continue;
			int iSmallIndex = Math.min(i, ihead);
			int iLargeIndex = Math.max(i, ihead);
			for (int j = 0; j < heads.size(); j++) {
				int jhead = heads.get(j);
				if (i==j || jhead == -1) continue;
				int jSmallIndex = Math.min(j, jhead);
				int jLargeIndex = Math.max(j, jhead);
				if(iSmallIndex < jSmallIndex && iLargeIndex < jLargeIndex && jSmallIndex < iLargeIndex) return false;
				if(iSmallIndex > jSmallIndex && jLargeIndex > iSmallIndex && iLargeIndex > jLargeIndex) return false;
			}
		}
		return true;
	}

	public static boolean checkIsTree(List<Integer> heads) {
		HashMap<Integer, List<Integer>> tree = new HashMap<Integer, List<Integer>>();
		for (int i = 0; i < heads.size(); i++) {
			int ihead = heads.get(i);
			if (ihead == -1) continue;
			if (tree.containsKey(ihead)) {
				tree.get(ihead).add(i);
			} else {
				List<Integer> children = new ArrayList<>();
				children.add(i);
				tree.put(ihead, children);
			}
		}
		boolean[] visited = new boolean[heads.size()];
		Arrays.fill(visited, false);
		visited[0] = true;
		traverse(visited, 0, tree);
		for(int i = 0; i < visited.length; i++)
			if (!visited[i])
				return false;
		return true;
	}
	
	
	private static void traverse(boolean[] visited, int parent, HashMap<Integer, List<Integer>> tree) {
		if (tree.containsKey(parent)) {
			for(int child: tree.get(parent)) {
				visited[child] = true;
				traverse(visited, child, tree);
			}
		}
		
	}
	
	
}
