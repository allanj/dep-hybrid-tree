/** Statistical Natural Language Processing System
    Copyright (C) 2014-2015  Lu, Wei

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.statnlp.example.sempar_bak;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * @author wei_lu
 *
 */
public class SemanticForest implements Serializable{
	
	private static final long serialVersionUID = -7625529039437668047L;
	
	private SemanticForestNode _root;
	
	public SemanticForest(SemanticForestNode root){
		this._root = root;
	}
	
	public double getScore(){
		return this._root.getScore();
	}
	
	public SemanticForestNode getRoot(){
		return this._root;
	}
	
	public int getHeight(){
		return this._root.getHeight();
	}
	
	@Override
	public boolean equals(Object o){
		if(o instanceof SemanticForest){
			SemanticForest forest = (SemanticForest)o;
			return this._root.equals(forest._root);
		}
		return false;
	}
	
	public ArrayList<SemanticForestNode> getAllNodes(){
		ArrayList<SemanticForestNode> nodes = this._root.getAllNodes();
		for(int k = 0; k<nodes.size(); k++)
			nodes.get(k).setId(k);
		return nodes;
	}
	
	public SemanticForestNode getNode(int id){
		return this.getAllNodes().get(id);
	}
	
//	@Override
	public String toString(){
		return this._root.toString();
	}
	
}