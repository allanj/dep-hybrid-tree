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
package org.statnlp.example.depsemtree;

import org.statnlp.commons.types.Token;
/**
 * @author wei_lu
 *
 */
public class HybridPattern extends Token{
	
	private static final long serialVersionUID = -3236262436003910366L;
	
	
	public HybridPattern(String name){
		super(name);
	}
	
	public int minLen(){
		return this._form.length();
	}
	
	public int maxLen(){
		int len = 0;
		for(int k = 0; k<this._form.length(); k++){
			char c = this._form.charAt(k);
			if(c=='X' || c=='Y' || c=='W'){
				return Integer.MAX_VALUE;
			} else {
				len ++;
			}
		}
		return len;
	}
	
	public boolean isA(){
		return this._form.equals("A");
	}
	
	public boolean isB(){
		return this._form.equals("B");
	}
	
	public boolean isC(){
		return this._form.equals("C");
	}
	
	public boolean isw(){
		return this._form.equals("w");
	}
	
	public boolean isW(){
		return this._form.equals("W");
	}
	
	public boolean isX(){
		return this._form.equals("X");
	}
	
	public boolean isY(){
		return this._form.equals("Y");
	}
	
	public char getFormat(int index){
		if(index>=0)
			return this._form.charAt(index);
		return this._form.charAt(this._form.length()+index);
	}
	
	public void setId(int id){
		this._id = id;
	}
	
	@Override
	public int hashCode(){
		return this._form.hashCode() + 7;
	}
	
	@Override
	public boolean equals(Object o){
		if(o instanceof HybridPattern){
			HybridPattern p = (HybridPattern)o;
			return this._form.equals(p._form);
		}
		return false;
	}
	
	
	@Override
	public String toString(){
		return "PATTERN:"+this._form;
	}
	
}
