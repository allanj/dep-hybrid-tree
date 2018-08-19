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
public class SemanticType extends Token{
	
	private static final long serialVersionUID = -5752387687146596764L;
	
	
	public SemanticType(String name){
		super(name);
	}
	
	
	@Override
	public boolean equals(Object o){
		if(o instanceof SemanticType){
			return ((SemanticType)o)._form.equals(this._form);
		}
		return false;
	}
	
	@Override
	public int hashCode(){
		return this._form.hashCode() + 7;
	}
	
	@Override
	public String toString(){
		return "Type:"+this._form;
	}
	
}
