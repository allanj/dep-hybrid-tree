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

import org.statnlp.commons.types.Sentence;
import org.statnlp.example.base.BaseInstance;


/**
 * @author wei_lu
 *
 */
public class SemTextInstance extends BaseInstance<SemTextInstance, Sentence, SemanticForest>{
	
	private static final long serialVersionUID = -8190693110092491424L;
	
	private String _mrl;
	
	public SemTextInstance(int instanceId, double weight) {
		super(instanceId, weight);
	}
	
	public SemTextInstance(int instanceId, double weight, Sentence input, SemanticForest output, String mrl) {
		super(instanceId, weight);
		this.input = input;
		this.output = output;
		this._mrl = mrl;
	}
	
	public String getMRL(){
		return this._mrl;
	}
	
	@Override
	public int size() {
		return this.input.length();
	}
	
	public Sentence duplicateInput(){
		return this.input;
	}
	
	public SemanticForest duplicateOutput() {
		return this.output;
	}
	
}
