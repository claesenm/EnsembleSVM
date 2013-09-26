/**
 *  Copyright (C) 2013 KU Leuven
 *
 *  This file is part of EnsembleSVM.
 *
 *  EnsembleSVM is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published
 *  by the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  EnsembleSVM is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with EnsembleSVM.  If not, see <http://www.gnu.org/licenses/>.
 *
 * io.cpp
 *
 *      Author: Marc Claesen
 */

#include "SparseVector.hpp"
#include "Kernel.hpp"
#include "Models.hpp"
#include "Ensemble.hpp"
#include "Util.hpp"
#include "LibSVM.hpp"
#include "io.hpp"
#include "DataFile.hpp"
#include "svm.h"
#include <sstream>
#include <string>
#include <stdio.h>
#include <cstring>
#include <iostream>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>

using std::endl;
using std::string;
using std::unique_ptr;
using std::istringstream;
using std::istream;
using namespace ensemble;

namespace ensemble{

/**
 * General io functions
 */


/**
 * 1. Attempts to read an SVMEnsemble.
 * 2. Attempts to read an SVMModel.
 * 3. Attempts to read a LibSVM model and converts it to an SVMModel.
 */
//unique_ptr<Model> Model::load(const string &fname){
//	std::ifstream file(fname.c_str(),std::ios::in);
//	unique_ptr<Model> model;
//	try{
//		unique_ptr<SVMEnsemble> ensemble = SVMEnsemble::read(file);
//		model = unique_ptr<Model>(ensemble.get());
//		ensemble.release();
//	}catch(invalid_file_exception &e){
//		// first try block has read first line, we must reset position
//		file.clear();
//		file.seekg(0, std::ios_base::beg);
//
//		unique_ptr<SVMModel> svmmodel;
//		try{
//			svmmodel = SVMModel::read(file,nullptr);
//		}catch(invalid_file_exception &e){
//			file.close();
//			svm_model *libsvmmodel = svm_load_model(fname.c_str());
//			svmmodel = LibSVM::convert(unique_ptr<svm_model>(libsvmmodel));
//		}
//		model = unique_ptr<Model>(svmmodel.get());
//		svmmodel.release();
//	}
//	if(file.is_open())
//		file.close();
//	return model;
//}

unique_ptr< std::deque<double> > ReadIndividualPenaltiesFromFile(const std::string &fname){

	std::ifstream file(fname.c_str(),std::ios::in);

	// read values, 1 per line
	double value;
	unique_ptr< std::deque<double> > penalties(new std::deque<double>());
	while(file.good()){
		file >> value;
		penalties->push_back(value);
	}

	file.close();

	return penalties;
}

void readBootstrapMask(const std::string &fname, std::vector< std::list<unsigned>*> &mask, char delim){
	std::istringstream liness, tokenss;
	std::ifstream file(fname.c_str(),std::ios::in);

	std::string line, token;

	// mask.size() == nummodels
	for(std::vector< std::list<unsigned>*>::iterator I=mask.begin(),E=mask.end();I!=E;++I){
		getline(file,line);
		liness.clear();
		liness.str(line);
		*I=new std::list<unsigned>;

		unsigned idx;
		while(liness.good()){
			token.clear();
			tokenss.clear();
			getline(liness,token,delim);
			tokenss.str(token);
			tokenss >> idx;
			(*I)->push_back(idx);
		}
	}

	file.close();
}

void readWeightMask(const std::string &fname, std::vector< SparseVector* > &mask, char delim){
	std::ifstream file(fname.c_str(),std::ios::in);

	// mask.size() == nummodels
	for(std::vector< SparseVector*>::iterator I=mask.begin(),E=mask.end();I!=E;++I){
		unique_ptr<SparseVector> ptr(SparseVector::read(file,false));
		*I=ptr.get();
		ptr.release();
	}
	file.close();
}

void readLabels(std::ifstream &file, char delim, const std::string &poslabel, const std::string &neglabel, std::deque<unsigned> &pos, std::deque<unsigned> &neg, bool posvall){
	string chunk;

	unsigned idx=1;
	while(file.good()){

		// read first column
		chunk.clear();
		getline(file,chunk,delim);
		if(chunk.empty())
			break;

		if(chunk.compare(poslabel)==0)
			pos.push_back(idx);
		else if(posvall)
			neg.push_back(idx);
		else if(chunk.compare(neglabel)==0)
			neg.push_back(idx);
//		else
//			exit_with_err(std::string("Encountered unknown label: ")+chunk+" ... missing -posvall flag?");
		// when encountering an unkown label, ignore it and parse rest of the data

		// ignore rest
		file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		++idx;
	}
}

void readCrossvalMask(const std::string &filename, std::map<unsigned, std::deque<unsigned> > &mask){
	std::ifstream file(filename.c_str());
	std::string line;
	std::istringstream iss;

	unsigned linenum=1, fold;
	std::map<unsigned, std::deque<unsigned> >::iterator F;
	while(file.good()){
		getline(file,line);
		iss.clear();
		iss.str(line);
		iss >> fold;

		F=mask.find(fold);
		if(F!=mask.end()){
			F->second.push_back(linenum);
		}else{
			std::deque<unsigned> thisfold(1,linenum);
			mask.insert(std::make_pair(fold,thisfold));
		}

		++linenum;
	}
	file.close();
}


} // ensemble namespace
