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
 * merge-models.cpp
 *
 *      Author: Marc Claesen
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include "CLI.hpp"
#include "Util.hpp"
#include "SparseVector.hpp"
#include "Models.hpp"
#include "Ensemble.hpp"
#include "io.hpp"
#include "LibSVM.hpp"
#include "BinaryWorkflow.hpp"
#include <errno.h>

using std::unique_ptr;
using std::string;
using namespace ensemble;

std::string toolname("merge-models");

unique_ptr<SVMModel> loadSVMModel(const std::string& fname){
	unique_ptr<SVMModel> model;
	{
		std::ifstream file(fname);
		auto binmodel = BinaryModel::deserialize(file);
		if(binmodel.get()){
			SVMModel* ptr = dynamic_cast<SVMModel*>(binmodel.release());
			assert(ptr && "Error: file did not contain an SVMModel.");
			model.reset(ptr);
		}
	}
	if(!model.get()){
		unique_ptr<svm_model> libsvm(svm_load_model(fname.c_str()));
		unique_ptr<SVMModel> mod = LibSVM::convert(std::move(libsvm));
		model.reset(mod.release());
	}
	return std::move(model);
}

unique_ptr<SVMEnsemble> readFirstModel(const std::string &fname){ // fixme: use factory

	// use factory to load a binary model
	unique_ptr<BinaryModel> model;
	{
		std::ifstream file(fname);
		model = BinaryModel::deserialize(file);
	}

	// if no model was loaded, the file contains a standard libsvm model
	std::unique_ptr<SVMModel> svmmodel;
	if(!model.get()){
		unique_ptr<svm_model> libsvm(svm_load_model(fname.c_str()));
		unique_ptr<SVMModel> mod = LibSVM::convert(std::move(libsvm));
		svmmodel.reset(mod.release());
	}

	if(!svmmodel.get()){
		// check if the loaded model is an ensemble, if so we're done
		if(SVMEnsemble *ens = dynamic_cast<SVMEnsemble*>(model.get())){
			model.release();
			return unique_ptr<SVMEnsemble>(ens);
		}

		// check if the loaded model is a workflow, which may contain an ensemble
		if(BinaryWorkflow *flow = dynamic_cast<BinaryWorkflow*>(model.get())){
			unique_ptr<BinaryModel> predictor = flow->release_predictor();
			if(SVMEnsemble* ens=
					dynamic_cast<SVMEnsemble*>(predictor.get())){
				predictor.release();
				return unique_ptr<SVMEnsemble>(ens);
			}

			if(SVMModel* m = dynamic_cast<SVMModel*>(predictor.get())){
				svmmodel.reset(m);
				predictor.release();
			}else{
				assert(false && "Unknown predictor in binary workflow!");
			}

		}
	}

	unique_ptr<Kernel> kernel(svmmodel->getKernel()->clone()); // fixme: avoid copying of kernel
	unique_ptr<SVMEnsemble> ensemble(new SVMEnsemble(std::move(kernel)));
	ensemble->add(std::move(svmmodel));
	return std::move(ensemble);
}

unique_ptr<SVMEnsemble> mergeRange(const std::string &basename, unsigned startidx, unsigned stopidx){
	if(startidx > stopidx)
		exit_with_err("Start extention > stop extention specified in -range!");

	std::stringstream ss;
	string filename;
	ss.clear();
	ss << basename << startidx;
	filename=ss.str();

	unique_ptr<SVMEnsemble> ensemble = readFirstModel(filename);

	for(unsigned i=startidx+1;i<=stopidx;++i){
		ss.str("");
		ss << basename  << i;
		filename = ss.str();
		unique_ptr<SVMModel> svmmodel = loadSVMModel(filename);
		ensemble->add(std::move(svmmodel));
	}
	return ensemble;
}

int main(int argc, char **argv)
{
	// initialize help
	std::string helpheader(
			"Merges models <model1> and <model2> into an ensemble model with majority voting.\n"
			"OR\n"
			"merges models <basename><start> to <basename><stop> into an ensemble model.\n"
			"Please note that all models must use the same kernel. \n"
			"Base models can be generic SVM models or LIBSVM models. \n\n"
			"Options:\n"
	), helpfooter("");

	// intialize arguments
	std::deque<CLI::BaseArgument*> allargs;

	vector<string> multilinedesc;
	string description, keyword;

	keyword="--help";
	CLI::SilentFlagArgument help(keyword);
	allargs.push_back(&help);

	keyword="--h";
	CLI::SilentFlagArgument help2(keyword);
	allargs.push_back(&help2);

	keyword="--version";
	CLI::SilentFlagArgument version(keyword);
	allargs.push_back(&version);

	keyword="--v";
	CLI::SilentFlagArgument version2(keyword);
	allargs.push_back(&version2);

	keyword="-model1";
	multilinedesc.push_back("first model to merge (in conjunction with -model2)");
	multilinedesc.push_back("can be used to append to an existing ensemble (in -model1)");
	CLI::Argument<string> model1(multilinedesc,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&model1);

	keyword="-model2";
	description="second model to merge (in conjunction with -model1)";
	CLI::Argument<string> model2(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&model2);

	keyword="-base";
	multilinedesc.clear();
	multilinedesc.push_back("model base name to merge a range (in conjunction with -range)");
	multilinedesc.push_back("merges <base><start> to <base><stop>");
	CLI::Argument<string> base(multilinedesc,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&base);

	description="model range: <start> <stop> (in conjunction with -base)";
	keyword="-range";
	CLI::Argument<unsigned> range(description,keyword,CLI::Argument<unsigned>::Content(2,0));
	allargs.push_back(&range);

	description = "output file";
	keyword = "-o";
	CLI::Argument<string> ofile(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&ofile);

	string basename, outname;
	std::stringstream ss;

	if(argc == 1)
		exit_with_help(allargs,helpheader,helpfooter);

	ParseCLI(argv,argc,1,allargs);

	if(help.configured() || help2.configured())
		exit_with_help(allargs,helpheader,helpfooter,true);
	if(version.configured() || version2.configured())
		exit_with_version(toolname);

	unique_ptr<SVMEnsemble> ensemble(nullptr);

	std::cout << model1[0] << std::endl;
	std::cout << model2[0] << std::endl;
	std::cout << ofile[0] << std::endl;

	if(base && range && ofile){
		ensemble=mergeRange(base[0],range[0],range[1]);

	}else if(model1 && model2 && ofile){
		// merging 2 models
		ensemble=readFirstModel(model1[0]);

		unique_ptr<SVMModel> svmmodel=loadSVMModel(model2[0]);
		ensemble->add(std::move(svmmodel));

	}else{
		exit_with_err("Illegal command line options specified.");
	}

	auto flow = defaultBinaryWorkflow(std::unique_ptr<BinaryModel>(ensemble.release()));

	// write out ensemble
	std::ofstream outfile;
	outfile.open(ofile[0].c_str());
	outfile << *flow;
	outfile.close();
}
