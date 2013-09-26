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
 * esvm-edit.cpp
 *
 *      Author: Marc Claesen
 */

/*************************************************************************************************/

#include "config.h"
#include "CLI.hpp"
#include "Util.hpp"
#include "pipeline/pipelines.hpp"
#include "BinaryWorkflow.hpp"
#include "LibSVM.hpp"

#include <iostream>
#include <sstream>
#include <fstream>

/*************************************************************************************************/

using std::unique_ptr;
using std::string;
using namespace ensemble;

std::string toolname("esvm-edit");

/*************************************************************************************************/

std::unique_ptr<SVMModel> readLIBSVM(const std::string& fname){
	std::unique_ptr<svm_model> libsvm(svm_load_model(fname.c_str()));
	return LibSVM::convert(std::move(libsvm));
}

std::pair<std::vector<double>,double> readLIBLINEAR(const std::string& fname){
	std::ifstream file(fname);

	std::pair<std::vector<double>, double> p;
	size_t nr_feature=0;

	for(std::string line; getline(file,line);){
		std::istringstream in(line);
		std::string keyword;
		in >> keyword;
		if(keyword.compare("bias")==0)
			in >> p.second;
		if(keyword.compare("nr_feature")==0)
			in >> nr_feature;
		if(keyword.compare("w")==0)
			break;
	}

	p.first.reserve(nr_feature);
	double d;
	for(size_t i=0;i<nr_feature;++i){
		file >> d;
		p.first.push_back(d);
	}

	for(size_t i=0;i<nr_feature;++i)
		std::cout << p.first[i] << " ";

	return std::move(p);
}

/**
 * Reads the content of fname into the resulting vector.
 *
 * If offset=true, an offset is read from the second line and placed at the end of the result.
 */
std::pair<std::vector<double>,double> readFile(const std::string& fname, bool offset=false){
	std::ifstream file(fname);
	std::string line;

	assert(file && "Error opening file.");
	getline(file,line);
	std::istringstream ss(line);

	std::pair<std::vector<double>, double> p
		(std::vector<double>(std::istream_iterator<double>(ss),std::istream_iterator<double>()),0.0);
	if(offset){
		assert(file && "Error reading offset from second line in file.");
		file >> p.second;
	}

	return std::move(p);
}

/*************************************************************************************************/

int main(int argc, char **argv)
{
	std::cin.sync_with_stdio(false);

	std::string helpheader(
			"Edit elements of an existing workflow with given configuration.\n"
			"\nArguments:\n"
	), helpfooter("");


	// intialize arguments
	deque<CLI::BaseArgument*> allargs;

	string description;
	string keyword;
	vector<string> multilinedesc;

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

	keyword = "-model";
	description = "model file, containing the workflow to be editted";
	CLI::Argument<string> model(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&model);

	description = "output file for edited workflow (default: overwrite file in -model)";
	keyword = "-o";
	CLI::Argument<string> ofile(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&ofile);

	keyword = "-print";
	multilinedesc.push_back("print part of the binary workflow to standard output");
	multilinedesc.push_back("when printing, no modifications are made to the workflow");
	multilinedesc.push_back("1 -- preprocessing");
	multilinedesc.push_back("2 -- predictor");
	multilinedesc.push_back("3 -- postprocessing");
	multilinedesc.push_back("4 -- threshold");
	CLI::Argument<unsigned> print(multilinedesc,keyword,CLI::Argument<unsigned>::Content(1,0));
	allargs.push_back(&print);
	multilinedesc.clear();

	description = "file containing linear preprocessing (output by svm-scale)";
	keyword = "-pre";
	CLI::Argument<string> preprocessing(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&preprocessing);

	keyword = "-post";
	multilinedesc.push_back("set the postprocessing aggregation scheme to use in the workflow");
	multilinedesc.push_back("1 -- *  majority voting: f(x)=alpha*x / sum(alpha)");
	multilinedesc.push_back("2 -- *  logistic regression: f(x)=1/(1+exp[-(alpha*x+b)])");
	multilinedesc.push_back("3 -- ** LIBSVM model (binary classifier)");			// todo
	multilinedesc.push_back("4 -- ** LIBLINEAR model (binary classifier)"); 		// todo
	multilinedesc.push_back("*  final threshold automatically set to 0.5"); 			// todo
	multilinedesc.push_back("** final threshold automatically set to 0.0"); 			// todo
	CLI::Argument<unsigned> post(multilinedesc,keyword,CLI::Argument<unsigned>::Content(1,0));
	allargs.push_back(&post);
	multilinedesc.clear();

	keyword = "-postpars";
	multilinedesc.push_back("file containing parameters of the selected postprocessing scheme");
	multilinedesc.push_back("file content depends on choice of postprocessing scheme");
	multilinedesc.push_back("post=1 -- optional white spaced file:");
	multilinedesc.push_back("          line 1: alpha coefficient per base model");
	multilinedesc.push_back("post=2 -- optional white spaced file:");
	multilinedesc.push_back("          line 1: alpha coefficient per base model");
	multilinedesc.push_back("          line 2: b");
	multilinedesc.push_back("post=3 -- mandatory LIBSVM model file");
	multilinedesc.push_back("post=4 -- mandatory LIBLINEAR model file");
	CLI::Argument<string> pars(multilinedesc,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&pars);
	multilinedesc.clear();

	description = "set the final decision threshold used by the workflow";
	keyword = "-threshold";
	CLI::Argument<double> threshold(description,keyword,CLI::Argument<double>::Content(1,0.5));
	allargs.push_back(&threshold);

	ParseCLI(argv,argc,1,allargs);

	if(help.configured() || help2.configured())
		exit_with_help(allargs,helpheader,helpfooter,true);
	if(version.configured() || version2.configured())
		exit_with_version(toolname);

	if(argc==1)
		exit_with_help(allargs,helpheader,helpfooter);

	// verify command line arguments
	bool err=false;
	if(!model.configured()){
		std::cerr << "Model file not configured (see -model).";
		err=true;
	}
	if(!post.configured() && pars.configured()){
		std::cerr << "Postprocessing scheme not specified but parameters given.";
		err=true;
	}
	if(err)
		exit_with_err("Invalid configuration specified via command line.");

	std::unique_ptr<BinaryWorkflow> flow;

	/*************************************************************************************************/

	// read workflow

	{
		auto modelptr = BinaryModel::load(model[0]);
		flow.reset(dynamic_cast<BinaryWorkflow*>(modelptr.release()));
		if(!flow.get()) exit_with_err("esvm-edit can only be used on binary workflows.");
	}

	/*************************************************************************************************/

	// do prints if requested

	if(print.configured()) switch(print[0]){
	case 0 :
		exit(EXIT_FAILURE);
	case 1 :
		flow->print_preprocessing(std::cout);
		exit(EXIT_SUCCESS);
	case 2 :
		flow->print_predictor(std::cout);
		exit(EXIT_SUCCESS);
	case 3 :
		flow->print_postprocessing(std::cout);
		exit(EXIT_SUCCESS);
	case 4 :
		flow->print_threshold(std::cout);
		exit(EXIT_SUCCESS);
	default:
		exit(EXIT_FAILURE);
	}

	/*************************************************************************************************/

	// make requested modifications

	bool modified=false;

	// include preprocessing if specified

	if(preprocessing.configured()){
		// todo
		modified=true;
	}

	// modify postprocessing to the desired parametrized scheme

	if(post.configured()){
		std::unique_ptr<BinaryWorkflow::Postprocessing> postproc;
		unsigned numinputs = flow->num_predictor_outputs();
		switch(post[0]){
		case 1:
		{
			pipeline::Factory<pipeline::MajorityVote> f;
			if(pars.configured()){
				auto pair = readFile(pars[0],false);
				postproc.reset(f(std::move(pair.first)).release());
			}else{
				postproc.reset(f(numinputs).release());
			}
			flow->set_threshold(0.5);
			break;
		}
		case 2:
		{
			pipeline::Factory<pipeline::LogisticRegression> f;
			if(pars.configured()){
				auto pair = readFile(pars[0],true);
				postproc.reset(f(std::move(pair.first),pair.second).release());
			}else{
				postproc.reset(f(numinputs).release());
			}
			flow->set_threshold(0.5);
			break;
		}
		case 3:
		{
			assert(pars.configured() && "LIBSVM model file must be specified.");
			auto svm = readLIBSVM(pars[0]);
			pipeline::Factory<pipeline::BinarySVMAggregation> f;
			postproc.reset(f(std::move(svm)).release());
			flow->set_threshold(0.0);
			break;
		}
		case 4:
		{
			assert(pars.configured() && "LIBLINEAR model file must be specified.");
			auto pair = readLIBLINEAR(pars[0]);
			pipeline::Factory<pipeline::LinearAggregation> f;
			postproc.reset(f(std::move(pair.first),pair.second).release());
			flow->set_threshold(0.0);
			break;
		}
		default:
			exit_with_err("Invalid number specified in for -aggregation.");
		}
		flow->set_postprocessing(std::move(postproc));
		modified=true;
	}

	// modify threshold if specified

	if(threshold.configured()){
		flow->set_threshold(threshold[0]);
		modified=true;
	}

	/*************************************************************************************************/

	// save modified workflow and exit

	if(modified){
		std::string outputfilename = ofile.configured() ? ofile[0] : model[0];
		std::ofstream ofstream(outputfilename.c_str());
		ofstream << *flow;
		ofstream.close();
	}

	exit(EXIT_SUCCESS);
}
