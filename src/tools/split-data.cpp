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
 * split-data.cpp
 *
 *      Author: Marc Claesen
 */

#include "CLI.hpp"
#include "io.hpp"
#include "Util.hpp"
#include "DataFile.hpp"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <errno.h>
#include <deque>
#include <vector>
#include <algorithm>

using std::vector;
using std::deque;
using std::unique_ptr;
using std::string;
using namespace ensemble;

std::string toolname("split-data");

int main(int argc, char **argv)
{
	// initialize help
	std::string helpheader(
			"Splits the data set into a mutually exclusive training and testing set, as specified by the user.\n"
			"\n"
			"Options:\n"
	), helpfooter("");

	// intialize arguments
	std::deque<CLI::BaseArgument*> allargs;
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

	keyword="-data";
	description="data file (must be labeled)";
	CLI::Argument<string> data(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&data);

	keyword="-train";
	description="output training file";
	CLI::Argument<string> trainfname(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&trainfname);

	keyword="-test";
	description="output testing file";
	CLI::Argument<string> testfname(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&testfname);

	keyword = "-labels";
	vector<string> multilinedesc;
	multilinedesc.push_back("labels per class");
	multilinedesc.push_back("<positive label> <negative label> (default '+1 -1')");
	CLI::Argument<string>::Content labelscontent(2,"+1");
	labelscontent[1]=std::string("-1");
	CLI::Argument<string> labels(multilinedesc,keyword,labelscontent);
	allargs.push_back(&labels);
	multilinedesc.clear();

	description = "treat all labels != positive as negative (e.g. 1 vs all multiclass)";
	keyword = "-posvall";
	CLI::FlagArgument posvall(description,keyword,false);
	allargs.push_back(&posvall);

	description = "number of positives in training set";
	keyword = "-npostrain";
	CLI::Argument<unsigned> npostrain(description,keyword,CLI::Argument<unsigned>::Content(1,0));
	allargs.push_back(&npostrain);

	description = "number of negatives in training set";
	keyword = "-nnegtrain";
	CLI::Argument<unsigned> nnegtrain(description,keyword,CLI::Argument<unsigned>::Content(1,0));
	allargs.push_back(&nnegtrain);

	description = "number of positives in test set";
	keyword = "-npostest";
	CLI::Argument<unsigned> npostest(description,keyword,CLI::Argument<unsigned>::Content(1,0));
	allargs.push_back(&npostest);

	description = "number of negatives in test set";
	keyword = "-nnegtest";
	CLI::Argument<unsigned> nnegtest(description,keyword,CLI::Argument<unsigned>::Content(1,0));
	allargs.push_back(&nnegtest);

	description = "fraction of entire data set to use in test";
	keyword = "-testfrac";
	CLI::Argument<double> testfrac(description,keyword,CLI::Argument<double>::Content(1,0));
	allargs.push_back(&testfrac);

	description = "column delimiter in data file (default: ' ')";
	keyword = "-delim";
	char d=*" ";
	CLI::Argument<char> delim(description,keyword,CLI::Argument<char>::Content(1,d));
	allargs.push_back(&delim);

	description = "enables verbose mode, which outputs various information to stdout";
	keyword = "-v";
	CLI::FlagArgument verbose(description,keyword,false);
	allargs.push_back(&verbose);

	if(argc==1)
		exit_with_help(allargs,helpheader,helpfooter);

	ParseCLI(argv,argc,1,allargs);

	if(help.configured() || help2.configured())
		exit_with_help(allargs,helpheader,helpfooter,true);
	if(version.configured() || version2.configured())
		exit_with_version(toolname);

	bool validargs=true;
	if(!data.configured()){
		std::cerr << "Data file not specified (see -data)." << std::endl;
		validargs=false;
	}
	if(!trainfname.configured()){
		std::cerr << "Output training file not specified (see -train)." << std::endl;
		validargs=false;
	}
	if(!testfname.configured()){
		std::cerr << "Output test file not specified (see -test)." << std::endl;
		validargs=false;
	}
	if(!labels.configured()){
		std::cerr << "Class labels not specified (see -labels)." << std::endl;
		validargs=false;
	}
	if(!testfrac.configured()){
		if(!npostrain.configured()){
			std::cerr << "Test fraction and number of positives in training set unspecified (see -testfrac & -npostrain)." << std::endl;
			validargs=false;
		}
		if(!nnegtrain.configured()){
			std::cerr << "Test fraction and number of negatives in training set unspecified (see -testfrac & -nnegtrain)." << std::endl;
			validargs=false;
		}
		if(!npostest.configured()){
			std::cerr << "Test fraction and number of positives in test set unspecified (see -testfrac & -npostest)." << std::endl;
			validargs=false;
		}
		if(!nnegtest.configured()){
			std::cerr << "Test fraction and number of negatives in test set unspecified (see -testfrac & -nnegtest)." << std::endl;
			validargs=false;
		}
	}
	if(!validargs)
		exit_with_err("Invalid command line arguments provided.");

	// read data file to identify indices of positives and negatives
	std::deque<unsigned> pos, neg;

	std::ifstream datafile(data[0].c_str(),std::ios::in);
	readLabels(datafile,delim[0],labels[0],labels[1],pos,neg,posvall.value());

	if(verbose){
		std::cout << "Read " << pos.size() << " positives and " << neg.size() << " negatives from " << data[0] << "." << std::endl;
	}

	unsigned nptr, npte, nntr, nnte;
	if(testfrac){
		nptr=static_cast<unsigned>((1-testfrac[0])*pos.size()+0.5);
		nntr=static_cast<unsigned>((1-testfrac[0])*neg.size()+0.5);
		npte=pos.size()-nptr;
		nnte=neg.size()-nntr;
	}else{
		nptr=npostrain[0];
		nntr=nnegtrain[0];
		npte=npostest[0];
		nnte=nnegtest[0];
	}

	if(verbose){
		std::cout << "npostrain " << nptr << ", nnegtrain " << nntr << ", npostest " << npte << ", nnegtest " << nnte << std::endl;
	}

	// sanity check
	if(nptr+npte > pos.size())
		exit_with_err("Sum of positives in training+testing set specified is larger than total amount of positives in data file.");
	if(nntr+nnte > neg.size())
		exit_with_err("Sum of negatives in training+testing set specified is larger than total amount of negatives in data file.");

	// initialize RNG
	srand(time(nullptr));

	// random shuffle indices
	std::random_shuffle(pos.begin(),pos.end());
	std::random_shuffle(neg.begin(),neg.end());

	std::vector<unsigned> trainidx(nptr+nntr), testidx(npte+nnte);
	unsigned posidx=0, negidx=0, traini=0, testi=0;
	for(;posidx<nptr;++posidx,++traini)
		trainidx[traini]=pos[posidx];
	for(;negidx<nntr;++negidx,++traini)
		trainidx[traini]=neg[negidx];
	for(;posidx<nptr+npte;++posidx,++testi)
		testidx[testi]=pos[posidx];
	for(;negidx<nntr+nnte;++negidx,++testi)
		testidx[testi]=neg[negidx];

	// sort for efficient copying from original data
	std::sort(trainidx.begin(),trainidx.end());
	std::sort(testidx.begin(),testidx.end());

	datafile.clear();
	datafile.seekg(0, std::ios::beg);
	std::ofstream trainfile(trainfname[0].c_str(),std::ios::out);
	std::ofstream testfile(testfname[0].c_str(),std::ios::out);

	std::vector<unsigned>::const_iterator Itr=trainidx.begin(),Etr=trainidx.end(),
			Ite=testidx.begin(),Ete=testidx.end();

	unsigned idx=1;
	std::string line;
	while(Itr!=Etr && Ite!=Ete){
		if(*Itr < *Ite){
			// next sampled instance for training set
			line.clear();
			for(;idx<*Itr;++idx)
				getline(datafile,line);
			++Itr;

			getline(datafile,line);
			++idx;
			trainfile << line << std::endl;
		}else{
			// next sampled instance for test set
			line.clear();
			for(;idx<*Ite;++idx)
				getline(datafile,line);
			++Ite;

			getline(datafile,line);
			++idx;
			testfile << line << std::endl;
		}
	}
	while(Itr!=Etr){
		// next sampled instance for training set
		line.clear();
		for(;idx<*Itr;++idx)
			getline(datafile,line);
		++Itr;

		getline(datafile,line);
		++idx;
		trainfile << line << std::endl;
	}
	while(Ite!=Ete){
		// next sampled instance for test set
		line.clear();
		for(;idx<*Ite;++idx)
			getline(datafile,line);
		++Ite;

		getline(datafile,line);
		++idx;
		testfile << line << std::endl;
	}

	trainfile.close();
	testfile.close();
	datafile.close();
	return EXIT_SUCCESS;
}
