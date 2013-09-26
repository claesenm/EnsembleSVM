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
 * cross-validate.cpp
 *
 *      Author: Marc Claesen
 */

/*************************************************************************************************/

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
#include <math.h>
#include <sstream>

/*************************************************************************************************/

using std::vector;
using std::deque;
using std::unique_ptr;
using std::string;
using namespace ensemble;

std::string toolname("cross-validate");

/*************************************************************************************************/

int main(int argc, char **argv)
{
	// initialize help
	std::string helpheader(
			"Creates a cross-validation mask as specified by the user. Labels must be in first column.\n"
			"This mask is a column vector containing a cross-validation index per instance (1:nfolds).\n"
			"The cross-validation mask can be used by the bootstrap and esvm-predict tools.\n"
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

	keyword="-o";
	description="output file containing cross-validation mask";
	CLI::Argument<string> ofname(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&ofname);

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

	description = "number of folds (default 10)";
	keyword = "-nfolds";
	CLI::Argument<unsigned> nfolds(description,keyword,CLI::Argument<unsigned>::Content(1,10));
	allargs.push_back(&nfolds);

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
	if(!ofname.configured()){
		std::cerr << "Output file not specified (see -o)." << std::endl;
		validargs=false;
	}
	if(!labels.configured()){
		std::cerr << "Class labels not specified (see -labels)." << std::endl;
		validargs=false;
	}
	if(!validargs)
		exit_with_err("Invalid command line arguments provided.");

	/*************************************************************************************************/

	// read data file to identify indices of positives and negatives
	std::deque<unsigned> pos, neg;

	std::ifstream datafile(data[0].c_str(),std::ios::in);
	readLabels(datafile,delim[0],labels[0],labels[1],pos,neg,posvall.value());
	// fixme: if user is not using posvall flag, cross-validate pipeline will not function properly

	if(verbose){
		std::cout << "Read " << pos.size() << " positives and " << neg.size() << " negatives from " << data[0] << "." << std::endl;
	}

	// determine fold sizes
	// all folds contain "frac" points
	unsigned posfrac=floor((double)pos.size()/nfolds[0]);
	unsigned negfrac=floor((double)neg.size()/nfolds[0]);
	// some folds contain additional points
	unsigned posrem=pos.size() % nfolds[0];
	unsigned negrem=neg.size() % nfolds[0];

	// sanity check
	if(posfrac==0){
		std::ostringstream oss;
		oss << "Illegal configuration: attempting to use more folds (";
		oss << nfolds[0] << ") than available positive points (" << pos.size() << ")!";
		exit_with_err(oss.str());
	}
	if(negfrac==0){
		std::ostringstream oss;
		oss << "Illegal configuration: attempting to use more folds (";
		oss << nfolds[0] << ") than available negative points (" << neg.size() << ")!";
		exit_with_err(oss.str());
	}

	// initialize RNG
	srand(time(nullptr));

	// random shuffle indices
	std::random_shuffle(pos.begin(),pos.end());
	std::random_shuffle(neg.begin(),neg.end());

	// use a map to store mapping of <index, fold> pairs which are automatically sorted
	unsigned lastpos=0, lastneg=0, numpos=0, numneg=0, i=0;
	std::map<unsigned,unsigned> mapping;
	for(unsigned foldidx=1;foldidx<=nfolds[0];++foldidx){
		// store positives, first posrem folds have 1 additional positive
		numpos=posfrac;
		if(posrem > 0 && foldidx <= posrem)
			++numpos;
		for(i=0;i<numpos;++i){
			mapping.insert(std::make_pair(pos[lastpos+i],foldidx));
		}
		lastpos+=i;

		// store negatives, first negrem folds have 1 additional negative
		numneg=negfrac;
		if(negrem > 0 && foldidx <= negrem)
			++numneg;
		for(i=0;i<numneg;++i){
			mapping.insert(std::make_pair(neg[lastneg+i],foldidx));
		}
		lastneg+=i;
	}

	// write results to output file
	std::ofstream ofile(ofname[0].c_str());
	for(std::map<unsigned,unsigned>::const_iterator I=mapping.begin(),E=mapping.end();I!=E;++I)
		ofile << I->second << std::endl;
	ofile.close();

	return EXIT_SUCCESS;
}
