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
 * bootstrap.cpp
 *
 *      Author: Marc Claesen
 */

/*************************************************************************************************/

#include "CLI.hpp"
#include "DataFile.hpp"
#include "io.hpp"
#include "Util.hpp"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <errno.h>
#include <deque>
#include <vector>
#include <algorithm>

/*************************************************************************************************/

using std::vector;
using std::deque;
using std::unique_ptr;
using std::string;
using namespace ensemble;

std::string toolname("bootstrap");

/*************************************************************************************************/

/**
 * Draws a bootstrap sample of elements in set and stores them in sample.
 *
 * sample.size() is used to determine size of bootstrap sample.
 */
template <typename T>
void bootstrap(const deque<T> &set, vector<T> &sample){
	for(unsigned i=0;i<sample.size();++i){
		sample[i]=set[(rand() % set.size())];
	}
}

/*************************************************************************************************/

int main(int argc, char **argv)
{
	std::cin.sync_with_stdio(false);

	// initialize help
	std::string helpheader(
			"Generates a bootstrap mask for the given data set as specified by the user.\n"
			"Output comprises one line per bootstrap sample, containing indices of sampled instances.\n"
			"Indexing is 1-based, meaning that the first data instance has index 1.\n"
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
	description="output file";
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

	description = "number of positives per bootstrap sample";
	keyword = "-npos";
	CLI::Argument<unsigned> npos(description,keyword,CLI::Argument<unsigned>::Content(1,0));
	allargs.push_back(&npos);

	description = "number of negatives per bootstrap sample";
	keyword = "-nneg";
	CLI::Argument<unsigned> nneg(description,keyword,CLI::Argument<unsigned>::Content(1,0));
	allargs.push_back(&nneg);

	description = "number of bootstrap samples to make (default: 1)";
	keyword = "-nboot";
	CLI::Argument<unsigned> nboot(description,keyword,CLI::Argument<unsigned>::Content(1,1));
	allargs.push_back(&nboot);

	keyword = "-xval";
	multilinedesc.push_back("file containing cross-validation mask (cfr. cross-validate tool)");
	multilinedesc.push_back("excludes the fold specified in -xvalfold from bootstrap");
	CLI::Argument<string> xval(multilinedesc,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&xval);
	multilinedesc.clear();

	multilinedesc.push_back("treats specified cross-validation fold as test fold (not sampled)");
	multilinedesc.push_back("requires cross-validation file to be specified (cfr -xval)");
	keyword = "-xvalfold";
	CLI::Argument<unsigned> xvalfold(multilinedesc,keyword,CLI::Argument<unsigned>::Content(1,1));
	allargs.push_back(&xvalfold);
	multilinedesc.clear();

	description = "column delimiter in data file (default: ' ')";
	keyword = "-delim";
	char d=*" ";
	CLI::Argument<char> delim(description,keyword,CLI::Argument<char>::Content(1,d));
	allargs.push_back(&delim);

	description = "append generated masks to the output file, rather than overwriting it";
	keyword = "-append";
	CLI::FlagArgument append(description,keyword,false);
	allargs.push_back(&append);

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
	if(!npos.configured()){
		std::cerr << "Number of positives unspecified (see -npos)." << std::endl;
		validargs=false;
	}
	if(!nneg.configured()){
		std::cerr << "Number of negatives unspecified (see -nneg)." << std::endl;
		validargs=false;
	}
	if(xval.configured() && !xvalfold.configured()){
		std::cerr << "Specified cross-validation mask but not index." << std::endl;
		validargs=false;
	}
	if(!xval.configured() && xvalfold.configured()){
		std::cerr << "Specified cross-validation index but not mask." << std::endl;
		validargs=false;
	}
	if(!validargs)
		exit_with_err("Invalid command line arguments provided.");

	// read data file to identify indices of positives and negatives
	std::deque<unsigned> pos, neg;
	std::ifstream datafile(data[0].c_str(),std::ios::in);
	readLabels(datafile,delim[0],labels[0],labels[1],pos,neg,posvall.value());
	datafile.close();

	if(verbose){
		std::cout << "Read " << pos.size() << " positives and " << neg.size() << " negatives from " << data[0] << "." << std::endl;
	}

	/*************************************************************************************************/

	// read cross-validation mask if specified
	if(xval){
		std::map<unsigned, std::deque<unsigned> > xvalmask;
		readCrossvalMask(xval[0],xvalmask);

		std::map<unsigned, std::deque<unsigned> >::const_iterator F=xvalmask.find(xvalfold[0]);
		if(F==xvalmask.end()){
			exit_with_err(std::string("Could not find specified cross-validation fold in mask (cfr. -xval, -xvalfold)."));
		}

		// delete elements in specified fold from pos and neg
		for(std::deque<unsigned>::const_iterator I=F->second.begin(),E=F->second.end();I!=E;++I){
			bool found=false;
			for(std::deque<unsigned>::iterator Ipos=pos.begin(),Epos=pos.end();Ipos!=Epos;++Ipos){
				if(*Ipos==*I){
					found=true;
					pos.erase(Ipos);
				}
			}
			if(!found){
				for(std::deque<unsigned>::iterator Ineg=neg.begin(),Eneg=neg.end();Ineg!=Eneg;++Ineg){
					if(*Ineg==*I){
						found=true;
						neg.erase(Ineg);
					}
				}
			}
		}
	}

	if(xval.configured() && verbose){
		std::cout << pos.size() << " positives and " << neg.size() << " negatives left after filtering cross-validation fold " << xvalfold[0] << "." << std::endl;
	}

	/*************************************************************************************************/

	// initialize RNG
	srand(time(nullptr));

	// draw bootstrap samples and output to ofile
	std::ios::openmode mode=std::ios::out;
	if(append)
		mode=std::ios::app;

	std::ofstream ofile(ofname[0].c_str(),mode);
	vector<unsigned> posbootstrap(npos[0],0), negbootstrap(nneg[0],0), sample(npos[0]+nneg[0],0);
	for(unsigned i=0;i<nboot[0];++i){
		bootstrap(pos,posbootstrap);
		bootstrap(neg,negbootstrap);
		std::sort(posbootstrap.begin(),posbootstrap.end());
		std::sort(negbootstrap.begin(),negbootstrap.end());
		std::merge(posbootstrap.begin(),posbootstrap.end(),negbootstrap.begin(),negbootstrap.end(),sample.begin());
		vector<unsigned>::const_iterator I=sample.begin(),E=sample.end();
		if(I==E)
			exit_with_err("Empty bootstrap sample!");
		ofile << *I;
		++I;
		for(;I!=E;++I){
			ofile << " " << *I;
		}
		ofile << std::endl;
	}
	ofile.close();

	return EXIT_SUCCESS;
}
