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
 * dummy-generator.cpp
 *
 *      Author: Marc Claesen
 */

#include "CLI.hpp"
#include "Util.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <algorithm>

using std::string;
using std::stringstream;
using std::pair;

using namespace ensemble;

std::string toolname("dummy-generator");

int main(int argc, char **argv)
{
	// initialize help
	string helpheader(
			"Replace categorical variables with binary dummy variables for specified columns.\n"
			"\nArguments:\n"
	), helpfooter("");


	// intialize arguments
	deque<CLI::BaseArgument*> allargs;

	string keyword("-data");
	string description("data file");
	CLI::Argument<string> data(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&data);

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

	keyword = "-o";
	description = "output file (== data file with dummy variables in specified columns)";
	CLI::Argument<string> ofile(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&ofile);

	vector<string> multilinedesc;
	keyword = "-cols";
	multilinedesc.push_back("column indices to process (left most column = 1)");
	multilinedesc.push_back("<amount of columns to process=n> <idx 1> <idx 2> ... <idx n>");
	CLI::RandomLengthArgument<unsigned> cols(multilinedesc,keyword,CLI::Argument<unsigned>::Content(1,0));
	allargs.push_back(&cols);

	keyword = "-save";
	description = "saves mapping of categories to dummy variables for future reference (optional, see -load)";
	CLI::Argument<string> save(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&save);

	keyword = "-load";
	description = "reference file containing mapping of categories to dummy variables to use (optional, see -save)";
	CLI::Argument<string> load(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&load);

	keyword = "-delim";
	description = "column delimiter (default whitespace)";
	CLI::Argument<char> delim(description,keyword,CLI::Argument<char>::Content(1,*" "));
	allargs.push_back(&delim);

	keyword = "-offset";
	description = "column offset against reference file";
	CLI::Argument<int> offset(description,keyword,CLI::Argument<int>::Content(1,0));
	allargs.push_back(&offset);

	keyword = "-zero";
	description = "return 0 dummy-vector if category is not in reference";
	CLI::FlagArgument zero(description,keyword,false);
	allargs.push_back(&zero);

	keyword = "-v";
	description = "enables verbose mode, which outputs various information to stdout";
	CLI::FlagArgument verbose(description,keyword,false);
	allargs.push_back(&verbose);

	ParseCLI(argv,argc,1,allargs);

	if(help.configured() || help2.configured())
		exit_with_help(allargs,helpheader,helpfooter,true);
	if(version.configured() || version2.configured())
		exit_with_version(toolname);


	if(argc==1)
		exit_with_help(allargs,helpheader,helpfooter);

	// assert correct information has been set to continue
	if(data && ofile && cols){

		std::istringstream iss, liness;
		std::ifstream ifs;
		ifs.open(data[0].c_str(),std::ifstream::in);

		unsigned numrows=0, numreplacecols=cols.size(), colidx=0;

		// map to hold index per category
		typedef std::map<string,unsigned> IndexMap;
		vector<IndexMap> storage(numreplacecols);

		// store indices in vector and sort them
		vector<unsigned> colsvec(cols.size(),0);
		for(unsigned i=0;i<colsvec.size();++i)
			colsvec[i]=cols[i];
		std::sort(colsvec.begin(),colsvec.end());

		std::string line, chunk;
		vector<unsigned>::const_iterator Icols;
		const vector<unsigned>::const_iterator Ecols=colsvec.end(), Bcols=colsvec.begin();

		// first loop: scan data file to identify values per category
		// or load categories from reference file
		if(load){
			std::ifstream loadfs(load[0].c_str(),std::ifstream::in);
			unsigned thiscolidx, colsvecidx=0;
			string category, colidxstr;
			while(getline(loadfs,line)){
				liness.clear();
				liness.str(line);
				getline(liness,colidxstr,delim[0]);
				thiscolidx = strtoul(colidxstr.c_str(),nullptr,10);

				if(colsvec[colsvecidx]+offset[0]!=thiscolidx)
					exit_with_err("Reference mapping does not correspond with columns provided via command line.");

				while(liness.good()){
					getline(liness,category,delim[0]);
					storage[colsvecidx].insert(make_pair(category,storage[colsvecidx].size()+1));
				}

				++colsvecidx;
			}

			while(getline(ifs,line))
				++numrows;

		}else{
			while(getline(ifs,line)){
				numrows++;
				Icols=colsvec.begin();

				liness.clear();
				liness.str(line);

				colidx=1;
				while(liness.good() && Icols!=Ecols){

					chunk.clear();
					if(colidx!=*Icols){
						// not processing a relevant column
						getline(liness,chunk,delim[0]);
						++colidx;
					}else{
						// processing a relevant column
						getline(liness,chunk,delim[0]);
						unsigned index=storage[Icols-Bcols].size()+1;
//						pair<map<string,unsigned>::iterator, bool > insertion=storage[Icols-Bcols].insert(make_pair(chunk,index));
						storage[Icols-Bcols].insert(make_pair(chunk,index));
						++Icols;
						++colidx;
					}
				}
			}
		}

		if(verbose){
			std::cout << "Data file contains " << numrows << " rows." << std::endl;

			for(unsigned i=0;i<colsvec.size();++i){
				std::cout << storage[i].size() << " categories in column " << colsvec[i] << ":";
				for(IndexMap::const_iterator Imap=storage[i].begin(),Emap=storage[i].end();Imap!=Emap;++Imap)
					std::cout << " \"" << Imap->first << "\"";
				std::cout << std::endl;
			}
		}

		if(save){
			// saving necessitates inverting the IndexMap to preserve order of categories when loading (IndexMap sorted on str)
			std::map<unsigned,string> inverted;

			std::ofstream refos(save[0].c_str());
			for(unsigned i=0;i<colsvec.size();++i){
				inverted.clear();
				refos << colsvec[i];
				for(IndexMap::const_iterator Imap=storage[i].begin(),Emap=storage[i].end();Imap!=Emap;++Imap)
					inverted.insert(make_pair(Imap->second,Imap->first));

				for(std::map<unsigned,string>::const_iterator Iinv=inverted.begin(),Einv=inverted.end();Iinv!=Einv;++Iinv)
					refos << delim[0] << Iinv->second;

				if(i<colsvec.size()-1)
					refos << std::endl;
			}
			refos.close();
		}

		// open output file
		std::ofstream ofstream(ofile[0].c_str());

		// rewind data file for second parse
		ifs.clear();
		ifs.seekg(0, std::ios::beg);

		// second loop: pipe unaltered columns to output and generate dummies where necessary
		unsigned rownum=0;
		while(getline(ifs,line)){
			if(rownum>0)
				ofstream << std::endl;

			liness.clear();
			liness.str(line);

			Icols=colsvec.begin();

			colidx=1;

			while(liness.good() && Icols!=Ecols){

				chunk.clear();
				if(colidx != *Icols){

					// not processing a relevant column
					getline(liness,chunk,delim[0]);
					if(colidx>1)
						ofstream << delim[0];
					ofstream << chunk;
					++colidx;

				}else{

					// processing a relevant column
					getline(liness,chunk,delim[0]);
					IndexMap::const_iterator F=storage[Icols-Bcols].find(chunk);

					unsigned dummyidx=0;

					if(F==storage[Icols-Bcols].end()){
						if(!zero.value()){
							std::cerr << "Chunk: " << chunk << std::endl;
							exit_with_err("Illegal state: category not found in dummy mapping!");
						}
					}else{
						dummyidx=F->second;
					}

					// binarize
					unsigned numdummies=storage[Icols-Bcols].size();

					for(unsigned dummyiter=1;dummyiter<=numdummies;++dummyiter){
						if(colidx>1 || dummyiter>1)
							ofstream << delim[0];
						if(dummyiter==dummyidx)
							ofstream << "1";
						else
							ofstream << "0";
					}

					++Icols;
					++colidx;
				}
			}

			if(liness.good()){
				std::string remainder;
				liness >> remainder;
				ofstream << "," << remainder;
			}

			++rownum;
		}

		ifs.close();
		ofstream.close();

	}else{
		if(argc!=1){
			exit_with_err("Missing/illegal options specified!");
		}
	}

	exit(EXIT_SUCCESS);
}
