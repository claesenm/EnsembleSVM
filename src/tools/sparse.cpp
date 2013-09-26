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
 * sparse.cpp
 *
 *      Author: Marc Claesen
 */

#include "CLI.hpp"
#include "Util.hpp"
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <deque>
#include <vector>

using std::string;
using std::istringstream;
using namespace ensemble;

std::string toolname("sparse");

unsigned sparsify(std::istream &is, std::ostream &os, char delim, bool labeled){
	string line, chunk;
	istringstream ss, ss2;
	bool first;
	double value;
	unsigned idx, numlines=0;

	while(is.good()){
		getline(is,line);

		if(line.length()==0)
			break;

		++numlines;

		ss.clear();
		ss.str(line);
		if(labeled){
			chunk.clear();
			getline(ss,chunk,delim);
			os << chunk << " ";
		}

		first=true;
		idx=1;
		while(ss.good()){
			chunk.clear();
			getline(ss,chunk,delim);

			ss2.clear();
			ss2.str(chunk);
			ss2 >> value;
			if(value!=0.0){
				if(!first)
					os << " ";
				first=false;

				os << idx << ":" << value;
			}
			++idx;
		}

		os << std::endl;
	}
	return numlines;
}

int main(int argc, char **argv)
{
	// initialize help
	std::string helpheader(
			"Constructs a sparse, space-seperated representation of given data set.\n"
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
	description="data file";
	CLI::Argument<string> data(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&data);

	keyword="-o";
	description="output file";
	CLI::Argument<string> ofname(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&ofname);

	description = "column delimiter in data file (default: ' ')";
	keyword = "-delim";
	char d=*" ";
	CLI::Argument<char> delim(description,keyword,CLI::Argument<char>::Content(1,d));
	allargs.push_back(&delim);

	description = "data file contains labels (in first column)";
	keyword = "-labeled";
	CLI::FlagArgument labeled(description,keyword,false);
	allargs.push_back(&labeled);

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
	if(!validargs)
		exit_with_err("Invalid command line arguments provided.");


	std::ifstream ifile(data[0].c_str(),std::ios::in);
	std::ofstream ofile(ofname[0].c_str());
	unsigned numlines = sparsify(ifile,ofile,delim[0],labeled.value());
	ifile.close();
	ofile.close();

	if(verbose)
		std::cout << "Data file contained " << numlines << " instances." << std::endl;

	return EXIT_SUCCESS;
}
