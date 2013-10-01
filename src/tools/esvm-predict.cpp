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
 * esvm-predict.cpp
 *
 *      Author: Marc Claesen
 */

/*************************************************************************************************/

#include "CLI.hpp"
#include "io.hpp"
#include "Util.hpp"
#include "Models.hpp"
#include "Ensemble.hpp"
#include "DataFile.hpp"
#include "ThreadPool.hpp"
#include "Executable.hpp"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <errno.h>
#include <deque>
#include <vector>
#include <functional>
#include <tuple>
#include <algorithm>

/*************************************************************************************************/

using std::vector;
using std::unique_ptr;
using std::string;
using namespace ensemble;

std::string toolname("esvm-predict");

/*************************************************************************************************/

/**
 * Obtains the base model correct rate based on the given prediction and the true label.
 */
double baseScore(const Prediction& pred, bool truth){
	Prediction::const_iterator I=pred.begin(),E=pred.end();
	++I;
	unsigned size = std::distance(I,E);
	unsigned numpos = std::count_if(I,E,[](double x){ return x > 0.0; });

	double score = (1.0*numpos)/size;
	if(!truth) score=1.0-score;
	return score;
}

/*************************************************************************************************/

#ifdef HAVE_PTHREAD

std::tuple<Prediction,bool,double> predict(const std::string& poslabel, const Model& model, std::shared_ptr<ConstDataLine> line){
	Prediction pred=model.predict(*line->rawSV());
	double baseacc;

	bool correct=true;
	if(line->labeled()){
		// number of positive predictions by base models
		baseacc=baseScore(pred,true);
		if(line->rawLabel()->compare(poslabel)==0){ // positive label
			if(poslabel.compare(pred.getLabel())!=0){
				correct=false;
			}
		}else{
			baseacc=1-baseacc;
			if(poslabel.compare(pred.getLabel())==0){
				correct=false;
			}
		}
	}
	return std::make_tuple(std::move(pred),correct,baseacc);
}

#endif

/*************************************************************************************************/

int main(int argc, char **argv)
{
	std::cin.sync_with_stdio(false);

	// initialize help
	std::string helpheader(
			"Performs predictions for test instances in given data file, using the model specified by -model.\n"
			"In the output file, each line contains the predicted label and decision values."
			"\n\n"
			"Options:\n"
	), helpfooter("");

	// intialize arguments
	std::deque<CLI::BaseArgument*> allargs;
	string description, keyword;
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

	keyword="-data";
	description="test data file";
	CLI::Argument<string> datafname(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&datafname);

	keyword="-model";
	description="model file (LIBSVM model/generic SVM model/ensemble model)";
	CLI::Argument<string> modelfname(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&modelfname);

	keyword="-o";
	description="output file";
	CLI::Argument<string> ofname(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&ofname);

	keyword = "-xval";
	multilinedesc.push_back("file containing cross-validation mask (cfr. cross-validate tool)");
	multilinedesc.push_back("predicts instances in fold -xvalfold, requires labels (-labeled)");
	CLI::Argument<string> xval(multilinedesc,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&xval);
	multilinedesc.clear();

	multilinedesc.push_back("treats specified cross-validation fold as test fold");
	multilinedesc.push_back("requires cross-validation file to be specified (cfr -xval)");
	keyword = "-xvalfold";
	CLI::Argument<unsigned> xvalfold(multilinedesc,keyword,CLI::Argument<unsigned>::Content(1,1));
	allargs.push_back(&xvalfold);
	multilinedesc.clear();

	description = "data file in csv format (default: space separated)";
	keyword = "-csv";
	CLI::FlagArgument csv(description,keyword,false);
	allargs.push_back(&csv);

	description = "data file in sparse csv format (default: space separated)";
	keyword = "-sparsecsv";
	CLI::FlagArgument sparsecsv(description,keyword,false);
	allargs.push_back(&sparsecsv);

	description = "data file contains labels for performance assessment (default: off)";
	keyword = "-labeled";
	CLI::FlagArgument labeled(description,keyword,false);
	allargs.push_back(&labeled);

	description = "include base model decision values in output (columns 3:end)";
	keyword = "-base";
	CLI::FlagArgument base(description,keyword,false);
	allargs.push_back(&base);

	ParseCLI(argv,argc,1,allargs);

	if(help.configured() || help2.configured())
		exit_with_help(allargs,helpheader,helpfooter,true);
	if(version.configured() || version2.configured())
		exit_with_version(toolname);

	if(!(datafname && ofname && modelfname))
		exit_with_help(allargs,helpheader,helpfooter);

	bool validargs=true;
	if(xval.configured() && !xvalfold.configured()){
		std::cerr << "Specified cross-validation mask but not index." << std::endl;
		validargs=false;
	}
	if(xval.configured() && !labeled.value()){
		std::cerr << "Specified cross-validation test fold but using unlabeled data file." << std::endl;
		validargs=false;
	}
	if(!xval.configured() && xvalfold.configured()){
		std::cerr << "Specified cross-validation index but not mask." << std::endl;
		validargs=false;
	}
	if(!validargs)
		exit_with_err("Invalid command line arguments provided.");

	/*************************************************************************************************/

	// get correct list of indices if bootstrap mask is specified
	const std::deque<unsigned> *indices(nullptr);
	std::map<unsigned, std::deque<unsigned> > xvalmask;
	if(xval.configured()){
		readCrossvalMask(xval[0],xvalmask);

		std::map<unsigned, std::deque<unsigned> >::const_iterator F=xvalmask.find(xvalfold[0]);
		if(F==xvalmask.end()){
			exit_with_err(std::string("Could not find specified cross-validation fold in mask (cfr. -xval, -xvalfold)."));
		}

		indices=&(F->second);
	}

	unique_ptr<BinaryModel> model=BinaryModel::load(modelfname[0].c_str());
	unique_ptr<DataFile> data(nullptr);

	if(labeled){
		unique_ptr<LabeledDataFile> labdata(nullptr);
		if(csv)
			labdata=LabeledDataFile::readf(datafname[0], FileFormats::CSV, indices);
		else if(sparsecsv)
			labdata=LabeledDataFile::readf(datafname[0], FileFormats::SparseCSV, indices);
		else
			labdata=LabeledDataFile::readf(datafname[0], FileFormats::DEFAULT, indices);
		data=unique_ptr<DataFile>(dynamic_cast<DataFile*>(labdata.release()));
	}else{
		if(csv)
			data=DataFile::readf(datafname[0], FileFormats::CSV);
		else if(sparsecsv)
			data=DataFile::readf(datafname[0], FileFormats::SparseCSV);
		else
			data=DataFile::readf(datafname[0], FileFormats::DEFAULT);
	}

	string poslabel=model->positive_label();

	/*************************************************************************************************/

#ifdef HAVE_PTHREAD
	std::function<std::tuple<Prediction,bool,double>(std::shared_ptr<ConstDataLine>)> fun =
			std::bind(predict,std::cref(poslabel),std::cref(*model.get()),std::placeholders::_1);

	ThreadPool<std::tuple<Prediction,bool,double>(std::shared_ptr<ConstDataLine>)> manager(std::move(fun));
#endif

	/*************************************************************************************************/

	/**
	  Main loop
	 */

	std::ofstream outfile(ofname[0].c_str());
	unsigned numinstances=0, numcorrect=0;
	double baseacc=0.0;
	for(unsigned instanceidx=0;instanceidx<data->size();++instanceidx){
		auto dataline = data->getdataline(instanceidx);

		/*************************************************************************************************/

#ifdef HAVE_PTHREAD

		/*************************************************************************************************/
		manager.addjob(dataline);
	}
	for(auto& job: manager){
		std::tuple<Prediction,bool,double> result=job.get();
		Prediction pred=std::get<0>(result);
		if(std::get<1>(result)) numcorrect++;
		baseacc+=std::get<2>(result);

		/*************************************************************************************************/

#else

		/*************************************************************************************************/

		Prediction pred=model->predict(*dataline->rawSV());
		double basepos=0.0;
		if(labeled){
			basepos = baseScore(pred,true);
			const std::string& label=*dataline->rawLabel();
			if(pred.getLabel().compare(poslabel)==0){ // positive prediction
				if(label.compare(poslabel)==0){
					// true positive
					++numcorrect;
						baseacc+=basepos;
				}else{
					baseacc+=1-basepos;
				}
			}else{ // negative prediction
				if(label.compare(poslabel)==0){
					// false negative
						baseacc+=basepos;

				}else{
					// true negative
					++numcorrect;
						baseacc+=1-basepos;
				}
			}
		}

		/*************************************************************************************************/

#endif

		/*************************************************************************************************/
		numinstances++;
		if(base.value())
			outfile << pred << std::endl;
		else
			outfile << pred.getLabel() << " " << pred[0] << std::endl;
	}

	if(labeled){
		double acc=(1.0*numcorrect)/numinstances;
		std::cout << "Accuracy: " << acc << " base model accuracy: " << baseacc/numinstances << std::endl;
	}

	return 0;
}
