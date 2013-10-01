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
 * esvm-train.cpp
 *
 *      Author: Marc Claesen
 */

/*************************************************************************************************/

#include "config.h"
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
#include "DataFile.hpp"
#include "BinaryWorkflow.hpp"
#include "Executable.hpp"
#include <errno.h>
#include <functional>
#include <cmath>


#ifdef HAVE_PTHREAD
#include "ThreadPool.hpp"
#endif

/*************************************************************************************************/

using std::unique_ptr;
using std::string;
using namespace ensemble;

std::string toolname("esvm-train");

/*************************************************************************************************/

/**
 * Ensures correct concurrent access to the SVMEnsemble it handles.
 * Uses a mutex to prevent multiple threads from adding models simultaneously.
 */
class Manager{
private:
	std::unique_ptr<SVMEnsemble> ensemble;
#ifdef HAVE_PTHREAD
	std::mutex m;
#endif

public:
	Manager(std::unique_ptr<Kernel> kernel, const SVMEnsemble::LabelMap& map):ensemble(new SVMEnsemble(std::move(kernel),map)){};
	Manager(const Manager& o) = delete;
	Manager(Manager&& o) = delete;
	Manager &operator=(const Manager& o) = delete;
	Manager &operator=(Manager&& o) = delete;

	void add(std::unique_ptr<SVMModel> model){
#ifdef HAVE_PTHREAD
		std::unique_lock<std::mutex> lock{m};
#endif
		ensemble->add(std::move(model));
	}

	const SVMEnsemble* get() const{ return ensemble.get(); }
	const Kernel* getKernel() const{ return ensemble->getKernel(); }

	/**
	 * Transfers ownership of the ensemble.
	 */
	std::unique_ptr<SVMEnsemble> transfer(){ return std::unique_ptr<SVMEnsemble>(ensemble.release()); }
};

/*************************************************************************************************/

bool readBootstrapLine(std::istream &stream, std::list<unsigned> &mask, char delim){
	if(!stream.good()){
		exit_with_err("Error reading bootstrap line.");
	}

	std::string line, token;
	unsigned idx;

	bool read=false;

	getline(stream,line);
	std::istringstream liness(line), tokenss;
	while(liness.good()){
		token.clear();
		tokenss.clear();
		getline(liness,token,delim);
		tokenss.str(token);
		tokenss >> idx;
		mask.push_back(idx);
		read=true;
	}
	return read;
}

void parallel_train_ptrs(svm_problem *problem, svm_parameter *par, Manager& mgr){
	std::unique_ptr<SVMModel> model(LibSVM::libsvm_train(std::make_pair(
			std::unique_ptr<svm_problem>(problem),std::unique_ptr<svm_parameter>(par))
	));
	mgr.add(std::move(model));
}

/*************************************************************************************************/

int main(int argc, char **argv)
{
	std::cin.sync_with_stdio(false);

	std::string helpheader(
			"Constructs an ensemble of weighted SVMs with given configuration.\n"
			"\nArguments:\n"
	), helpfooter("");


	// intialize arguments
	deque<CLI::BaseArgument*> allargs;

	string description("training data file");
	string keyword("-data");
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

	description = "data file in csv format (default: sparse space separated)";
	keyword = "-csv";
	CLI::FlagArgument csv(description,keyword,false);
	allargs.push_back(&csv);

	description = "data file in sparse csv format (default: sparse space separated)";
	keyword = "-sparsecsv";
	CLI::FlagArgument sparsecsv(description,keyword,false);
	allargs.push_back(&sparsecsv);

	description = "output file for ESVM model (default 'a.out')";
	keyword = "-o";
	CLI::Argument<string> ofile(description,keyword,CLI::Argument<string>::Content(1,"a.out"));
	allargs.push_back(&ofile);

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

	keyword = "-nmodels";
	description = "amount of individual SVM models in ensemble (default '1')";
	CLI::Argument<unsigned> nmodels(description,keyword,CLI::Argument<unsigned>::Content(1,1));
	allargs.push_back(&nmodels);

	keyword = "-bootstrap";
	multilinedesc.push_back("file containing bootstrap samples per model (see bootstrap tool)");
	multilinedesc.push_back("if unspecified, all training instances are used in each model");
	CLI::Argument<string> bootstrap(multilinedesc,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&bootstrap);
	multilinedesc.clear();

	keyword = "-pospen";
	description="misclassification penalty coefficient for positive class";
	CLI::Argument<double> pospen(description,keyword,CLI::Argument<double>::Content(1.0,1));
	allargs.push_back(&pospen);
	multilinedesc.clear();

	keyword = "-negpen";
	description="misclassification penalty coefficient for negative class";
	CLI::Argument<double> negpen(description,keyword,CLI::Argument<double>::Content(1.0,1));
	allargs.push_back(&negpen);
	multilinedesc.clear();

	keyword = "-penalties";
	multilinedesc.push_back("space-seperated file with sparse penalties per instance");
	multilinedesc.push_back("each line represents weights for instances used to train a single model");
	multilinedesc.push_back("line format: <idx>:<penalty> pairs, space-seperated; only sampled instances are listed");
	multilinedesc.push_back("overrides -pospen, -negpen and -bootstrap");
	CLI::Argument<string> penfile(multilinedesc,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&penfile);
	multilinedesc.clear();

	// kernel arguments
	keyword = "-kfun";
	multilinedesc.push_back("set type of kernel function (default 2)");
	multilinedesc.push_back("0 -- linear: u'*v");
	multilinedesc.push_back("1 -- polynomial: (gamma*u'*v + coef0)^degree");
	multilinedesc.push_back("2 -- radial basis function: exp(-gamma*|u-v|^2)");
	multilinedesc.push_back("3 -- sigmoid: tanh(gamma*u'*v + coef0)");
	CLI::Argument<unsigned> kfun(multilinedesc,keyword,CLI::Argument<unsigned>::Content(1,2));
	allargs.push_back(&kfun);
	multilinedesc.clear();

	description = "set degree in kernel function (default 3)";
	keyword = "-degree";
	CLI::Argument<unsigned> degree(description,keyword,CLI::Argument<unsigned>::Content(1,3));
	allargs.push_back(&degree);

	description = "set gamma in kernel function (default 1.0)";
	keyword = "-gamma";
	CLI::Argument<double> gamma(description,keyword,CLI::Argument<double>::Content(1,1.0));
	allargs.push_back(&gamma);

	description = "set coef0 in kernel function (default 0.0)";
	keyword = "-coef0";
	CLI::Argument<double> coef0(description,keyword,CLI::Argument<double>::Content(1,0.0));
	allargs.push_back(&coef0);

	description = "configure cache size (in MB) used by LIBSVM (default 100.0)";
	keyword = "-cache";
	CLI::Argument<double> cachesize(description,keyword,CLI::Argument<double>::Content(1,100.0));
	allargs.push_back(&cachesize);

#ifdef HAVE_PTHREAD
	description = "set number of threads (default: number of hardware threads)";
	keyword = "-threads";
	CLI::Argument<unsigned> threads(description,keyword,CLI::Argument<unsigned>::Content(1,0));
	allargs.push_back(&threads);
#endif

	description = "use logistic regression for aggregation (default: majority voting)";
	keyword = "-logistic";
	CLI::FlagArgument logistic(description,keyword,false);
	allargs.push_back(&logistic);

	description = "enables verbose mode, which outputs various information to stdout";
	keyword = "-v";
	CLI::FlagArgument verbose(description,keyword,false);
	allargs.push_back(&verbose);

	ParseCLI(argv,argc,1,allargs);

	if(help.configured() || help2.configured())
		exit_with_help(allargs,helpheader,helpfooter,true);
	if(version.configured() || version2.configured())
		exit_with_version(toolname);

	if(argc==1)
		exit_with_help(allargs,helpheader,helpfooter);

	// verify command line arguments
	bool err=false;
	if(!data.configured()){
		std::cerr << "Data file not configured (see -data).";
		err=true;
	}
	if(!ofile.configured()){
		std::cerr << "Output file not configured (see -o).";
		err=true;
	}
	if(!(penfile || (pospen && negpen))){
		std::cerr << "Penalties not specified (see -pospen, -negpen, -penalties).";
		err=true;
	}
#ifdef HAVE_PTHREAD
	if(threads.configured() && threads[0]==0){
		std::cerr << "Number of threads must be > 0.";
		err=true;
	}
#endif
	if(err)
		exit_with_err("Invalid configuration specified via command line.");

	/*************************************************************************************************/

	// read and construct kernel and construct ensemble
	unique_ptr<Kernel> kernel=KernelFactory(kfun[0],degree[0],gamma[0],coef0[0]);

	// construct internal label translation map
	SVMEnsemble::LabelMap map;
	map.insert(make_pair("1",labels[0]));
	map.insert(make_pair("-1",labels[1]));
	Manager mgr(std::move(kernel),std::move(map));

	int format=FileFormats::DEFAULT;
	if(csv)
		format=FileFormats::CSV;
	else if(sparsecsv)
		format=FileFormats::SparseCSV;

	// index training data
	unique_ptr<IndexedFile> traindata(new IndexedFile(data[0]));

	// initialize files
	std::ifstream weightfile, bootfile;
	if(penfile) weightfile.open(penfile[0].c_str(),std::ios::in);
	if(bootstrap) bootfile.open(bootstrap[0].c_str(),std::ios::in);

	std::deque<SparseVector*> storage;
	vector<const SparseVector*> bsdata;
	vector<bool> bslabels;
	vector<double> bspenalties;

	// initialize bootstrap to contain each training instance if no file specified
	std::list<unsigned> bootstrapidx;
	if(!bootstrap.configured() && !penfile.configured()){
		for(unsigned i=1;i<=traindata->size();++i)
			bootstrapidx.push_back(i);
	}

	/*************************************************************************************************/


#ifdef HAVE_PTHREAD
	unsigned numthreads=threads.configured() ? threads[0] : NUM_HARDWARE_THREADS;
	numthreads=numthreads ? numthreads : 1;

	std::function<void(svm_problem*,svm_parameter*)> fun=std::bind(parallel_train_ptrs,std::placeholders::_1,std::placeholders::_2,std::ref(mgr));
	ThreadPool<void(svm_problem*,svm_parameter*)> threadmanager{std::move(fun),numthreads,numthreads}; // use maxjobs=numthreads to ensure no waiting

#endif

	/*************************************************************************************************/

	// main loop: build models with correct parameters and add to ensemble;
	for(unsigned nummodel=0;nummodel<nmodels[0];++nummodel){

		// read bootstrap and weights from file
		if(penfile){
			if(!weightfile.good())
				exit_with_err(std::string("Unable to read line from penalty file (-penalties)"));

			std::string penline;
			getline(weightfile,penline);
			std::istringstream liness(penline);
			unique_ptr<SparseVector> weights=SparseVector::read(liness);

			bootstrapidx.clear();
			if(bspenalties.size() < weights->numNonzero())
				bspenalties.resize(weights->numNonzero());

			unsigned widx=0;
			for(SparseVector::const_iterator Iw=weights->begin(),Ew=weights->end();Iw!=Ew;++Iw,++widx){
				bootstrapidx.push_back(Iw->first);
				bspenalties[widx]=Iw->second;
			}
		}else if(bootstrap){
			// read line from bootstrap file if configured
			bootstrapidx.clear();
			if(!readBootstrapLine(bootfile,bootstrapidx,*" "))
				exit_with_err("Error reading bootstrap file.");
		}

		// resize scratch spaces if necessary
		if(bootstrapidx.size() > bsdata.size()){
			bsdata.resize(bootstrapidx.size(),nullptr);
			bslabels.resize(bootstrapidx.size());

			if(!penfile.configured()) // already done if we use instance weighting
				bspenalties.resize(bootstrapidx.size());
		}

		unsigned instanceidx=0; // 1-based, first data instance has index 1
		for(std::list<unsigned>::const_iterator I=bootstrapidx.begin(),E=bootstrapidx.end();I!=E;++I,++instanceidx){

			std::string line((*traindata)[*I]);
			unique_ptr<DataLine> dataline=LabeledDataFile::readline(line,format);
			unique_ptr<SparseVector> inst=dataline->getSV();
			bsdata[instanceidx]=inst.get();
			storage.push_back(inst.get());
			inst.release();

			// if no individual penalties were configured, use 1 as instance penalty
			if(!penfile.configured())
				bspenalties[instanceidx]=1;

			// check if this belongs to the positive class
			if(labels[0].compare(*dataline->rawLabel())==0) bslabels[instanceidx]=true;
			else if(posvall.value())
				bslabels[instanceidx]=false;
			else if(labels[1].compare(*dataline->rawLabel())==0) bslabels[instanceidx]=false;
			else exit_with_err(std::string("Encountered unknown label on line: ") + *dataline->rawLabel());
		}

		/**
		 * Ensure first data instance is positive.
		 */
		//  find index of first positive data instance
		unsigned posidx=0;
		for(;posidx<bootstrapidx.size();++posidx){
			if(bslabels[posidx])
				break;
		}
		if(posidx!=0){
			// move labels
			bslabels[0]=true;
			bslabels[posidx]=false;

			// move data
			const SparseVector *tmpvector=bsdata[0];
			bsdata[0]=bsdata[posidx];
			bsdata[posidx]=tmpvector;

			// move penalties
			double tmppenalty=bspenalties[0];
			bspenalties[0]=bspenalties[posidx];
			bspenalties[posidx]=tmppenalty;
		}

		unique_ptr<SVMModel> model;
		LibSVM::full_svm_problem problem;

#ifdef HAVE_PTHREAD
		double libsvmcache=cachesize[0]/numthreads;
#else
		double libsvmcache=cachesize[0];
#endif

		if(penfile.configured()){

			problem=LibSVM::construct_BSVM_problem(mgr.getKernel(), 1, 1,
					libsvmcache, bsdata, bslabels,	bspenalties, bootstrapidx.size());

		}else{

			problem=LibSVM::construct_BSVM_problem(mgr.getKernel(), pospen[0],negpen[0],
					libsvmcache, bsdata, bslabels,	bspenalties, bootstrapidx.size());
		}


#ifdef HAVE_PTHREAD

		threadmanager.addjob(problem.first.get(),problem.second.get());
		problem.first.release();
		problem.second.release();

#else

		model=LibSVM::libsvm_train(std::move(problem));
		mgr.add(std::move(model));

#endif

		for(std::deque<SparseVector*>::iterator I=storage.begin(),E=storage.end();I!=E;++I)
			delete *I;
		storage.clear();
	}

#ifdef HAVE_PTHREAD

	threadmanager.wait();
	threadmanager.join();

#endif

	if(verbose){
		std::cout << "num_distinct_sv " << mgr.get()->numDistinctSV() << " total_sv " << mgr.get()->numTotalSV() << std::endl;
	}

	// wrap the ensemble into a Workflow so we can adjust it later
	std::unique_ptr<BinaryModel> model(mgr.transfer().release());
	std::unique_ptr<BinaryWorkflow> flow = defaultBinaryWorkflow(std::move(model),!logistic.value());

	if(penfile) weightfile.close();
	if(bootstrap) bootfile.close();

	// write ensemble to outputfile
	std::ofstream ofstream(ofile[0].c_str());
//	ofstream << *mgr.get();
	ofstream << *flow;
	ofstream.close();

	exit(EXIT_SUCCESS);
}
