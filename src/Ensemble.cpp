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
 * Ensemble.cpp
 *
 *      Author: Marc Claesen
 */

#define ENSEMBLE_CPP

#include "Ensemble.hpp"
#include "Util.hpp"
#include "io.hpp"
#include "config.h"
#include "PredicatedFactory.hpp"
#include <set>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <memory>

using std::unique_ptr;


namespace{

std::streamsize PRECISION = 16;

} // anonymous namespace

namespace ensemble{

class SVMEnsembleImpl:public Ensemble{
private:
	SVMEnsembleImpl &operator=(const SVMEnsembleImpl &e);

public:
	typedef std::deque<std::shared_ptr<SparseVector>> SVDeque;
	typedef SVDeque::iterator sv_iterator;
	typedef SVDeque::const_iterator sv_const_iterator;
	typedef std::map<const SparseVector*,int,SVSort> SVMap;
	typedef std::map<SVMModel*,int> modelCont;
	typedef modelCont::iterator iterator;
	typedef modelCont::const_iterator const_iterator;
	typedef std::map< std::string, std::string > LabelMap;

protected:
	// kernel, either parameterized or a matrix (UserdefKernel)
	unique_ptr<Kernel> kernel;

	// deque containing actual distinct SVs
	SVDeque svJumpTable;

	// we use a set to exploit fast search when adding additional SVs.
	SVMap supportVectors;

	// maps indices in the ensemble to svJumpTable
	std::deque<unsigned> SVindex;

	// map containing model and initial index of this model in SVindex
	modelCont models;

	// used during predictions
	mutable std::vector<std::vector<double>> denseSVs;

	// used to change internal model labels to labels as found in data file
	LabelMap labelmap; // fixme: remove

	std::vector<double> predict_by_cache(const std::vector<double>& cache) const;

	/**
	 * Returns the Prediction based on the decision values, if majority voting is used.
	 */
	Prediction decval2prediction(std::vector<double>&& decision_vals) const;

public:
	/**
	 * Constructs an SVMEnsembleImpl using <kernel>. The newly created object acquires ownership of <kernel>.
	 */
	SVMEnsembleImpl(unique_ptr<Kernel> kernel);
	SVMEnsembleImpl(unique_ptr<Kernel> kernel, const LabelMap &labelmap);
	SVMEnsembleImpl(const SVMEnsemble* ens, std::vector<unique_ptr<SVMModel>>&& models);
	SVMEnsembleImpl(const SVMEnsemble* ens, std::vector<unique_ptr<SVMModel>>&& models,
			const std::string& positive, const std::string& negative);

	/**
	 * Ensemble prediction: returns the predicted label and scores.
	 *
	 * First score is the ensemble consensus for the given label.
	 * Subsequent scores are decision values per base model.
	 */
	virtual Prediction predict(const SparseVector &i) const override;

	/**
	 * Dense prediction.
	 */
	virtual Prediction predict(const std::vector<double> &v) const override;

	/**
	 * Returns the base model decision values for prediction of the test instance.
	 */
	virtual std::vector<double> decision_value(const SparseVector &i) const override final;
	virtual std::vector<double> decision_value(const std::vector<double> &i) const override final;

	// Adds SVM model *m to the SVMEnsembleImpl.
	virtual void add(std::unique_ptr<SVMModel> m, const SVMEnsemble* ens);

	unsigned getSVindex(unsigned ensembleidx) const;
	unsigned getSVindex(unsigned localidx, const SVMModel * const mod) const;

	/**
	 * Returns the SV at <ensidx> within the ensemble.
	 */
	std::shared_ptr<SparseVector> getSV(unsigned ensidx);
	const SparseVector* getSV(unsigned ensidx) const;

	iterator begin();
	iterator end();
	const_iterator begin() const;
	const_iterator end() const;

	sv_iterator sv_begin();
	sv_iterator sv_end();
	sv_const_iterator sv_begin() const;
	sv_const_iterator sv_end() const;

	size_t size() const;
	size_t numDistinctSV() const;
	size_t numTotalSV() const;

	virtual ~SVMEnsembleImpl();

	const Kernel *getKernel() const;

	virtual void printSV(std::ostream &os, int SVidx) const;

	virtual void serialize(std::ostream& os) const override;
	static unique_ptr<SVMEnsemble> read(std::istream &iss);
//	static unique_ptr<ensemble::SVMEnsemble> load(const string &fname);

	double density() const;

	/**
	 * Translate specified base-model label to an output label.
	 */
	std::string translate(const std::string &label) const;

	virtual std::string positive_label() const override;
	virtual std::string negative_label() const override;
	virtual size_t num_outputs() const override;

};


SVMEnsembleImpl::iterator SVMEnsembleImpl::begin(){ return models.begin(); }
SVMEnsembleImpl::iterator SVMEnsembleImpl::end(){ return models.end(); }
SVMEnsembleImpl::const_iterator SVMEnsembleImpl::begin() const{ return models.begin(); }
SVMEnsembleImpl::const_iterator SVMEnsembleImpl::end() const{ return models.end(); }

SVMEnsembleImpl::sv_iterator SVMEnsembleImpl::sv_begin(){ return svJumpTable.begin(); }
SVMEnsembleImpl::sv_iterator SVMEnsembleImpl::sv_end(){ return svJumpTable.end(); }
SVMEnsembleImpl::sv_const_iterator SVMEnsembleImpl::sv_begin() const{ return svJumpTable.begin(); }
SVMEnsembleImpl::sv_const_iterator SVMEnsembleImpl::sv_end() const{ return svJumpTable.end(); }

const Kernel *SVMEnsembleImpl::getKernel() const{ return kernel.get(); }

Ensemble::Ensemble(const Ensemble &e):BinaryModel(e){}
Ensemble::Ensemble():BinaryModel(){}

SVMEnsembleImpl::SVMEnsembleImpl(unique_ptr<Kernel> kernel):Ensemble(),kernel(std::move(kernel)){}
SVMEnsembleImpl::SVMEnsembleImpl(unique_ptr<Kernel> kernel, const LabelMap &labelmap):Ensemble(),kernel(std::move(kernel)),labelmap(labelmap){}

SVMEnsembleImpl::SVMEnsembleImpl(const SVMEnsemble* ens, std::vector<unique_ptr<SVMModel>>&& models, const std::string& positive, const std::string& negative)
:Ensemble(),kernel(models[0]->kernel){
	std::string pos=models[0]->positive_label();
	std::string neg=models[0]->negative_label();
	labelmap.insert(std::make_pair(pos,positive));
	labelmap.insert(std::make_pair(neg,negative));
	for(auto& model: models){
		assert(model->positive_label().compare(pos)==0 && "Internal model labels do not match!");
		assert(model->negative_label().compare(neg)==0 && "Internal model labels do not match!");
		add(std::move(model),ens);
	}
}

SVMEnsembleImpl::SVMEnsembleImpl(const SVMEnsemble* ens, std::vector<unique_ptr<SVMModel>>&& models)
:Ensemble(),kernel(models[0]->kernel){
	std::string pos=models[0]->positive_label();
	std::string neg=models[0]->negative_label();
	labelmap.insert(std::make_pair(pos,pos));
	labelmap.insert(std::make_pair(neg,neg));
	for(auto& model: models){
		assert(model->positive_label().compare(pos)==0 && "Internal model labels do not match!");
		assert(model->negative_label().compare(neg)==0 && "Internal model labels do not match!");
		add(std::move(model),ens);
	}
}


size_t SVMEnsembleImpl::size() const{ return models.size();}
size_t SVMEnsembleImpl::numDistinctSV() const{ return svJumpTable.size(); }
size_t SVMEnsembleImpl::numTotalSV() const{
	int sum=0;
	for(const_iterator I=begin(),E=end();I!=E;++I)
		sum+=I->first->size();
	return sum;
}

std::shared_ptr<SparseVector> SVMEnsembleImpl::getSV(unsigned ensidx){
	return svJumpTable[ensidx];
}

const SparseVector* SVMEnsembleImpl::getSV(unsigned ensidx) const{
	return svJumpTable[ensidx].get();
}

void SVMEnsembleImpl::add(std::unique_ptr<SVMModel> m, const SVMEnsemble* ens){
	int startidx = SVindex.size();
	SVMModel *newmodel = m.get();

	models.insert(std::make_pair(newmodel,startidx));

	if(labelmap.empty()){
		std::string pos=m->positive_label();
		std::string neg=m->negative_label();
		labelmap.insert(std::make_pair(pos,pos));
		labelmap.insert(std::make_pair(neg,neg));
	}

	// configure the SVMModel to know that it is contained ens
	if(newmodel->ens != nullptr && newmodel->ens != ens)
		exit_with_err("Attempting to add model to multiple ensembles!");

	newmodel->ens = ens;

	// make sure kernels are identical, abort if not
	if(*newmodel->kernel!=*kernel.get()){
		std::cout << *newmodel->kernel << std::endl;
		std::cout << *kernel << std::endl;
		exit_with_err("Attempting to add model with different kernel to ensemble!");
	}

	if(newmodel->kernel!=kernel.get()){
		delete newmodel->kernel;
		newmodel->kernel=kernel.get();
	}

	// extract SVs
	int SVnum=0;
	for(SVMModel::iterator Im=newmodel->begin(),Em=newmodel->end();Im!=Em;++Im,++SVnum){
		int jtIdx=svJumpTable.size();

		std::pair<SVMap::iterator,bool > insertion=supportVectors.insert(std::make_pair(Im->get(),jtIdx));
		bool svIsNew=insertion.second;
		SVMap::iterator inspair=insertion.first;

		jtIdx = inspair->second;	// if insertion failed, jtIdx equals the existing idx
		if(svIsNew){
			// supportvector did NOT exist yet, add to jt
			svJumpTable.push_back(*Im);
		}

		newmodel->redirectSV(Im,svJumpTable.at(jtIdx));

		// update index
		SVindex.push_back(jtIdx);
	}
	m.release();
}

std::string SVMEnsembleImpl::translate(const std::string &label) const{
	if(labelmap.empty())
		return std::string(label);

	LabelMap::const_iterator F=labelmap.find(label);
	if(F==labelmap.end())
		exit_with_err("Translating unknown label!");

	return F->second;
}

double SVMEnsembleImpl::density() const{
	size_t maxdim=0;
	size_t totallength=0;

	for(sv_const_iterator I=sv_begin(),E=sv_end();I!=E;++I){
		size_t len=(*I)->numNonzero();
		if(len!=0){
			totallength+=len;
			size_t thissize=(*I)->rbegin()->first;
			maxdim=maxdim > thissize ? maxdim : thissize;
		}
	}
	return static_cast<double>(totallength)/maxdim;
}

// fixme not exactly efficient
std::string SVMEnsembleImpl::positive_label() const{
	return labelmap.find(models.begin()->first->positive_label())->second;
}
// fixme not exactly efficient
std::string SVMEnsembleImpl::negative_label() const{
	return labelmap.find(models.begin()->first->negative_label())->second;
}
size_t SVMEnsembleImpl::num_outputs() const{
	return size();
}
std::vector<double> SVMEnsembleImpl::predict_by_cache(const std::vector<double>& cache) const{

	// initialize voting map: <label, vote count>
	std::map<string,unsigned> votes;
	const SVMModel *firstmodel=models.begin()->first;
	for(unsigned i=0;i<firstmodel->getNumClasses();++i)
		votes.insert(std::make_pair(translate(firstmodel->getLabel(i)),0));

	std::vector<double> decision_vals;
	decision_vals.reserve(size());

	std::map<string,unsigned>::iterator Iv, Ev;
	for(const_iterator I=begin(),E=end();I!=E;++I){
		// make predictions with cached kernel evaluations
		unsigned modelsize=I->first->size();
		std::vector<double> kernelevals(modelsize,0);

		for(unsigned idx=0;idx<modelsize;++idx){
			kernelevals[idx]=cache.at(getSVindex(idx+I->second));
		}

		decision_vals.emplace_back(I->first->predict_by_cache(kernelevals));
	}

	return std::move(decision_vals);
}

Prediction SVMEnsembleImpl::decval2prediction(std::vector<double>&& decision_vals) const{

	Prediction pred(size()+1);
	unsigned numpos = std::count_if(decision_vals.begin(),decision_vals.end(),
			[](double decval){ return decval > 0; });
	if(2*numpos > size()){
		pred.setLabel(positive_label());
		pred[0]=1.0*numpos/size();
	}else{
		pred[0]=1.0-1.0*numpos/size();
		pred.setLabel(negative_label());
	}
	std::copy(decision_vals.begin(),decision_vals.end(),pred.begin()+1);
	return std::move(pred);
}

Prediction SVMEnsembleImpl::predict(const SparseVector &x) const{
	std::vector<double> decvals=decision_value(x);
	return decval2prediction(std::move(decvals));
}

Prediction SVMEnsembleImpl::predict(const std::vector<double>& x) const{
	std::vector<double> decvals=decision_value(x);
	return decval2prediction(std::move(decvals));
}

std::vector<double> SVMEnsembleImpl::decision_value(const SparseVector &x) const{
	assert(models.size()>0 && "Trying to make predictions with an empty ensemble!");

	// maintain records of which kernel evaluations have already been performed
	unsigned numdistinctSV=svJumpTable.size();

	// fill the cache
	std::vector<double> cache;
	cache.reserve(numdistinctSV);

	unsigned i=0;
	for(SVDeque::const_iterator I=svJumpTable.begin(),E=svJumpTable.end();I!=E;++I,++i)
		cache.emplace_back(kernel->k_function(I->get(),&x));

	return predict_by_cache(cache);
}
std::vector<double> SVMEnsembleImpl::decision_value(const std::vector<double> &x) const{
	assert(models.size()>0 && "Trying to make predictions with an empty ensemble!");

	// maintain records of which kernel evaluations have already been performed
	unsigned numdistinctSV=svJumpTable.size();

	// fill the cache
	std::vector<double> cache;
	cache.reserve(numdistinctSV);

	if(denseSVs.size() != numdistinctSV){
		denseSVs.clear();
		// initialize the dense SV cache, only happens once
		denseSVs.reserve(numdistinctSV);
		for(sv_const_iterator I=sv_begin(),E=sv_end();I!=E;++I)
			denseSVs.emplace_back((*I)->dense());
	}

	for(unsigned i=0;i<numdistinctSV;++i)
		cache.emplace_back(kernel->k_function(denseSVs[i].begin(),denseSVs[i].end(),x.begin(),x.end()));

	return predict_by_cache(cache);
}


unsigned SVMEnsembleImpl::getSVindex(unsigned ensembleidx) const{
	return SVindex.at(ensembleidx);
}

SVMEnsembleImpl::~SVMEnsembleImpl(){
	for(iterator I=begin(),E=end();I!=E;++I)
		delete I->first;
}

void SVMEnsembleImpl::printSV(std::ostream &os, int SVidx) const{
	os << *svJumpTable[SVidx] << std::endl;
}

// fixme: inefficient
unsigned SVMEnsembleImpl::getSVindex(unsigned localidx, const SVMModel* const mod) const{
	modelCont::const_iterator F=models.find(const_cast<SVMModel* const>(mod));
	assert(F!=models.end() && "Model not found in Ensemble!");
	return SVindex.at(localidx+F->second);
}

/**
 * IO FUNCTIONS
 */

/**
 * ENSEMBLE FUNCTIONS
 */

//unique_ptr<BinaryModel> SVMEnsembleImpl::deserialize(std::istream &iss){
//	return unique_ptr<BinaryModel>(SVMEnsembleImpl::read(iss).release());
//}

unique_ptr<SVMEnsemble> SVMEnsembleImpl::read(std::istream &iss){

	std::string key, line;
	unsigned nummodels, numsv;

	getline(iss,line);
	std::istringstream liness(line);
	key.clear();
	liness >> key;
	if(key.compare("num_distinct_sv")!=0) // invalid model file!
		exit_with_err("Invalid ensemble SVM model: num_distinct_sv not specified.");
	liness >> numsv;

	// (optional) read labels: format "labels internal external internal external ..."
	SVMEnsembleImpl::LabelMap map;
	bool labelline=false;
	getline(iss,line);
	liness.clear();
	liness.str(line);
	key.clear();
	liness >> key;
	if(key.compare("labelmap")==0){
		labelline=true;
		std::string external, internal;
		while(liness.good()){
			internal.clear();
			external.clear();
			liness >> internal;
			if(!liness.good())
				exit_with_err("Attempting to read illegal label line from model file.");
			liness >> external;
			map.insert(make_pair(internal,external));
		}

		getline(iss,line);
		liness.clear();
		liness.str(line);
		key.clear();
		liness >> key;
	}

	// read num_models
	if(key.compare("num_models")!=0) // invalid model file!
		exit_with_err(std::string("Invalid ensemble SVM model: num_models not specified. Got: ")+key);
	liness >> nummodels;

	// read kernel
	unique_ptr<Kernel> kernel=Kernel::read(iss);

	// read SVs
	line.clear();
	getline(iss,line);
	if(line.compare("*** SV ***")!=0) // invalid model file!
		exit_with_err("Invalid ensemble SVM model: start of SVs at wrong position.");

	unique_ptr<SVMEnsembleImpl> ens(nullptr);
	if(labelline)
		ens.reset(new SVMEnsembleImpl(std::move(kernel),map));
	else
		ens.reset(new SVMEnsembleImpl(std::move(kernel)));

	for(unsigned i=0;i<numsv;++i){
		unique_ptr<SparseVector> sv=SparseVector::read(iss);
		ens->svJumpTable.emplace_back(sv.get());
		ens->supportVectors.insert(std::make_pair(sv.get(),i));
		sv.release();
	}

	unique_ptr<SVMEnsemble> ensemble(new SVMEnsemble(std::move(ens)));

	// read models
	line.clear();
	getline(iss,line);
	if(line.compare("*** MODELS ***")!=0) // invalid model file!
		exit_with_err("Invalid ensemble SVM model: start of models at wrong position.");

	for(unsigned i=0;i<nummodels;++i){
		unique_ptr<SVMModel> model=SVMModel::read(iss,ensemble.get()); // fixme: use factory?
		ensemble->add(std::move(model));
	}

	return std::move(ensemble);
}

void SVMEnsembleImpl::serialize(std::ostream& os) const{
	os.precision(PRECISION);
	os << "SVMEnsemble" << std::endl;
	os << "num_distinct_sv " << numDistinctSV() << std::endl;

	// output label map if it's not empty
	if(!labelmap.empty()){
		os << "labelmap";
		for(SVMEnsembleImpl::LabelMap::const_iterator I=labelmap.begin(),E=labelmap.end();I!=E;++I)
			os << " " << I->first << " " << I->second;
		os << std::endl;
	}

	os << "num_models " << size() << std::endl;
	os << *getKernel();
	os << "*** SV ***" << std::endl;
	for(SVMEnsembleImpl::sv_const_iterator I=sv_begin(),E=sv_end();I!=E;++I)
		os << **I << std::endl;
	os << "*** MODELS ***" << std::endl;
	for(SVMEnsembleImpl::const_iterator I=begin(),E=end();I!=E;++I){
		os << *I->first;
	}
}

//unique_ptr<SVMEnsemble> SVMEnsembleImpl::load(const string &fname){
//	std::ifstream file(fname.c_str(),std::ios::in);
//	unique_ptr<SVMEnsembleImpl> ensemble = SVMEnsembleImpl::read(file);
//	file.close();
//	return ensemble;
//}

























} // ensemble namespace




namespace ensemble{

SVMEnsemble::iterator SVMEnsemble::begin(){
	return iterator(pImpl->begin());
}
SVMEnsemble::iterator SVMEnsemble::end(){
	return iterator(pImpl->end());
}
SVMEnsemble::const_iterator SVMEnsemble::begin() const{
	return const_iterator(pImpl->begin());
}
SVMEnsemble::const_iterator SVMEnsemble::end() const{
	return const_iterator(pImpl->end());
}

SVMEnsemble::sv_iterator SVMEnsemble::sv_begin(){
	return sv_iterator(pImpl->sv_begin());
}
SVMEnsemble::sv_iterator SVMEnsemble::sv_end(){
	return sv_iterator(pImpl->sv_end());
}
SVMEnsemble::sv_const_iterator SVMEnsemble::sv_begin() const{
	return sv_const_iterator(pImpl->sv_begin());
}
SVMEnsemble::sv_const_iterator SVMEnsemble::sv_end() const{
	return sv_const_iterator(pImpl->sv_end());
}

const Kernel *SVMEnsemble::getKernel() const{
	return pImpl->getKernel();
}

SVMEnsemble::SVMEnsemble(unique_ptr<Kernel> kernel)
:Ensemble(),
 pImpl(new SVMEnsembleImpl(std::move(kernel)))
{}
SVMEnsemble::SVMEnsemble(unique_ptr<Kernel> kernel, const LabelMap &labelmap)
:Ensemble(),
 pImpl(new SVMEnsembleImpl(std::move(kernel), labelmap))
{}

SVMEnsemble::SVMEnsemble(std::vector<unique_ptr<SVMModel>>&& models, const std::string& positive, const std::string& negative)
:Ensemble(),
 pImpl(new SVMEnsembleImpl(this,std::move(models),positive,negative))
 {}

SVMEnsemble::SVMEnsemble(std::vector<unique_ptr<SVMModel>>&& models)
:Ensemble(),
 pImpl(new SVMEnsembleImpl(this,std::move(models)))
{}

SVMEnsemble::SVMEnsemble(unique_ptr<SVMEnsembleImpl> impl)
:Ensemble(),
 pImpl(std::move(impl))
{}


size_t SVMEnsemble::size() const{ return pImpl->size();}
size_t SVMEnsemble::numDistinctSV() const{ return pImpl->numDistinctSV(); }
size_t SVMEnsemble::numTotalSV() const{ return pImpl->numTotalSV(); }

std::shared_ptr<SparseVector> SVMEnsemble::getSV(unsigned ensidx){
	return pImpl->getSV(ensidx);
}

const SparseVector* SVMEnsemble::getSV(unsigned ensidx) const{
	return static_cast<const SVMEnsembleImpl*>(pImpl.get())->getSV(ensidx);
}

void SVMEnsemble::add(std::unique_ptr<SVMModel> m){
	pImpl->add(std::move(m),this);
}

std::string SVMEnsemble::translate(const std::string &label) const{
	return pImpl->translate(label);
}

double SVMEnsemble::density() const{
	return pImpl->density();
}

// fixme not exactly efficient
std::string SVMEnsemble::positive_label() const{
	return pImpl->positive_label();
}
// fixme not exactly efficient
std::string SVMEnsemble::negative_label() const{
	return pImpl->negative_label();
}
size_t SVMEnsemble::num_outputs() const{
	return size();
}
//std::vector<double> SVMEnsemble::predict_by_cache(const std::vector<double>& cache) const{
//
//	// initialize voting map: <label, vote count>
//	std::map<string,unsigned> votes;
//	const SVMModel *firstmodel=models.begin()->first;
//	for(unsigned i=0;i<firstmodel->getNumClasses();++i)
//		votes.insert(std::make_pair(translate(firstmodel->getLabel(i)),0));
//
//	std::vector<double> decision_vals;
//	decision_vals.reserve(size());
//
//	std::map<string,unsigned>::iterator Iv, Ev;
//	for(const_iterator I=begin(),E=end();I!=E;++I){
//		// make predictions with cached kernel evaluations
//		unsigned modelsize=I->first->size();
//		std::vector<double> kernelevals(modelsize,0);
//
//		for(unsigned idx=0;idx<modelsize;++idx){
//			kernelevals[idx]=cache.at(getSVindex(idx+I->second));
//		}
//
//		decision_vals.emplace_back(I->first->predict_by_cache(kernelevals));
//	}
//
//	return std::move(decision_vals);
//}

//Prediction SVMEnsemble::decval2prediction(std::vector<double>&& decision_vals) const{
//
//	Prediction pred(size()+1);
//	unsigned numpos = std::count_if(decision_vals.begin(),decision_vals.end(),
//			[](double decval){ return decval > 0; });
//	if(2*numpos > size()){
//		pred.setLabel(positive_label());
//		pred[0]=1.0*numpos/size();
//	}else{
//		pred[0]=1.0-1.0*numpos/size();
//		pred.setLabel(negative_label());
//	}
//	std::copy(decision_vals.begin(),decision_vals.end(),pred.begin()+1);
//	return std::move(pred);
//}

Prediction SVMEnsemble::predict(const SparseVector &x) const{
	return pImpl->predict(x);
}

Prediction SVMEnsemble::predict(const std::vector<double>& x) const{
	return pImpl->predict(x);
}

std::vector<double> SVMEnsemble::decision_value(const SparseVector &x) const{
	return pImpl->decision_value(x);
}
std::vector<double> SVMEnsemble::decision_value(const std::vector<double> &x) const{
	return pImpl->decision_value(x);
}

unsigned SVMEnsemble::getSVindex(unsigned ensembleidx) const{
	return pImpl->getSVindex(ensembleidx);
}

SVMEnsemble::~SVMEnsemble(){}

void SVMEnsemble::printSV(std::ostream &os, int SVidx) const{
	pImpl->printSV(os,SVidx);
}

unsigned SVMEnsemble::getSVindex(unsigned localidx, const SVMModel* const mod) const{
	return pImpl->getSVindex(localidx,mod);
}

/**
 * IO FUNCTIONS
 */

/**
 * ENSEMBLE FUNCTIONS
 */

REGISTER_BINARYMODEL_CPP(SVMEnsemble)

unique_ptr<BinaryModel> SVMEnsemble::deserialize(std::istream &iss){
	return unique_ptr<BinaryModel>(SVMEnsemble::read(iss).release());
}

unique_ptr<SVMEnsemble> SVMEnsemble::read(std::istream &iss){
	return SVMEnsembleImpl::read(iss);
}

void SVMEnsemble::serialize(std::ostream& os) const{
	pImpl->serialize(os);
}

std::ostream &operator<<(std::ostream &os, const SVMEnsemble &ens){
	ens.serialize(os);
	return os;
}

unique_ptr<SVMEnsemble> SVMEnsemble::load(const string &fname){
	std::ifstream file(fname.c_str(),std::ios::in);
	unique_ptr<SVMEnsemble> ensemble = SVMEnsemble::read(file);
	file.close();
	return ensemble;
}

} // ensemble namespace
