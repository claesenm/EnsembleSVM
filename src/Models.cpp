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
 * Models.cpp
 *
 *      Author: Marc Claesen
 */

#include "Util.hpp"
#include "Models.hpp"
#include "Ensemble.hpp"
#include "io.hpp"
#include "LibSVM.hpp"
#include <cassert>
#include <algorithm>
#include <typeinfo>
#include <stdio.h>
#include <sstream>
#include <fstream>

using std::string;
using std::endl;

namespace{

string INENSEMBLE_STR("in_ensemble");
string CONSTANTS_STR("rho");
string NRCLASS_STR("nr_class");
string TOTALSV_STR("total_sv");
string LABEL_STR("label");
string NRSV_STR("nr_sv");
string SV_STR("SV");

std::streamsize PRECISION = 16;

/**
 * Reads class-related information from <is>.
 *
 * Specifically:
 * 	nr_class <unsigned>
 * 	total_sv <unsigned>
 * 	label <str> <str> ...
 * 	nr_sv <unsigned <unsigned> ...
 */
SVMModel::Classes readClasses(std::istream &is){
	string line, keyword;
	std::istringstream iss;

	unsigned nr_class, total_sv;

	// read nr_class
	getline(is,line);
	iss.str(line);
	iss >> keyword;
	if(keyword.compare(NRCLASS_STR)!=0)
		exit_with_err(string("Invalid model file, expecting nr_class but got ") + line);
	iss >> nr_class;
	iss.clear();

	// read total_sv
	getline(is,line);
	iss.str(line);
	iss >> keyword;
	if(keyword.compare(TOTALSV_STR)!=0)
		exit_with_err(string("Invalid model file, expecting total_sv but got ") + line);
	iss >> total_sv;
	iss.clear();

	SVMModel::Classes classes(nr_class,make_pair(string(""),0));

	// read labels
	getline(is,line);
	iss.str(line);
	iss >> keyword;
	if(keyword.compare(LABEL_STR)!=0)
		exit_with_err(string("Invalid model file, expecting label but got ") + line);
	for(unsigned i=0;i<nr_class;++i){
		string label;
		iss >> label;
		classes.at(i).first=label;
	}
	iss.clear();

	// read nr_sv
	getline(is,line);
	iss.str(line);
	iss >> keyword;
	if(keyword.compare(NRSV_STR)!=0)
		exit_with_err(string("Invalid model file, expecting nr_sv but got ") + line);
	unsigned nr_sv;
	for(unsigned i=0;i<nr_class;++i){
		iss >> nr_sv;
		classes.at(i).second=nr_sv;
	}
	iss.clear();

	// sanity check
	unsigned SVinclasses=0;
	for(SVMModel::Classes::const_iterator I=classes.begin(),E=classes.end();I!=E;++I)
		SVinclasses+=I->second;
	if(SVinclasses!=total_sv)
		exit_with_err(string("Illegal model, total_sv != sum(#SV in classes)!"));

	return std::move(classes);
}

/**
 * Reads the constants line from an SVMModel stream in <is>.
 *
 * This line should have the following form (k amount of classes):
 * constants <double 1> <double 2> ... <double k*(k-1)/2>
 */
std::vector<double> readConstants(std::istream &is, unsigned numclasses){
	string line, keyword;
	getline(is,line);
	std::istringstream iss;
	iss.str(line);
	iss >> keyword;
	if(keyword.compare(CONSTANTS_STR)!=0)
		exit_with_err(string("Invalid model file: expecting constants but recieved: ") + line);

	unsigned numconstants = numclasses*(numclasses-1)/2;
	std::vector<double> constants(numconstants,0);
	for(unsigned i=0;i<numconstants;++i)
		iss >> constants[i];
	return std::move(constants);
}

} // anonymous namespace

namespace ensemble{

Prediction::Prediction(unsigned numdecisions):scores(numdecisions,0){}
Prediction::Prediction(const string &label, const ScoreCont &scores):label(label),scores(scores){}
Prediction::Prediction(const string &label, ScoreCont&& scores):label(label),scores(scores){}
Prediction::Prediction(string&& label, ScoreCont&& scores):label(std::move(label)),scores(std::move(scores)){}
Prediction::Label Prediction::getLabel() const{ return label; }
Prediction::iterator Prediction::begin(){ return scores.begin(); }
Prediction::iterator Prediction::end(){ return scores.end(); }
Prediction::const_iterator Prediction::begin() const{ return scores.begin(); }
Prediction::const_iterator Prediction::end() const{ return scores.end(); }
void Prediction::setLabel(const Label &label){ this->label=label; }
void Prediction::setScore(Score score, unsigned idx){ scores.at(idx)=score; }
Prediction::Score Prediction::getScore(unsigned idx) const{ return scores.at(idx); }
Prediction::Score &Prediction::operator[](unsigned idx){ return scores.at(idx); }

Model::Model(){}
Model::Model(const Model &orig){}
Prediction Model::predict(const struct svm_node *x) const{
	SparseVector sv(x);
	return predict(sv);
}
std::vector<double> Model::decision_value(const struct svm_node *x) const{
	SparseVector sv(x);
	return decision_value(sv);
}


BinaryModel::BinaryModel():Model(){}
BinaryModel::BinaryModel(const BinaryModel& orig):Model(orig){}
BinaryModel::~BinaryModel(){}

std::ostream &operator<<(std::ostream &os, const BinaryModel &model){
	model.serialize(os);
	return os;
}

std::unique_ptr<BinaryModel> BinaryModel::deserialize(std::istream& is){
	typedef SelectiveFactory<BinaryModel,const std::string&,std::istream&> MetaFactory;
	std::string line;
	getline(is,line);

	assert(!line.empty() && "First line in model file must not be blank.");
	assert(is && "Unable to read from stream.");
	auto models = MetaFactory::Produce(line,is);
	if(models.size() == 0) return std::unique_ptr<BinaryModel>(nullptr);
	assert(models.size() < 2 && "Retrieved multiple models!");
	return std::move(models[0]);
}

unique_ptr<BinaryModel> BinaryModel::load(const string &fname){
	std::ifstream file(fname.c_str(),std::ios::in);
	assert(file && "Unable to open file.");
	return BinaryModel::deserialize(file);
}

const std::vector<double> &SVMModel::getConstants() const{	return constants; }
double SVMModel::getConstant(unsigned i) const{ return constants.at(i); }

SVMModel::SVMModel(const SVMModel &m)
:BinaryModel(m),
 ens(nullptr),
 weights(m.weights),
 classes(m.classes),
 constants(m.constants),
 kernel(nullptr)
{
	unique_ptr<Kernel> ker = m.getKernel()->clone();
	kernel = ker.release();
	SVs.reserve(m.size());

	// deep copy SVs
	for(const_iterator Im=m.begin(),Em=m.end();Im!=Em;++Im){
		SVs.emplace_back(new SparseVector(*Im->get()));
	}
}
SVMModel::SVMModel(SVMModel&& m)
:BinaryModel(std::move(m)),
 ens(m.ens),
 weights(std::move(m.weights)),
 classes(std::move(m.classes)),
 constants(std::move(m.constants)),
 kernel(std::move(m.kernel))
{}

SVMModel::SVMModel(SV_container&& SVs, Weights&& weights, Classes&& classes, std::vector<double>&& constants, unique_ptr<Kernel> kernel)
:BinaryModel(),
 ens(nullptr),
 SVs(SVs),
 weights(weights),
 classes(classes),
 constants(constants),
 kernel(kernel.release())
{}

SVMModel::SVMModel(SV_container&& SVs, Weights&& weights, Classes&& classes, std::vector<double>&& constants, const SVMEnsemble *ens)
:BinaryModel(),
 ens(ens),
 SVs(SVs),
 weights(weights),
 classes(classes),
 constants(constants),
 kernel(nullptr) // adding an SVMModel to 2 ensembles is impossible, e.g. kernel is const in SVMModel.
{
	// const_casting the Kernel to avoid copying (UserdefKernel can be huge)
	// kernel in an SVMModel is effectively const because it can only get changed when adding the SVMModel to an ensemble
	// or when destroying a standalone SVMModel
	// adding an SVMModel to 2 ensembles is impossible, e.g. kernel is const in SVMModel.
	kernel = const_cast<Kernel*>(ens->getKernel());
}

size_t SVMModel::size() const{	return SVs.size(); }
const SparseVector& SVMModel::operator[](int idx) const{ return *SVs.at(idx); }

void SVMModel::redirectSV(iterator &I, std::shared_ptr<SparseVector> newtarget){
	if(*I!=newtarget){ // safety to prevent redirecting to the same element
		assert((*I==nullptr || **I==*newtarget) && "Trying to redirect to unequal SV!"); // sanity check
		assert(newtarget && "Redirecting to nullptr!");

		SVs.at(I-SVs.begin())=newtarget;
	}
}

SVMModel::~SVMModel(){	// if this SVMModel is inside an ensemble, do not delete SVs and kernel
	if(ens==nullptr){
		delete kernel;
	}
}

unsigned SVMModel::getNumClasses() const{ return classes.size(); }
unsigned SVMModel::getNumSV(unsigned i) const{ return classes.at(i).second; }
string SVMModel::getLabel(unsigned i) const{ return classes.at(i).first; }
void SVMModel::updateLabel(const std::string &current, const std::string &replacement){
	bool replaced=false;
	for(Classes::iterator I=classes.begin(),E=classes.end();I!=E;++I){
		if(current.compare(I->first)==0){
			I->first=replacement;
			replaced=true;
		}
	}
	if(!replaced)
		exit_with_err(std::string("Unable to retrieve class label '") + current + "' to replace with '" + replacement + "'.");
}

Prediction SVMModel::predict(const SparseVector &v) const{
	std::vector<double> value = decision_value(v);
	if(value[0] > 0) return Prediction(getLabel(0),value);
	else return Prediction(getLabel(1),std::move(value));
}

Prediction SVMModel::predict(const std::vector<double> &v) const{
	std::vector<double> value = decision_value(v);
	if(value[0] > 0) return Prediction(getLabel(0),value);
	else return Prediction(getLabel(1),std::move(value));
}

std::vector<double> SVMModel::decision_value(const SparseVector &v) const{
	unsigned numSV=size(), i=0;
	std::vector<double> kernelevals(numSV,0);

	double value=0;
	for(SVMModel::const_iterator I=begin(),E=end();I!=E;++I,++i){
		value=kernel->k_function(I->get(),&v);
		kernelevals[i]=value;
	}

	std::vector<double> decval(1,predict_by_cache(kernelevals));
	return std::move(decval);
}
// fixme: inefficient implementation
std::vector<double> SVMModel::decision_value(const std::vector<double> &v) const{
	std::vector<double> kernelevals;
	kernelevals.reserve(size());

	SparseVector sparse(v); // inefficient
	for(auto& sv: *this){
		kernelevals.emplace_back(kernel->k_function(sv.get(),&sparse));
	}

	std::vector<double> decval(1,predict_by_cache(kernelevals));
	return std::move(decval);
}

double SVMModel::predict_by_cache(const std::vector<double> &kernelevals) const{
	assert(kernelevals.size()==size() && "Invalid kernel cache vector supplied!");
	return svm_predict_values(kernelevals);
}

double SVMModel::svm_predict_values(const std::vector<double> &kernelevals) const{
	unsigned i;
	unsigned nr_class = getNumClasses();

	std::vector<unsigned> start(nr_class,0);
	for(i=1;i<nr_class;++i)
		start[i]=start[i-1]+getNumSV(i-1);

	std::vector<int> vote(nr_class,0);

	// we deal with binary svm, so no need for loops
	// commented out in case we want to extend to multiclass later
	unsigned p=0, j=1;
	i=0;
//	for(i=0;i<nr_class;i++){
//		for(unsigned j=i+1;j<nr_class;j++)
//		{
			double sum = 0;
			unsigned si = start[i];
			unsigned sj = start[j];
			unsigned ci = getNumSV(i);
			unsigned cj = getNumSV(j);

			unsigned k;
			Weights::const_iterator Icoef1=weight_begin(j-1), Ecoef1=weight_end(j-1);
			Weights::const_iterator Icoef2=weight_begin(i), Ecoef2=weight_end(i);

			Icoef1+=si;
			Icoef2+=sj;
			for(k=0;k<ci;k++)
				sum += *Icoef1++ * kernelevals[si+k];
			for(k=0;k<cj;k++)
				sum += *Icoef2++ * kernelevals[sj+k];

			sum -= getConstant(p);
			return sum;

//			pred[p] = sum;

//			if(pred[p] > 0)
//				++vote[i];
//			else
//				++vote[j];
//			p++;
//		}
//	}

//	unsigned vote_max_idx = 0;
//	for(i=1;i<nr_class;i++)
//		if(vote[i] > vote[vote_max_idx])
//			vote_max_idx = i;
}

unsigned SVMModel::getStartOfClass(unsigned classidx) const{
	unsigned start=0;
	for(unsigned j=0;j<classidx;++j)
		start+=classes.at(j).second;
	return start;
}

std::string SVMModel::positive_label() const{
	return getLabel(0);
}
std::string SVMModel::negative_label() const{
	return getLabel(1);
}
size_t SVMModel::num_outputs() const{ return 1; }

SVMModel::iterator SVMModel::begin(){return SVs.begin();}
SVMModel::iterator SVMModel::end(){ return SVs.end(); }
SVMModel::const_iterator SVMModel::begin() const{ return SVs.begin(); }
SVMModel::const_iterator SVMModel::end() const{ return SVs.end(); }
SVMModel::iterator SVMModel::begin(unsigned classidx){
	SVMModel::iterator I=SVs.begin()+getStartOfClass(classidx);
	return I;
}
SVMModel::iterator SVMModel::end(unsigned classidx){
	unsigned numsv=classes.at(classidx).second;
	SVMModel::iterator I=SVs.begin()+getStartOfClass(classidx)+numsv;
	return I;
}
SVMModel::const_iterator SVMModel::begin(unsigned classidx) const{
	SVMModel::const_iterator I=SVs.begin()+getStartOfClass(classidx);
	return I;
}
SVMModel::const_iterator SVMModel::end(unsigned classidx) const{
	unsigned numsv=classes.at(classidx).second;
	SVMModel::const_iterator I=SVs.begin()+getStartOfClass(classidx)+numsv;
	return I;
}

SVMModel::weight_iter SVMModel::weight_begin(){ return weights.begin();}
SVMModel::weight_iter SVMModel::weight_end(){ return weights.end(); }
SVMModel::const_weight_iter SVMModel::weight_begin() const{ return weights.begin(); }
SVMModel::const_weight_iter SVMModel::weight_end() const{ return weights.end(); }
SVMModel::weight_iter SVMModel::weight_begin(unsigned decfunidx){
	assert(decfunidx<getNumClasses()-1 && "Illegal decision function index.");
	SVMModel::weight_iter I=weights.begin()+decfunidx*size();
	return I;
}
SVMModel::weight_iter SVMModel::weight_end(unsigned decfunidx){
	assert(decfunidx<getNumClasses()-1 && "Illegal decision function index.");
	SVMModel::weight_iter I=weights.begin()+(1+decfunidx)*size();
	return I;
}
SVMModel::const_weight_iter SVMModel::weight_begin(unsigned decfunidx) const{
	assert(decfunidx<getNumClasses()-1 && "Illegal decision function index.");
	SVMModel::const_weight_iter I=weights.begin()+decfunidx*size();
	return I;
}
SVMModel::const_weight_iter SVMModel::weight_end(unsigned decfunidx) const{
	assert(decfunidx<getNumClasses()-1 && "Illegal decision function index.");
	SVMModel::const_weight_iter I=weights.begin()+(1+decfunidx)*size();
	return I;
}

const Kernel *SVMModel::getKernel() const{ return kernel; }

/**
 * IO FUNCTIONS
 */
std::ostream &operator<<(std::ostream &os, const SVMModel &model){
	model.serialize(os);
	return os;
}

void SVMModel::serialize(std::ostream& os) const{
	std::ostringstream stream(std::ostringstream::out);

	os << "SVMModel" << std::endl;

	// if the model does not belong to an ensemble: print kernel properties
	if(ens == nullptr){
		os << INENSEMBLE_STR << " 0" << endl;
		os << *getKernel();
	}else{
		os << INENSEMBLE_STR << " 1" << endl;
	}

	os << NRCLASS_STR << " " << getNumClasses() << endl;
	os << TOTALSV_STR << " " << size() << endl;

	os << LABEL_STR;
	for(unsigned i=0;i<getNumClasses();++i)
		os << " " << getLabel(i);
	os << endl;

	os << NRSV_STR;
	for(unsigned i=0;i<getNumClasses();++i)
		os << " " << getNumSV(i);
	os << endl;


	os << CONSTANTS_STR;
	unsigned k=getNumClasses();
	for(unsigned i=0;i<k*(k-1)/2;++i)
		os << " " << getConstant(i);
	os << endl;

	os << SV_STR << std::endl;

	std::list<SVMModel::const_weight_iter> iterators;
	for(unsigned i=0;i<getNumClasses()-1;++i)
		iterators.push_back(weight_begin(i));

	if(ens==nullptr){
		// model does not belong to an ensemble, so print SVs in full
		// print rows of the following form (k classes):
		// <weight 1> <weight 2> ... <weight k-1> SV
		for(SVMModel::const_iterator I=begin(),E=end();I!=E;++I){
			// print out weights per 1v1 pair
			for(std::list<SVMModel::const_weight_iter>::iterator Iw=iterators.begin(),Ew=iterators.end();Iw!=Ew;++Iw){
				os << **Iw << " ";
				++*Iw;
			}
			// print out SV
			os << **I << endl;
		}
	}else{
		// model belongs to an ensemble
		// print rows of the following form (k classes):
		// <weight 1> <weight 2> ... <weight k-1> <index in ensemble SVs>
		for(unsigned i=0;i<size();++i){
			// print out weights per 1v1 pair
			for(std::list<SVMModel::const_weight_iter>::iterator Iw=iterators.begin(),Ew=iterators.end();Iw!=Ew;++Iw){
				os << **Iw << " ";
				++*Iw;
			}
			// print out SV
			os << ens->getSVindex(i,this) << endl;
		}
	}

//	return os;
}


unique_ptr<BinaryModel> SVMModel::deserialize(std::istream& is){
	return std::unique_ptr<BinaryModel>(SVMModel::read(is).release());
}

unique_ptr<SVMModel> SVMModel::read(std::istream &is, SVMEnsemble *ens){

	unique_ptr<SVMModel> model;

	string line, keyword;
	if(ens){ // fixme: clean this up
		getline(is,line); // read SVMModel line
		line.clear();
	}

	// check first line to see if we're dealing with a libsvm model or a true SVMModel
	getline(is,line);

	std::istringstream iss;
	iss.str(line);
	iss >> keyword;
	if(keyword.compare(INENSEMBLE_STR)==0){
		unsigned inensemble;
		iss >> inensemble;

		// sanity check: if the model belongs to an ensemble, ens cannot be nullptr
		assert((inensemble==1)==(ens!=nullptr));

		// acquire kernel
		unique_ptr<Kernel> kernel(nullptr);
		if(inensemble==0){
			// read kernel
			kernel = Kernel::read(is);
		}

		Classes&& classes = readClasses(is);

		// read constants
		std::vector<double>&& constants = readConstants(is,classes.size());

		// calculate total amount of SVs in this SVMModel
		unsigned numSV=0;
		for(unsigned i=0;i<classes.size();++i)
			numSV+=classes.at(i).second;

		unsigned k=classes.size();
		SVMModel::Weights weights(numSV*(k-1),0);
		SVMModel::SV_container SVs;
		SVs.reserve(numSV);

		getline(is,line);
		if(line.compare(SV_STR)!=0)
			exit_with_err(string("Invalid model file: expecting SV but got ") + line);

		// fill weights and SVs by reading lines of the form (k classes):
		//	<weight 1> ... <weight k-1> <SV> (standalone model)
		// or
		//	<weight> ... <weight k-1> <SVidx> (model belonging to ensemble)
		for(unsigned i=0;i<numSV;++i){
			if(!is.good())
				exit_with_err("Premature end of file while reading model!");

			iss.clear();
			getline(is,line);
			iss.str(line);

			double weight;
			for(unsigned j=0;j<k-1;++j){
				iss >> weight;
				weights.at(i+j*numSV)=weight;
			}

			if(inensemble==1){
				// model is part of ensemble, read the SV index
				unsigned svidx;
				iss >> svidx;
				SVs.emplace_back(ens->getSV(svidx));
			}else{
				// model is standalone, read the SV
				unique_ptr<SparseVector> SVptr =SparseVector::read(iss);
				SVs.emplace_back(std::make_shared<SparseVector>(*SVptr.release()));
			}
		}

		SVMModel *model;
		if(inensemble)
			model = new SVMModel(std::move(SVs),std::move(weights),std::move(classes),std::move(constants),ens);
		else
			model = new SVMModel(std::move(SVs),std::move(weights),std::move(classes),std::move(constants),std::move(kernel));

		return unique_ptr<SVMModel>(model);

	}else{ // non-generic model
		throw invalid_file_exception();
	}
	return model;
}

unique_ptr<SVMModel> SVMModel::load(const string &fname){
	std::ifstream file(fname.c_str(),std::ios::in);

	unique_ptr<SVMModel> model;
	try{
		model = SVMModel::read(file,nullptr);
	}catch(invalid_file_exception &e){
		const char *filename=fname.c_str();
		svm_model *libsvmmodel = svm_load_model(filename);
		model = LibSVM::convert(unique_ptr<svm_model>(libsvmmodel));
	}
	file.close();
	return model;
}

std::ostream &operator<<(std::ostream &os, const Prediction &pred){
	os << pred.label;
	for(Prediction::const_iterator I=pred.begin(),E=pred.end();I!=E;++I)
		os << " " << *I;

	return os;
}

} // ensemble namespace
