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
 * Workflow.cpp
 *
 *      Author: Marc Claesen
 */

#include <algorithm>
#include "BinaryWorkflow.hpp"
#include "Util.hpp"

using namespace ensemble::pipeline;

/*************************************************************************************************/

namespace{

/*************************************************************************************************/

template <typename T>
struct Helper;

template<typename Res, typename Arg>
struct Helper<MultistagePipe<Res(Arg)>>{
	static Res eval(const std::unique_ptr<MultistagePipe<Res(Arg)>>& p, Arg&& input){
		return (*p)(std::move(input));
	}
};

/*************************************************************************************************/

} // anonymous namespace

/*************************************************************************************************/

namespace ensemble{

/*************************************************************************************************/

BinaryWorkflow::BinaryWorkflow(	std::unique_ptr<Preprocessing> preprocess,
				std::unique_ptr<BinaryModel> pred,
				std::unique_ptr<Postprocessing> postprocess,
				double threshold
			)
:preprocessing(std::move(preprocess)),
 predictor(std::move(pred)),
 postprocessing(std::move(postprocess)),
 threshold(threshold),
 positive(predictor->positive_label()),
 negative(predictor->negative_label())
{
	assert(predictor.get() && "Predictor may not be nullptr!");
	if(postprocessing.get() && postprocessing->num_inputs())
		assert(postprocessing->num_inputs()==predictor->num_outputs()
				&& "Number of post processing inputs does not match predictor outputs!");
}
BinaryWorkflow::BinaryWorkflow(
				std::unique_ptr<BinaryModel> pred,
				std::unique_ptr<Postprocessing> postprocess,
				double threshold
			)
:preprocessing(nullptr),
 predictor(std::move(pred)),
 postprocessing(std::move(postprocess)),
 threshold(threshold),
 positive(predictor->positive_label()),
 negative(predictor->negative_label())
{
	assert(predictor.get() && "Predictor may not be nullptr!");
	if(postprocessing.get() && postprocessing->num_inputs())
		assert(postprocessing->num_inputs()==predictor->num_outputs()
				&& "Number of post processing inputs does not match predictor outputs!");
}
BinaryWorkflow::BinaryWorkflow(
				std::unique_ptr<BinaryModel> pred,
				double threshold
			)
:preprocessing(nullptr),
 predictor(std::move(pred)),
 postprocessing(nullptr),
 threshold(threshold),
 positive(predictor->positive_label()),
 negative(predictor->negative_label())
{
	assert(predictor.get() && "Predictor may not be nullptr!");
}

/*************************************************************************************************/

BinaryWorkflow::~BinaryWorkflow(){}

size_t BinaryWorkflow::num_outputs() const{
	return 1;
}

void BinaryWorkflow::set_preprocessing(std::unique_ptr<Preprocessing> pipe){
	preprocessing.reset(pipe.release());
}
void BinaryWorkflow::set_prediction(std::unique_ptr<BinaryModel> pipe){
	if(postprocessing.get() && postprocessing->num_inputs())
		assert(pipe->num_outputs() == postprocessing->num_inputs() &&
			"Number of predictor outputs does not match number of postprocessing inputs!");
	predictor.reset(pipe.release());
}
void BinaryWorkflow::set_postprocessing(std::unique_ptr<Postprocessing> pipe){
	if(pipe->num_inputs()) assert(pipe->num_inputs() == predictor->num_outputs() &&
			"Number of predictor outputs does not match number of postprocessing inputs!");
	postprocessing.reset(pipe.release());
}
void BinaryWorkflow::set_threshold(double threshold){
	this->threshold=threshold;
}
size_t BinaryWorkflow::num_inputs() const{
	if(preprocessing.get()) return preprocessing->num_inputs();
	return 0;
}

size_t BinaryWorkflow::num_predictor_outputs() const{
	return predictor->num_outputs();
}

Prediction BinaryWorkflow::predict(const SparseVector& v) const{
	std::vector<double> decvals=decision_value(v);

	if(decvals[0] > threshold)
		return Prediction(positive,std::move(decvals));
	return Prediction(negative,std::move(decvals));
}
Prediction BinaryWorkflow::predict(const std::vector<double> &i) const{
	SparseVector v(i);	// todo inefficient
	return predict(std::move(v));
}

std::vector<double> BinaryWorkflow::decision_value(const SparseVector &i) const{

	Vector intermediate;
	if(preprocessing.get()){
		SparseVector icp(i); // todo inefficient
		icp = Helper<Preprocessing>::eval(preprocessing,std::move(icp));
		intermediate = predictor->decision_value(std::move(icp));
	}else{
		intermediate = predictor->decision_value(i);
	}

	Vector result(1,0.0);
	result.reserve(intermediate.size()+1);
	std::copy(intermediate.begin(),intermediate.end(),std::back_inserter(result));
	result[0] = Helper<Postprocessing>::eval(postprocessing,std::move(intermediate));
	return std::move(result);
}
std::vector<double> BinaryWorkflow::decision_value(const std::vector<double> &i) const{
	SparseVector v(i);	// todo inefficient
	return decision_value(std::move(v));
}
const BinaryModel* BinaryWorkflow::get_predictor() const{
	return predictor.get();
}
std::unique_ptr<BinaryModel> BinaryWorkflow::release_predictor(){
	return std::move(predictor);
}

std::string BinaryWorkflow::positive_label() const{
	return positive;
}
std::string BinaryWorkflow::negative_label() const{
	return negative;
}

void BinaryWorkflow::print_preprocessing(std::ostream& os) const{
	if(preprocessing.get()) os << *preprocessing;
	else os << std::endl;
}
void BinaryWorkflow::print_predictor(std::ostream& os) const{
	os << *predictor;
}
void BinaryWorkflow::print_postprocessing(std::ostream& os) const{
	if(postprocessing.get()) os << *postprocessing;
	else os << std::endl;
}
void BinaryWorkflow::print_threshold(std::ostream& os) const{
	os << threshold;
}

/*************************************************************************************************/

void BinaryWorkflow::serialize(std::ostream& os) const{
	os << "BinaryWorkflow" << std::endl;

	os << "preprocessing" << std::endl;
	print_preprocessing(os);

	os << "predictor" << std::endl;
	print_predictor(os);

	os << "postprocessing" << std::endl;
	print_postprocessing(os);

	os << "threshold" << std::endl;
	print_threshold(os);
}

std::ostream& operator<<(std::ostream& os, const BinaryWorkflow& flow){
	flow.serialize(os);
	return os;
}

std::unique_ptr<BinaryModel> BinaryWorkflow::deserialize(std::istream& stream){
	/**
	 * Get preprocessor.
	 */
	typedef ensemble::PredicatedFactory<Preprocessing,
			const std::string&,std::istream&> PreprocessorFactory;
	typedef std::unique_ptr<Preprocessing> PreprocessingPtr;

	std::string line;
	getline(stream,line);
	assert(line.compare("preprocessing")==0 &&
			"Illegal format for binary workflow, expecting preprocessing!");

	line.clear();
	getline(stream,line);
	std::unique_ptr<Preprocessing> preprocessor(nullptr);
	if(!line.empty()){
		std::vector<PreprocessingPtr> preproc=PreprocessorFactory::Produce(line,stream);

		assert(preproc.size()<2 && "Error, retrieved multiple preprocessors from stream!");
		if(preproc.size()==1) preprocessor=std::move(preproc[0]);
	}

	/**
	 * Get predictor.
	 */
	typedef ensemble::PredicatedFactory<BinaryModel,
			const std::string&,std::istream&> PredictorFactory;
	typedef std::unique_ptr<BinaryModel> PredictorPtr;

	line.clear();
	getline(stream,line);
	assert(line.compare("predictor")==0 &&
				"Illegal format for binary workflow, expecting predictor!");

	line.clear();
	getline(stream,line);

	std::unique_ptr<BinaryModel> predictor(nullptr);
	std::vector<PredictorPtr> pred=PredictorFactory::Produce(line,stream);
	assert(pred.size()==1 && "Error, retrieved multiple or no predictors from stream!");
	predictor.reset(pred[0].release());
	pred[0].release();

	/**
	 * Get postprocessor.
	 */
	typedef ensemble::PredicatedFactory<Postprocessing,
			const std::string&,std::istream&> PostprocessorFactory;
	typedef std::unique_ptr<Postprocessing> PostprocessingPtr;

	line.clear();
	getline(stream,line);
	assert(line.compare("postprocessing")==0 &&
				"Illegal format for binary workflow, expecting postprocessing!");

	std::unique_ptr<Postprocessing> postprocessor(nullptr);
	getline(stream,line);
	if(!line.empty()){
		std::vector<PostprocessingPtr> postproc=PostprocessorFactory::Produce(line,stream);

		assert(postproc.size()<2 && "Error, retrieved multiple preprocessors from stream!");
		if(postproc.size()==1) postprocessor=std::move(postproc[0]);
	}

	/**
	 * Get threshold.
	 */

	line.clear();
	getline(stream,line);
	assert(line.compare("threshold")==0 &&
				"Illegal format for binary workflow, expecting threshold!");

	double thresh=0.0;
	stream >> thresh;
	return std::unique_ptr<BinaryModel>(new BinaryWorkflow(
				std::move(preprocessor),
				std::move(predictor),
				std::move(postprocessor),
				thresh
			));
}

REGISTER_BINARYMODEL_CPP(BinaryWorkflow)

/*************************************************************************************************/

std::unique_ptr<BinaryWorkflow> defaultBinaryWorkflow(std::unique_ptr<BinaryModel> model, bool mv){
	if(model->num_outputs()==1)
		return std::unique_ptr<BinaryWorkflow>(new BinaryWorkflow(std::move(model),0.0));

	typedef MultistagePipe<double(std::vector<double>)> PostProcess;
	std::unique_ptr<PostProcess> postprocessing;
	double threshold=0.5;
	if(mv){
		Factory<MajorityVote> f;
		postprocessing.reset(
				static_cast<PostProcess*>(f(model->num_outputs()).release()));
	}else{
		Factory<LogisticRegression> f;
		postprocessing.reset(
				static_cast<PostProcess*>(f(model->num_outputs()).release()));
	}

	return std::unique_ptr<BinaryWorkflow>
		(new BinaryWorkflow(std::move(model),std::move(postprocessing),threshold));
}

/*************************************************************************************************/

}

/*************************************************************************************************/
