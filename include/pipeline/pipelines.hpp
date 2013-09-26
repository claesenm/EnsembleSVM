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
 * pipelines.hpp
 *
 *      Author: Marc Claesen
 */

#ifndef PIPELINES_HPP_
#define PIPELINES_HPP_

/*************************************************************************************************/

#include "pipeline/blocks.hpp"
#include "PredicatedFactory.hpp"
#include "SparseVector.hpp"
#include <iostream>
#include <string>

using ensemble::SparseVector;

/*************************************************************************************************/

namespace pipeline{

/*************************************************************************************************/

template <typename T>
struct deserializer;

template <typename Res, typename Arg>
struct deserializer<Res(Arg)>{
	// the factory that will produce pipelines for this deserializer
	typedef ensemble::PredicatedFactory<MultistagePipe<Res(Arg)>,const std::string&,std::istream&> MetaFactory;

	std::unique_ptr<MultistagePipe<Res(Arg)>> operator()(std::istream& stream){
		std::string line;
		getline(stream,line);

		auto pipes = MetaFactory::Produce(line,stream);
		if(pipes.size()==0) return std::unique_ptr<MultistagePipe<Res(Arg)>>(nullptr);
		assert(pipes.size()==1 && "Error: retrieved multiple or no pipelines from stream.");

		return std::move(pipes[0]);
	}
};

/*************************************************************************************************/

/**
 * Weighted majority voting pipeline.
 * Input: std::vector<double>
 * Output: double
 *
 * Pipeline details:
 * 1. Threshold	- threshold = 0.0 (default), above = 1.0, below = 0.0
 * 2. Scale		- configurable (default: 1)
 * 3. Average	- weighted average
 */
MULTISTAGEPIPELINE(MajorityVote,double,std::vector<double>)

template <>
struct Factory<MajorityVote>{
	MULTISTAGEPIPELINE_FACTORY_TYPEDEFS(MajorityVote)
	typedef std::vector<double> Vector;

	static std::unique_ptr<MultistagePipe<Res(Arg)>> deserialize(std::istream& is){
		auto thresh = Factory<Threshold<Vector(Vector)>>::deserialize(is);
		auto scaled = Factory<Scale<Vector(Vector)>>::deserialize(is,std::move(thresh));
		auto avg = 	Factory<Average<double(Vector)>>::deserialize(is,std::move(scaled));
		return std::unique_ptr<MultistagePipe<Res(Arg)>>(new MajorityVote(std::move(avg)));
	}
	/**
	 * Constructs a basic MajorityVote pipeline with specified number of inputs.
	 * If numinputs remains unspecified, accepts any amount of inputs.
	 */
	std::unique_ptr<MajorityVote>
	operator()(size_t numinputs=0) const{
		Factory<Threshold<Vector(Vector)>> fact_thresh;
		auto thresh = fact_thresh(0.0,1.0,0.0,numinputs);
		Factory<Scale<Vector(Vector)>> fact_scaled;
		auto scale = fact_scaled(std::move(thresh),1.0);
		Factory<Average<double(Vector)>> fact_avg;
		auto avg = fact_avg(std::move(scale));
		return std::unique_ptr<MajorityVote>(new MajorityVote(std::move(avg)));
	}
	/**
	 * Constructs a basic weighted MajorityVote pipeline with
	 * specified scaling coefficients and threshold level.
	 */
	std::unique_ptr<MajorityVote>
	operator()(std::vector<double>&& coeffs, double threshold=0.0) const{
		double divisor = std::accumulate(coeffs.begin(),coeffs.end(),0.0);
		Factory<Threshold<Vector(Vector)>> fact_thresh;
		auto thresh = fact_thresh(threshold,1.0,0.0,coeffs.size());
		Factory<Scale<Vector(Vector)>> fact_scaled;
		auto scale = fact_scaled(std::move(thresh),std::move(coeffs));
		Factory<Average<double(Vector)>> fact_avg;
		auto avg = fact_avg(std::move(scale),divisor);
		return std::unique_ptr<MajorityVote>(new MajorityVote(std::move(avg)));
	}
	/**
	 * Constructs a basic weighted MajorityVote pipeline with
	 * specified scaling coefficients and threshold level.
	 */
	std::unique_ptr<MajorityVote>
	operator()(std::vector<double>& coeffs, double threshold=0.0) const{
		return operator()(std::move(coeffs),threshold);
	}
	/**
	 * Constructs a basic weighted MajorityVote pipeline with
	 * specified scaling coefficients and threshold level.
	 */
	std::unique_ptr<MajorityVote>
	operator()(const std::vector<double>& coeffs, double threshold=0.0) const{
		Vector tmp(coeffs);
		return operator()(std::move(tmp),threshold);
	}
};

MULTISTAGEPIPELINE_POST_FACTORY(MajorityVote)

/*************************************************************************************************/

/**
 * Logistic regression pipeline.
 * Input: std::vector<double>
 * Output: double
 *
 * Pipeline details:
 * 1. Scale		- configurable: coeffs
 * 2. Sum
 * 3. Offset	- configurable: offset
 * 4. Logistic
 */
MULTISTAGEPIPELINE(LogisticRegression,double,std::vector<double>)

template <>
struct Factory<LogisticRegression>{
	MULTISTAGEPIPELINE_FACTORY_TYPEDEFS(LogisticRegression)
	typedef std::vector<double> Vector;

	static std::unique_ptr<MultistagePipe<Res(Arg)>>
	deserialize(std::istream& is){
		auto scaled = Factory<Scale<Vector(Vector)>>::deserialize(is);
		auto sum = Factory<Sum<double(Vector)>>::deserialize(is,std::move(scaled));
		auto offset = Factory<Offset<double(double)>>::deserialize(is,std::move(sum));
		auto logistic = Factory<Logistic<double(double)>>::deserialize(is,std::move(offset));
		return std::unique_ptr<MultistagePipe<Res(Arg)>>(new LogisticRegression(std::move(logistic)));
	}

	/**
	 * Creates a logistic regression pipeline with given scale coefficients and offset.
	 */
	std::unique_ptr<LogisticRegression>
	operator()(std::vector<double>&& scale_coeffs, double offset=0) const{
		Factory<Scale<Vector(Vector)>> f_scale;
		auto scale = f_scale(std::move(scale_coeffs));

		Factory<Sum<double(Vector)>> f_sum;
		auto sum = f_sum(std::move(scale));

		Factory<Offset<double(double)>> f_offset;
		auto off = f_offset(std::move(sum),offset);

		Factory<Logistic<double(double)>> f_logistic;
		auto logistic = f_logistic(std::move(off));
		return std::unique_ptr<LogisticRegression>(new LogisticRegression(std::move(logistic)));
	}

	/**
	 * Creates a default logistic regression pipeline, with scale coeffs==1 and offset==0.
	 */
	std::unique_ptr<LogisticRegression>
	operator()(size_t numinputs=0) const{
		Factory<Scale<Vector(Vector)>> f_scale;
		auto scale = f_scale(1.0,numinputs);

		Factory<Sum<double(Vector)>> f_sum;
		auto sum = f_sum(std::move(scale));

		Factory<Offset<double(double)>> f_offset;
		auto off = f_offset(std::move(sum),0.0);

		Factory<Logistic<double(double)>> f_logistic;
		auto logistic = f_logistic(std::move(off));
		return std::unique_ptr<LogisticRegression>(new LogisticRegression(std::move(logistic)));
	}
};

MULTISTAGEPIPELINE_POST_FACTORY(LogisticRegression)

/*************************************************************************************************/

/**
 * Linear normalization pipeline.
 * Input: SparseVector
 * Output: SparseVector
 *
 * Pipeline details:
 * 1. Scale		- configurable: scale_coeffs
 * 2. Offset	- configurable: offset_coeffs
 */
MULTISTAGEPIPELINE(NormalizeLinear,SparseVector,SparseVector)

template <>
struct Factory<NormalizeLinear>{
	MULTISTAGEPIPELINE_FACTORY_TYPEDEFS(NormalizeLinear)
	typedef std::vector<double> Vector;

	static std::unique_ptr<MultistagePipe<Res(Arg)>>
	deserialize(std::istream& is){
		auto scaled = Factory<Scale<SparseVector(SparseVector)>>::deserialize(is);
		auto offset = Factory<Offset<SparseVector(SparseVector)>>::deserialize(is,std::move(scaled));
		return std::unique_ptr<MultistagePipe<Res(Arg)>>(new NormalizeLinear(std::move(offset)));
	}

	/**
	 * Creates a logistic regression pipeline with given scale coefficients and offset.
	 */
	std::unique_ptr<NormalizeLinear>
	operator()(std::vector<double>&& scale_coeffs, std::vector<double>&& offset_coeffs) const{
		assert(scale_coeffs.size()==offset_coeffs.size() && "Dimension mismatch between offset and scale!");

		Factory<Scale<SparseVector(SparseVector)>> f_scale;
		auto scale = f_scale(std::move(scale_coeffs),0); // num_inputs = 0 to disable checking

		Factory<Offset<SparseVector(SparseVector)>> f_offset;
		auto off = f_offset(std::move(scale),std::move(offset_coeffs));

		return std::unique_ptr<NormalizeLinear>(new NormalizeLinear(std::move(off)));
	}
};

MULTISTAGEPIPELINE_POST_FACTORY(NormalizeLinear)

/**
 * Use an SVMModel to aggregate data to a single decision value.
 * Input: std::vector<double>
 * Output: double
 *
 * Pipeline details:
 * 1. SVM	- must be supplied
 */
MULTISTAGEPIPELINE(BinarySVMAggregation,double,std::vector<double>)

template <>
struct Factory<BinarySVMAggregation>{
	MULTISTAGEPIPELINE_FACTORY_TYPEDEFS(BinarySVMAggregation)
	typedef std::vector<double> Vector;

	static std::unique_ptr<MultistagePipe<Res(Arg)>>
	deserialize(std::istream& is){
		auto svm = Factory<SVM<double(std::vector<double>)>>::deserialize(is);
		return std::unique_ptr<MultistagePipe<Res(Arg)>>(new BinarySVMAggregation(std::move(svm)));
	}

	/**
	 * Creates a logistic regression pipeline with given scale coefficients and offset.
	 */
	std::unique_ptr<BinarySVMAggregation>
	operator()(std::unique_ptr<SVMModel> svm) const{
		Factory<SVM<double(std::vector<double>)>> f;
		auto svmpipe = f(svm.release());

		return std::unique_ptr<BinarySVMAggregation>(new BinarySVMAggregation(std::move(svmpipe)));
	}
};

MULTISTAGEPIPELINE_POST_FACTORY(BinarySVMAggregation)

/*************************************************************************************************/

/**
 * Use an SVMModel to aggregate data to a single decision value.
 * Input: std::vector<double>
 * Output: double
 *
 * Pipeline details:
 * 1. SVM	- must be supplied
 */
MULTISTAGEPIPELINE(LinearAggregation,double,std::vector<double>)

template <>
struct Factory<LinearAggregation>{
	MULTISTAGEPIPELINE_FACTORY_TYPEDEFS(LinearAggregation)
	typedef std::vector<double> Vector;

	static std::unique_ptr<MultistagePipe<Res(Arg)>>
	deserialize(std::istream& is){
		auto scale = Factory<Scale<std::vector<double>(std::vector<double>)>>::deserialize(is);
		auto sum = Factory<Sum<double(std::vector<double>)>>::deserialize(is, std::move(scale));
		auto offset = Factory<Offset<double(double)>>::deserialize(is, std::move(sum));
		return std::unique_ptr<MultistagePipe<Res(Arg)>>(new LinearAggregation(std::move(offset)));
	}

	/**
	 * Creates a logistic regression pipeline with given scale coefficients and offset.
	 */
	std::unique_ptr<LinearAggregation>
	operator()(std::vector<double>&& coeffs, double offset) const{
		Factory<Scale<std::vector<double>(std::vector<double>)>> f_scale;
		auto scale = f_scale(std::move(coeffs));
		Factory<Sum<double(std::vector<double>)>> f_sum;
		auto sum = f_sum(std::move(scale));
		Factory<Offset<double(double)>> f_offset;
		auto off = f_offset(std::move(sum),offset);

		return std::unique_ptr<LinearAggregation>(new LinearAggregation(std::move(off)));
	}
};

MULTISTAGEPIPELINE_POST_FACTORY(LinearAggregation)

/*************************************************************************************************/

} // pipeline namespace

/*************************************************************************************************/

#endif /* PIPELINES_HPP_ */
