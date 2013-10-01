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
 * blocks.hpp
 *
 *      Author: Marc Claesen
 */

#ifndef BLOCKS_HPP_
#define BLOCKS_HPP_

/*************************************************************************************************/

#include "pipeline/core.hpp"
#include "SparseVector.hpp"
#include "Models.hpp"
#include <list>
#include <deque>
#include <cmath>
#include <cstdlib>
#include <sstream>

using ensemble::SVMModel;

/*************************************************************************************************/

namespace ensemble{
namespace pipeline{
namespace impl{

/*************************************************************************************************/

inline double Scale(double d, const std::vector<double>& coeff, size_t numoutputs){
	return d*coeff[0];
}

inline std::vector<double>&& Scale(std::vector<double>&& vec, const std::vector<double>& coeff, size_t numoutputs){
	if(coeff.size()==1){
		std::transform(vec.begin(),vec.begin()+numoutputs, vec.begin(),
				[&coeff](double x){ return x*coeff[0]; });
	}else{
		std::transform(vec.begin(),vec.begin()+numoutputs,
				coeff.begin(),vec.begin(),std::multiplies<double>());
	}
	if(numoutputs > 0) vec.resize(numoutputs);
	return std::move(vec);
}

/*************************************************************************************************/

} // ensemble::pipeline::impl namespace

/*************************************************************************************************/

/**
 * Used to scale the inputs, based on the scales used for construction.
 */
Derive_BB_CRTP(Scale) {
private:
	std::vector<double> coeff_;

protected:
	virtual void write_data(std::ostream& os) const override{
		os << coeff_.size() << std::endl;
		for(auto c: coeff_) os << c << " ";
		os << std::endl;
	}

public:
	Typedefs(Scale)
typedef std::tuple<std::vector<double>, size_t> CtorTuple;

	template<typename U = Internal, class = typename std::enable_if<!std::is_same<U, nullptr_t>::value>::type>
	Scale(std::unique_ptr<U> wrap, std::vector<double>& coeff, size_t size=0)
	:CRTPClass(std::move(wrap)),
	 coeff_(coeff)
	 {
		if(coeff_.size() > 1 && size) assert(coeff_.size() == size && "Sizes do not match!");
		if(coeff_.size() > 1) assert(coeff_.size() == BaseClass::internal()->num_outputs()
		&& "wrappee num_outputs and no. of coeff don't match!");
	}
	template<typename U = Internal, class = typename std::enable_if<std::is_same<U, nullptr_t>::value>::type>
	Scale(std::vector<double>& coeff)
	:CRTPClass(coeff.size()),
	 coeff_(coeff)
	{
		PipeBase::setOutputLen(coeff_.size());
	}
	template<typename U = Internal, class = typename std::enable_if<std::is_same<U, nullptr_t>::value>::type>
	Scale(std::vector<double>& coeff, size_t size)
	:CRTPClass(size),
	 coeff_(coeff)
	{
		if(size){ assert((coeff_.size()==1 || coeff_.size() == size) && "Sizes do not match!");
			PipeBase::setOutputLen(size);
		}else{
			if(coeff_.size()>1) PipeBase::setOutputLen(coeff_.size());
		}
	}
	template<typename U = Internal, class = typename std::enable_if<!std::is_same<U, nullptr_t>::value>::type>
	Scale(std::unique_ptr<U> wrap, std::vector<double>&& coeff)
	:CRTPClass(std::move(wrap),coeff.size()),
	 coeff_(std::move(coeff))
	 {
		assert(coeff.size() == BaseClass::internal()->num_outputs()
		&& "wrappee num_outputs and no. of coeff don't match!");
	}
	template<typename U = Internal, class = typename std::enable_if<std::is_same<U, nullptr_t>::value>::type>
	Scale(std::vector<double>&& coeff)
	:CRTPClass(coeff.size()),
	 coeff_(coeff)
	 {
		PipeBase::setOutputLen(coeff_.size());
	 }
	template<typename U = Internal, class = typename std::enable_if<std::is_same<U, nullptr_t>::value>::type>
	Scale(std::vector<double>&& coeff, size_t size)
	:CRTPClass(size),
	 coeff_(std::move(coeff))
	 {
		if(size){
			assert((coeff_.size()==1 || coeff_.size() == size) && "Sizes do not match!");
			PipeBase::setOutputLen(size);
		}else{
			if(coeff_.size()>1) PipeBase::setOutputLen(coeff_.size());
		}
	 }

	template<typename U = Internal, class = typename std::enable_if<!std::is_same<U, nullptr_t>::value>::type>
	Scale(std::unique_ptr<U> wrap, double coeff, size_t numinputs=0)
	:CRTPClass(std::move(wrap)),
	 coeff_(1,coeff)
	 {
		if(numinputs) assert(numinputs == PipeBase::num_outputs()
		&& "wrappee num_outputs and no. of coeff don't match!");
	 }
	template<typename U = Internal, class = typename std::enable_if<std::is_same<U, nullptr_t>::value>::type>
	Scale(double coeff, size_t numinputs)
	:CRTPClass(numinputs),
	 coeff_(1,coeff)
	 {}

	Scale(const DerivedClass& o)
		:CRTPClass(o),
		 coeff_(o.coeff_){}

	Scale(DerivedClass&& o)
	:CRTPClass(std::move(o)),
	 coeff_(std::move(o.coeff_))
	{}

	Ret process(Arg&& inputs) const override{
		return impl::Scale(std::move(inputs),coeff_,PipeBase::num_outputs());
	}
	virtual ~Scale(){}

	static CtorTuple deserialize(std::istream& is, size_t num_inputs, size_t num_outputs){

		if(!is) pipeline_error("Error reading Scale from input stream.");

		double d;
		size_t numcoeffs;
		is >> numcoeffs;
		std::vector<double> coeffs;
		coeffs.reserve(numcoeffs);

		// ignore newline
		is.ignore(1);

		std::string line;
		getline(is,line);
		std::istringstream ss(line);

		for(size_t i=0;i<numcoeffs;++i){
			if(!ss) pipeline_error("Error reading Scale from input stream.");
			ss >> d;
			std::back_inserter(coeffs) = d;
		}
		CtorTuple tuple=std::make_tuple(std::move(coeffs),num_inputs);
		return std::move(tuple);
	}
};

/*************************************************************************************************/

// OFFSET

/*************************************************************************************************/

namespace impl{

/*************************************************************************************************/

inline double Offset(double d, const std::vector<double>& offsets, size_t numoutputs){
	return d+offsets[0];
}

inline std::vector<double>&& Offset(std::vector<double>&& vec, const std::vector<double>& offsets, size_t numoutputs){
	if(offsets.size()==1){
		std::transform(vec.begin(),vec.begin()+numoutputs, vec.begin(),
				[&offsets](double x){ return x+offsets[0]; });
	}else{
		std::transform(vec.begin(),vec.begin()+numoutputs,
				offsets.begin(),vec.begin(),std::plus<double>());
	}
	if(numoutputs > 0) vec.resize(numoutputs);
	return std::move(vec);
}

/*************************************************************************************************/

} // impl namespace

/*************************************************************************************************/

/**
 * Used to scale the inputs, based on the scales used for construction.
 */
Derive_BB_CRTP(Offset) {
private:
	std::vector<double> offsets;

protected:
	virtual void write_data(std::ostream& os) const override{
		os << offsets.size() << std::endl;
		for(auto o: offsets) os << o << " ";
		os << std::endl;
	}

public:
	Typedefs(Offset)
typedef std::tuple<std::vector<double>, size_t> CtorTuple;

	template<typename U = Internal, class = typename std::enable_if<!std::is_same<U, nullptr_t>::value>::type>
	Offset(std::unique_ptr<U> wrap, std::vector<double>& offsets, size_t numinputs=0)
	:CRTPClass(std::move(wrap)),
	 offsets(offsets)
	 {
		if(numinputs > 0) assert(this->offsets.size() == BaseClass::internal()->num_outputs()
		&& "wrappee num_outputs and no. of offset don't match!");
		if(numinputs > 0) assert(numinputs == offsets.size() && "Specified number of inputs does not match with offsets!");
	}
	template<typename U = Internal, class = typename std::enable_if<std::is_same<U, nullptr_t>::value>::type>
	Offset(std::vector<double>& offset, size_t numinputs)
	:CRTPClass(numinputs),
	 offsets(offset)
	 {
		if(numinputs){
			assert((offsets.size()==1 || offsets.size() == numinputs) && "Sizes do not match!");
			PipeBase::setOutputLen(numinputs);
		}else{
			if(offsets.size()>1) PipeBase::setOutputLen(offsets.size());
		}
	 }
	template<typename U = Internal, class = typename std::enable_if<std::is_same<U, nullptr_t>::value>::type>
	Offset(std::vector<double>& offsets)
	:CRTPClass(offsets.size()),
	 offsets(offsets)
	 {
		PipeBase::setOutputLen(offsets.size());
	 }

	template<typename U = Internal, class = typename std::enable_if<!std::is_same<U, nullptr_t>::value>::type>
	Offset(std::unique_ptr<U> wrap, std::vector<double>&& offsets)
	:CRTPClass(std::move(wrap),offsets.size()),
	 offsets(std::move(offsets))
	 {
		assert(this->offsets.size() == BaseClass::internal()->num_outputs()
		&& "wrappee num_outputs and no. of offset don't match!");
	}
	template<typename U = Internal, class = typename std::enable_if<std::is_same<U, nullptr_t>::value>::type>
	Offset(std::vector<double>&& offsets)
	:CRTPClass(offsets.size()),
	 offsets(std::move(offsets))
	 {
		PipeBase::setOutputLen(offsets.size());
	 }
	template<typename U = Internal, class = typename std::enable_if<std::is_same<U, nullptr_t>::value>::type>
	Offset(std::vector<double>&& offset, size_t numinputs)
	:CRTPClass(numinputs),
	 offsets(std::move(offset))
	 {
		if(numinputs){
			assert((offsets.size()==1 || offsets.size() == numinputs) && "Sizes do not match!");
			PipeBase::setOutputLen(numinputs);
		}else{
			if(offsets.size()>1) PipeBase::setOutputLen(offsets.size());
		}
	 }

	template<typename U = Internal, class = typename std::enable_if<!std::is_same<U, nullptr_t>::value>::type>
	Offset(std::unique_ptr<U> wrap, double offset, size_t numinputs=0)
	:CRTPClass(std::move(wrap)),
	 offsets(1,offset)
	 {
		if(numinputs) assert(PipeBase::num_inputs()==numinputs);
	}
	template<typename U = Internal, class = typename std::enable_if<std::is_same<U, nullptr_t>::value>::type>
	Offset(double offset, size_t numinputs)
	:CRTPClass(numinputs),
	 offsets(1,offset)
	 {}

	Offset(const DerivedClass& o):CRTPClass(o),offsets(o.offsets){}
	Offset(DerivedClass&& o)
	:CRTPClass(std::move(o)),
	 offsets(std::move(o.offsets))
	{}

	Ret process(Arg&& inputs) const override{
		return impl::Offset(std::move(inputs),offsets,PipeBase::num_outputs());
	}

	virtual ~Offset() = default;

	static CtorTuple deserialize(std::istream& is, size_t num_inputs, size_t num_outputs){

		if(!is) pipeline_error("Error reading Scale from input stream.");

		double d;
		size_t numcoeffs;
		is >> numcoeffs;
		std::vector<double> offsets;
		offsets.reserve(numcoeffs);

		// ignore newline
		is.ignore(1);

		std::string line;
		getline(is,line);
		std::istringstream ss(line);

		for(size_t i=0;i<numcoeffs;++i){
			if(!ss) pipeline_error("Error reading Scale from input stream.");
			ss >> d;
			std::back_inserter(offsets) = d;
		}
		CtorTuple tuple=std::make_tuple(std::move(offsets),num_inputs);
		return std::move(tuple);
	}
};

/*************************************************************************************************/

namespace impl{

/*************************************************************************************************/

inline double Logistic(double d){
	return 1/(1+exp(-d));
}

inline float Logistic(float d){
	return 1/(1+exp(-d));
}

template <typename T>
inline std::vector<T>&& Logistic(std::vector<T>&& inputs){
	std::transform(inputs.begin(),inputs.end(),inputs.begin(),
			[](T x){ return T(1/(1+exp(-x))); }
	);
	return std::move(inputs);
}

template <typename T>
inline std::deque<T>&& Logistic(std::deque<T>&& inputs){
	std::transform(inputs.begin(),inputs.end(),inputs.begin(),
			[](T x){ return T(1/(1+exp(-x))); }
	);
	return std::move(inputs);
}

template <typename T>
inline std::list<T>&& Logistic(std::list<T>&& inputs){
	std::transform(inputs.begin(),inputs.end(),inputs.begin(),
			[](T x){ return T(1/(1+exp(-x))); }
	);
	return std::move(inputs);
}

/*************************************************************************************************/

} // impl namespace

/*************************************************************************************************/

/**
 * The logistic function:
 * F(t) = 1 / (1+exp(-t))
 *
 * If given a vector v, applies F elementwise.
 */
Derive_BB_CRTP(Logistic) {
protected:
	void write_data(std::ostream& os) const override{}

public:
	Typedefs(Logistic)
typedef std::tuple<size_t> CtorTuple;

	template<typename U = Internal, class = typename std::enable_if<!std::is_same<U, nullptr_t>::value>::type>
	Logistic(std::unique_ptr<U> wrap, size_t numinputs=0)
	:CRTPClass(std::move(wrap),numinputs){}

	template<typename U = Internal, class = typename std::enable_if<std::is_same<U, nullptr_t>::value>::type>
	Logistic(size_t numinputs=0)
	:CRTPClass(numinputs){}

	Logistic(const DerivedClass& o):CRTPClass(o){}
	Logistic(DerivedClass&& o)
	:CRTPClass(std::move(o)){}

	Ret process(Arg&& inputs) const override{
		return impl::Logistic(std::move(inputs));
	}

	~Logistic() = default;

	static CtorTuple deserialize(std::istream& is, size_t num_inputs, size_t num_outputs){
		return std::make_tuple(num_inputs);
	}
};

/*************************************************************************************************/

template<typename Ret, class Enable=void>
struct ThresholdScheme;

template <typename Ret>
struct ThresholdScheme<Ret, typename std::enable_if<std::is_compound<Ret>::value>::type>{
	std::vector<double> threshold;
	Ret above;
	Ret below;
	typedef typename Ret::value_type value_type;
};

template <typename Ret>
struct ThresholdScheme<Ret, typename std::enable_if<!std::is_compound<Ret>::value>::type>{
	double threshold;
	Ret above;
	Ret below;
	typedef Ret value_type;
};

} // pipeline namespace

/*************************************************************************************************/

namespace{

/*************************************************************************************************/

template <typename Ret>
void write_data_threshold(std::ostream& os, const pipeline::ThresholdScheme<Ret>& scheme,
		typename std::enable_if<std::is_compound<Ret>::value>::type* = 0){
	os << scheme.threshold.size();
	os << std::endl;
	for(auto d: scheme.threshold) os << d << " ";
	os << std::endl;
	for(auto d: scheme.above) os << d << " ";
	os << std::endl;
	for(auto d: scheme.below) os << d << " ";
	os << std::endl;
}

template <typename Ret>
void write_data_threshold(std::ostream& os, const pipeline::ThresholdScheme<Ret>& scheme,
		typename std::enable_if<!std::is_compound<Ret>::value>::type* = 0){
	os << "1" << std::endl;	// scheme size
	os << scheme.threshold;
	os << std::endl;
	os << scheme.above;
	os << std::endl;
	os << scheme.below;
	os << std::endl;
}

template <typename T>
pipeline::ThresholdScheme<T> read_scheme(size_t schemesize,
		std::istream& is, typename std::enable_if<!std::is_compound<T>::value>::type* = 0){

	typedef pipeline::ThresholdScheme<T> Scheme;
	Scheme scheme;

	// a scheme with a non-compound return type simply contains 3 numbers
	// which makes it trivial to read

	is >> scheme.threshold;
	is >> scheme.above;
	is >> scheme.below;

	return std::move(scheme);
}

template <typename T>
pipeline::ThresholdScheme<T> read_scheme(size_t schemesize,
		std::istream& is, typename std::enable_if<std::is_compound<T>::value>::type* = 0){
	typedef pipeline::ThresholdScheme<T> Scheme;

	Scheme scheme;
	scheme.threshold.reserve(schemesize);

	std::istringstream ss;
	std::string line;

	getline(is,line);
	ss.str(line);

	double d;
	if(ss){
		for(size_t i=0;i<schemesize;++i){
			if(!ss) pipeline_error("Error reading Offset from input stream.");
			ss >> d;
			std::back_inserter(scheme.threshold) = d;
		}

		getline(is,line);
		ss.clear();
		ss.str(line);
		for(size_t i=0;i<schemesize;++i){
			if(!ss) pipeline_error("Error reading Offset from input stream.");
			typename Scheme::value_type ret;
			ss >> ret;
			std::back_inserter(scheme.above) = ret;
		}

		getline(is,line);
		ss.clear();
		ss.str(line);
		for(size_t i=0;i<schemesize;++i){
			typename Scheme::value_type ret;
			if(!ss) pipeline_error("Error reading Offset from input stream.");
			ss >> ret;
			std::back_inserter(scheme.below) = ret;
		}
	}
	return std::move(scheme);
}

} // anonymous namespace

/*************************************************************************************************/

namespace pipeline{

/*************************************************************************************************/

// todo document inner demons

Derive_BB_CRTP(Threshold) {
public:
	typedef ThresholdScheme<Ret> Scheme;

private:
	Scheme scheme;

protected:
	virtual void write_data(std::ostream& os) const override{
		write_data_threshold(os,scheme);
	}

public:
	Typedefs(Threshold)

	typedef std::tuple<Scheme, size_t> CtorTuple;

	template<typename U = Internal, class = typename std::enable_if<!std::is_same<U, nullptr_t>::value>::type>
	Threshold(std::unique_ptr<U> wrap, Scheme scheme, size_t numinputs=0)
	:CRTPClass(std::move(wrap),numinputs),
	 scheme(scheme)
	 {}
	template<typename U = Internal, class = typename std::enable_if<std::is_same<U, nullptr_t>::value>::type>
	Threshold(Scheme scheme, size_t numinputs=0)
	:CRTPClass(numinputs),
	 scheme(scheme)
	 {}

	/*
	  Simple constructors when Ret is not compound, e.g. fundamental.
	*/

	template<typename U = Internal, typename V = Ret,	// helper types
			class = typename std::enable_if<
				!std::is_same<U, nullptr_t>::value &&	// Internal is not nullptr_t
				!std::is_compound<V>::value				// Ret is not compound
				>::type
			>
	Threshold(std::unique_ptr<U> wrap, double threshold, Ret above, Ret below)
	:CRTPClass(std::move(wrap),1),
	 scheme{threshold,above,below}
	{}
	template<typename U = Internal, typename V = Ret,	// helper types
			class = typename std::enable_if<
				std::is_same<U, nullptr_t>::value &&	// Internal == nullptr_t
				!std::is_compound<V>::value				// Ret is not compound
				>::type
			>
	Threshold(double threshold, Ret above, Ret below)
	:CRTPClass(1),
	 scheme{threshold,above,below}
	{}

	/*
	  Simple constructors when Ret is compound, e.g. a container.
	*/

	template<typename U = Internal, typename V = Ret,	// helper types
			class = typename std::enable_if<
				!std::is_same<U, nullptr_t>::value 	&&	// Internal is not nullptr_t
				std::is_compound<V>::value			&&	// Ret is compound
				!std::is_same<V,std::string>::value		// Ret is not std::string
				>::type
			>
	Threshold(std::unique_ptr<U> wrap, double threshold, typename V::value_type above, typename V::value_type below, size_t numinputs=0)
	:CRTPClass(std::move(wrap),numinputs),
	 scheme{{threshold},{above},{below}}
	{}
	template<typename U = Internal, typename V = Ret,	// helper types
			class = typename std::enable_if<
				std::is_same<U, nullptr_t>::value 	&&	// Internal == nullptr_t
				std::is_compound<V>::value		  	&&	// Ret is compound
				!std::is_same<V,std::string>::value  	// Ret is not std::string
				>::type
			>
	Threshold(double threshold, typename V::value_type above, typename V::value_type below, size_t numinputs=0)
	:CRTPClass(numinputs),
	 scheme{{threshold},{above},{below}}
	{}

	Threshold(const DerivedClass& o):CRTPClass(o),scheme(o.scheme){}
	Threshold(DerivedClass&& o)
	:CRTPClass(std::move(o)),
	scheme{std::move(o.scheme.threshold),std::move(o.scheme.above),std::move(o.scheme.below)}
	{}

	Ret process(Arg&& inputs) const override;
	virtual ~Threshold() = default;

	const Scheme& getScheme() const{ return scheme; }

	static CtorTuple deserialize(std::istream& is, size_t num_inputs, size_t num_outputs){
		std::string line;
		if(!is) pipeline_error("Error reading Offset from input stream.");

		getline(is,line);
		std::istringstream ss(line);

		unsigned schemesize=0;
		ss >> schemesize;

		Scheme scheme=read_scheme<Ret>(schemesize,is);
		CtorTuple tuple=std::make_tuple(std::move(scheme),num_inputs);
		return std::move(tuple);
	}
};

/*************************************************************************************************/

namespace{

/*************************************************************************************************/

template <typename Res, typename Iterator>
inline Res Threshold_impl(Res cont, Iterator begin, Iterator end, const ThresholdScheme<Res>& scheme){
	if(scheme.threshold.size()==1){
		std::transform(begin,end,std::back_inserter(cont),
				[&](typename Iterator::value_type x){
			return x > typename Iterator::value_type(scheme.threshold[0]) ? *scheme.above.begin() : *scheme.below.begin(); }
		);
	}else{
		std::vector<double>::const_iterator It=scheme.threshold.begin();
		typename Res::const_iterator Ia=scheme.above.begin(), Ib=scheme.below.begin();
		auto Ic = std::back_inserter(cont);
		for(;begin!=end;++begin,++It,++Ia,++Ib){
			*Ic=(*begin) > typename Iterator::value_type(*It) ? *Ia : *Ib;
		}
	}
	return std::move(cont);
}

template <typename Res, typename Iterator>
inline Res Threshold_impl(Iterator begin, Iterator end,
		const ThresholdScheme<Res>& scheme){
	Res cont;
	return Threshold_impl(std::move(cont),begin,end,scheme);
}

template <typename Res, typename Val>
inline std::vector<Res> Threshold_impl(typename std::vector<Val>::const_iterator begin,
		typename std::vector<Val>::const_iterator end,
		const ThresholdScheme<std::vector<Res>>& scheme){
	std::vector<Res> cont;
	cont.reserve(std::distance(begin,end));
	return Threshold_impl(std::move(cont),begin,end,scheme);
}

} // anonymous namespace

/*************************************************************************************************/

namespace impl{

/*************************************************************************************************/

template <
	typename Arg,
	typename Res
	>
typename std::enable_if<!std::is_compound<Res>::value, Res>::type
Threshold_fun(Arg arg, const ThresholdScheme<Res>& scheme){
	return (arg > Arg(scheme.threshold)) ? scheme.above : scheme.below;
}

template <
	typename Arg,
	typename Res
	>
typename std::enable_if<std::is_compound<Res>::value, Res>::type
Threshold_fun(Arg arg, const ThresholdScheme<Res>& scheme){
	return Threshold_impl(arg.begin(),arg.end(),scheme);
}

} // impl namespace

template <typename Ret, typename Arg, typename Internal>
Ret Threshold<Ret(Arg),Internal>::process(Arg&& inputs) const{
	return impl::Threshold_fun(std::move(inputs),scheme);
}

/*************************************************************************************************/

namespace impl{

/*************************************************************************************************/

template <typename Iterator>
inline typename Iterator::value_type Average(Iterator begin, Iterator end, double divisor){
	double d = divisor ? divisor : std::distance(begin,end);
	return typename Iterator::value_type(std::accumulate(begin,end,0.0)/d);
}

} // impl namespace

Derive_BB_CRTP(Average) {
private:
	double divisor=0.0;
public:
	Typedefs(Average)
	typedef std::tuple<double,size_t> CtorTuple;

protected:
	virtual void write_data(std::ostream& os) const override{ os << divisor << std::endl;	}

public:
	template<typename U = Internal, class = typename std::enable_if<!std::is_same<U, nullptr_t>::value>::type>
	Average(std::unique_ptr<U> wrap, double divisor=0.0, size_t numinputs=0)
	:CRTPClass(std::move(wrap),numinputs),
	 divisor(divisor)
	{
		BaseClass::setOutputLen(1);
	}
	template<typename U = Internal, class = typename std::enable_if<std::is_same<U, nullptr_t>::value>::type>
	Average(double divisor=0.0, size_t numinputs=0)
	:CRTPClass(numinputs),
	 divisor(divisor)
	 {
		BaseClass::setOutputLen(1);
	 }

	Average(const DerivedClass& o)
	:CRTPClass(o),
	 divisor(o.divisor)
	{}
	Average(DerivedClass&& o)
	:CRTPClass(std::move(o)),
	 divisor(o.divisor)
	{}

	Ret process(Arg&& inputs) const override{
		return impl::Average(inputs.begin(),inputs.end(),divisor);
	}

	virtual ~Average() = default;

	static CtorTuple deserialize(std::istream& is, size_t num_inputs, size_t num_outputs){
		double d;
		std::string line;
		getline(is,line);
		std::istringstream ss(line);
		ss >> d;
		return std::make_tuple(d,num_inputs);
	}
};

/*************************************************************************************************/

namespace impl{

/*************************************************************************************************/

// fixme: median broken for std::list

template <typename Arg, typename Tag>
inline typename Arg::value_type median_impl(Arg&& inputs, Tag){
	typename Arg::iterator I=inputs.begin();
	for(unsigned i=0;i<inputs.size()/2;++i)
		++I;
	std::nth_element(inputs.begin(), I, inputs.end());
	I=inputs.begin();
	for(unsigned i=0;i<inputs.size()/2;++i)
		++I;

	return *I;
}

template <typename Arg>
inline typename Arg::value_type median_impl(Arg&& inputs, std::random_access_iterator_tag){
	std::nth_element(inputs.begin(), inputs.begin()+inputs.size()/2, inputs.end());
	typename Arg::iterator it=inputs.begin()+inputs.size()/2;
	return *it;
}

template <typename Ret>
struct median{
	template<typename Arg>
	Ret operator()(Arg&& inputs) const{
	    typedef typename std::iterator_traits<typename Arg::iterator>::iterator_category category;
	    return Ret(median_impl(std::move(inputs),category()));
	}
};

/*************************************************************************************************/

} // impl namespace

/*************************************************************************************************/

Derive_BB_CRTP(Median) {
public:
	Typedefs(Median)
typedef std::tuple<size_t> CtorTuple;

protected:
	virtual void write_data(std::ostream& os) const override{}

public:
	template<typename U = Internal, class = typename std::enable_if<!std::is_same<U, nullptr_t>::value>::type>
	Median(std::unique_ptr<U> wrap, size_t numinputs=0)
	:CRTPClass(std::move(wrap),numinputs)
	{
		BaseClass::setOutputLen(1);
	}
	template<typename U = Internal, class = typename std::enable_if<std::is_same<U, nullptr_t>::value>::type>
	Median(size_t numinputs=0)
	:CRTPClass(numinputs)
	 {
		BaseClass::setOutputLen(1);
	 }

	Median(const DerivedClass& o):CRTPClass(o){}
	Median(DerivedClass&& o):CRTPClass(std::move(o)){}

	Ret process(Arg&& inputs) const override{
		impl::median<Ret> m;
		return m(std::move(inputs));
	}

	virtual ~Median() = default;

	static CtorTuple deserialize(std::istream& is, size_t num_inputs, size_t num_outputs){
		return std::make_tuple(num_inputs);
	}
};

/*************************************************************************************************/

namespace impl{

template <typename Iterator>
inline typename Iterator::value_type Sum(Iterator begin, Iterator end){
	return std::accumulate(begin,end,typename Iterator::value_type(0));
}

} // impl namespace

Derive_BB_CRTP(Sum) {
public:
	Typedefs(Sum)
typedef std::tuple<size_t> CtorTuple;

protected:
	virtual void write_data(std::ostream& os) const override{}

public:
	template<typename U = Internal, class = typename std::enable_if<!std::is_same<U, nullptr_t>::value>::type>
	Sum(std::unique_ptr<U> wrap, size_t numinputs=0)
	:CRTPClass(std::move(wrap),numinputs)
	{
		BaseClass::setOutputLen(1);
		if(numinputs) assert(BaseClass::num_inputs() == numinputs);
	}
	template<typename U = Internal, class = typename std::enable_if<std::is_same<U, nullptr_t>::value>::type>
	Sum(size_t numinputs=0)
	:CRTPClass(numinputs)
	 {
		BaseClass::setOutputLen(1);
	 }

	Sum(const DerivedClass& o)
	:CRTPClass(o)
	{}
	Sum(DerivedClass&& o)
	:CRTPClass(std::move(o))
	{}

	Ret process(Arg&& inputs) const override{
		return impl::Sum(inputs.begin(),inputs.end());
	}

	virtual ~Sum() = default;

	static CtorTuple deserialize(std::istream& is, size_t num_inputs, size_t num_outputs){
		return std::make_tuple(num_inputs);
	}
};

/*************************************************************************************************/

namespace impl{

template<typename T>
struct SVM_predict;

template<>
struct SVM_predict<std::vector<double>>{
	double operator()(const SVMModel& model, std::vector<double>&& arg) const{
		std::vector<double> decvals = model.decision_value(arg);
		return decvals[0];
	}
};

template<>
struct SVM_predict<SparseVector>{
	double operator()(const SVMModel& model, SparseVector&& arg) const{
		std::vector<double> decvals = model.decision_value(arg);
		return decvals[0];
	}
};

//template<typename T>
//double SVM_impl(const SVMModel& model, T&& arg);
//
//template<>
//double SVM_impl(const SVMModel& model, std::vector<double>&& arg){
//	std::vector<double> decvals = model.decision_value(arg);
//	return decvals[0];
//}
//
//template<>
//double SVM_impl(const SVMModel& model, SparseVector&& arg){
//	std::vector<double> decvals = model.decision_value(arg);
//	return decvals[0];
//}

} // impl namespace

template<typename T,typename N=nullptr_t> class SVM;
template<typename T, typename N>
struct ToStr<pipeline::SVM<T,N>>{
	static string get(){
		string type{"SVM"}, t("pipeline::");
		t.append(type);
		return t;
	}
};
template <typename Arg, typename Internal>
class SVM<double(Arg),Internal> final : public BB_CRTP<SVM<double(Arg),Internal>>{
public:
	typedef BasicBlock<double(Arg),Internal> BaseClass;
	typedef BB_CRTP<SVM<double(Arg),Internal> > CRTPClass;
	typedef typename BaseClass::Input Input;
	typedef SVM<double(Arg),Internal> DerivedClass;
	typedef typename CRTPClass::PipeBase PipeBase;
	typedef double result_type;
	typedef Arg argument_type;
	typedef std::tuple<SVMModel*,size_t> CtorTuple;

private:
	std::unique_ptr<SVMModel> svm;

	template<typename U = Internal, class = typename std::enable_if<!std::is_same<U, nullptr_t>::value>::type>
	SVM(std::unique_ptr<U> wrap, SVMModel* svm, size_t numinputs=0)
	:CRTPClass(std::move(wrap)),
	 svm(svm)
	{
		if(numinputs && BaseClass::num_outputs()) assert(numinputs==BaseClass::num_outputs());
		BaseClass::setOutputLen(1);
	}
	template<typename U = Internal, class = typename std::enable_if<std::is_same<U, nullptr_t>::value>::type>
	SVM(SVMModel* svm, size_t numinputs=0)
	:CRTPClass(numinputs),
	 svm(svm)
	 {
		BaseClass::setOutputLen(1);
	 }

protected:
	virtual void write_data(std::ostream& os) const override{
		os << *svm;
	}

public:
	template<typename U = Internal, class = typename std::enable_if<!std::is_same<U, nullptr_t>::value>::type>
	SVM(std::unique_ptr<U> wrap, std::unique_ptr<SVMModel> svm, size_t numinputs=0)
	:CRTPClass(std::move(wrap)),
	 svm(std::move(svm))
	{
		if(numinputs && BaseClass::num_outputs()) assert(numinputs==BaseClass::num_outputs());
		BaseClass::setOutputLen(1);
	}
	template<typename U = Internal, class = typename std::enable_if<std::is_same<U, nullptr_t>::value>::type>
	SVM(std::unique_ptr<SVMModel> svm, size_t numinputs)
	:CRTPClass(numinputs),
	 svm(std::move(svm))
	 {
		BaseClass::setOutputLen(1);
	 }

	SVM(const DerivedClass& o):CRTPClass(o),svm(new SVMModel(*o.svm)){}
	SVM(DerivedClass&& o):CRTPClass(std::move(o)),svm(std::move(o.svm)){}

	double process(Arg&& inputs) const override{
		impl::SVM_predict<Arg> p;
		return p(*svm,std::move(inputs));
	}

	virtual ~SVM() = default;

	static CtorTuple deserialize(std::istream& is, size_t num_inputs, size_t num_outputs){
		auto ptr = ensemble::BinaryModel::deserialize(is);
		SVMModel *model = dynamic_cast<SVMModel*>(ptr.release());
		assert(model && "Deserialized model is not an SVMModel as expected!");
		return std::make_tuple(model,num_inputs);
	}

	friend struct Factory<SVM<double(Arg)>>;
};

/*************************************************************************************************/

} // ensemble::pipeline namespace
} // ensemble namespace

/*************************************************************************************************/

#endif /* BLOCKS_HPP_ */
