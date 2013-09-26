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
 * core.hpp
 *
 *      Author: Marc Claesen
 */

#ifndef CORE_HPP_
#define CORE_HPP_

/*************************************************************************************************/

#include <iostream>
#include <vector>
#include <memory>
#include <numeric>
#include <functional>
#include <cassert>
#include <algorithm>
#include <tuple>
#include <type_traits>
#include <sstream>
#include "Type2str.hpp"

/*************************************************************************************************/

// fixme: modify serialization for pipelines and supply automatic deserialization for tested cases
// fixme: get rid of MultistagePipe, it's the same as any other pipe.

namespace{

__attribute__((noreturn)) void pipeline_error(std::string error){
	std::cout << error << std::endl;
	exit(EXIT_FAILURE);
}

__attribute__((noreturn)) void pipeline_error(){
	pipeline_error("Pipeline is in an illegal state!");
}

template <typename T>
bool check_size(const T& t, size_t size){
	return t.size()==size;
}
template<>
bool check_size(const double& t, size_t size){
	return size==1;
}
template<>
bool check_size(const float& t, size_t size){
	return size==1;
}
template<>
bool check_size(const bool& t, size_t size){
	return size==1;
}
template<>
bool check_size(const int& t, size_t size){
	return size==1;
}
template<>
bool check_size(const unsigned& t, size_t size){
	return size==1;
}

} // anonymous namespace

/*************************************************************************************************/

namespace pipeline{

template <typename T>
struct Factory{
	template <typename U, typename... Args>
	static U deserialize(Args... args){}
};

template <typename T> class Pipeline;

typedef decltype(nullptr) nullptr_t;

} // forward declarations in pipeline namespace

/*************************************************************************************************/

namespace{

using namespace pipeline;

// expanding tuple into multiple args
// http://stackoverflow.com/questions/10766112/c11-i-can-go-from-multiple-args-to-tuple-but-can-i-go-from-tuple-to-multiple
template <template <class, class=nullptr_t> class Derived, class Res, class Arg, typename Tuple, bool Done, int Total, int... N>
struct construct_impl{
	static Pipeline<Res(Arg)>*
	call(Factory<Derived<Res(Arg)>> f, Tuple && t){
		return construct_impl<Derived,Res,Arg, Tuple, Total == 1 + sizeof...(N),
				Total, N..., sizeof...(N)>::call(f, std::forward<Tuple>(t));
	}

	template <typename Internal>
	static Pipeline<Res(typename Internal::Input)>*
	call(Factory<Derived<Res(Arg)>> f, std::unique_ptr<Internal> internal, Tuple && t){
		return construct_impl<Derived,Res,Arg, Tuple, Total == 1 + sizeof...(N),
				Total, N..., sizeof...(N)>::call(f, std::move(internal), std::forward<Tuple>(t));
	}
};

template <template <class, class=nullptr_t> class Derived, class Res, class Arg, typename Tuple, int Total, int... N>
struct construct_impl<Derived,Res,Arg, Tuple, true, Total, N...>{
	static Pipeline<Res(Arg)>*
	call(Factory<Derived<Res(Arg)>> f, Tuple && t){
		auto p = f(std::get<N>(std::forward<Tuple>(t))...);
		Pipeline<Res(Arg)> *pipe=static_cast<Pipeline<Res(Arg)>*>(p.get());
		p.release();
		return pipe;
	}

	template <typename Internal>
	static Pipeline<Res(typename Internal::Input)>*
	call(Factory<Derived<Res(Arg)>> f, std::unique_ptr<Internal> internal, Tuple && t){
		auto p = f(std::move(internal),std::get<N>(std::forward<Tuple>(t))...);
		Pipeline<Res(typename Internal::Input)> *pipe=
				static_cast<Pipeline<Res(typename Internal::Input)>*>(p.get());
		p.release();
		return pipe;
	}
};

} // anonymous namespace

/*************************************************************************************************/

namespace pipeline{

// user invokes this
// http://stackoverflow.com/questions/10766112/c11-i-can-go-from-multiple-args-to-tuple-but-can-i-go-from-tuple-to-multiple
template <template <class, class=nullptr_t> class Derived, class Res, class Arg, typename Tuple>
Pipeline<Res(Arg)>* construct_pipe(Factory<Derived<Res(Arg)>> f, Tuple && t){
	typedef typename std::decay<Tuple>::type ttype;
	return construct_impl<Derived,Res,Arg,Tuple, 0 == std::tuple_size<ttype>::value,
			std::tuple_size<ttype>::value>::call(f, std::forward<Tuple>(t));
}
template <template <class, class=nullptr_t> class Derived, class Res, class Arg, class Internal, typename Tuple>
Pipeline<Res(typename Internal::Input)>* construct_pipe(Factory<Derived<Res(Arg)>> f, std::unique_ptr<Internal> internal, Tuple && t){
	typedef typename std::decay<Tuple>::type ttype;
	return construct_impl<Derived,Res,Arg,Tuple, 0 == std::tuple_size<ttype>::value,
			std::tuple_size<ttype>::value>::call(f, std::move(internal), std::forward<Tuple>(t));
}

template <typename T>
constexpr bool Is_nullptr(){ return false; }

template <>
constexpr bool Is_nullptr<nullptr_t>(){ return true; }

template<typename T>
struct identity{ typedef T type; };

/*************************************************************************************************/

// we need an abstract pipe because the base class of Internal is otherwise impossible to determine
class AbstractPipe{
public:
	virtual ~AbstractPipe(){}
};

template <
	typename Res, // Resulting type of functor
	typename Arg  // Argument type of functor
	>
class Pipeline<Res(Arg)> : public AbstractPipe{
private:
	size_t input_len;
	size_t output_len;

protected:
	Pipeline(size_t input_len, size_t output_len)
	:input_len(input_len),
	 output_len(output_len)
	{}

	Pipeline(size_t input_len)
	:input_len(input_len),
	 output_len(input_len)
	{}

	Pipeline(const Pipeline<Res(Arg)>& o)
	:input_len(o.input_len),
	 output_len(o.output_len)
	{}

	Pipeline(Pipeline<Res(Arg)>&& o)
	:input_len(o.input_len),
	 output_len(o.output_len)
	{}

	Pipeline() = delete;

	inline void setOutputLen(size_t len){ output_len=len; }

public:
	typedef Res result_type;
	typedef Arg argument_type;

	virtual Res operator()(Arg&& arg) const=0;
	inline Res operator()(const Arg& arg) const{ // fixme
		return operator()(Arg(arg));
	}
	inline Res operator()(Arg& arg) const{	// fixme
		Arg tmp(arg);
		return operator()(std::move(tmp));
	}

	virtual std::unique_ptr<Pipeline<Res(Arg)>> clone() const=0;

	/**
	 * Serializes the pipeline to a text representation.
	 */
	virtual void serialize(std::ostream& os) const=0;

	friend std::ostream &operator<<(std::ostream &os, const Pipeline<Res(Arg)> &v){
		v.serialize(os);
		return os;
	}

	virtual ~Pipeline(){};

	virtual size_t num_inputs() const{ return input_len; }
	inline size_t num_outputs() const{ return output_len; }

	// used by derived classes to denote the number of inputs at the start of the pipe
	virtual size_t internal_num_inputs() const{ return num_inputs(); }
};

/*************************************************************************************************/

/**
 * Basic element of Pipeline scheme (implemented via decorators).
 */
template <typename T, typename N=nullptr_t>
class BasicBlock;

template <
	typename Res, 		// resulting type of functor
	typename Arg, 		// argument type of functor
	typename Internal	// type of internal basicblock
	>
class BasicBlock<Res(Arg),Internal> : public Pipeline<Res(typename Internal::Input)>{
public:
	typedef Arg BasicBlockInput;
	typedef typename Internal::Input Input;
	typedef Res Result;
	typedef Pipeline<Res(typename Internal::Input)> PipeBase;

private:
	const std::string name;
	const std::unique_ptr<Internal> wrappee;

protected:
	BasicBlock(std::string name, std::unique_ptr<Internal> wrap, size_t numinputs=0)
	:PipeBase(wrap->num_outputs()),
	 name(name),
	 wrappee(std::move(wrap))
	{
		assert(wrappee && "Wrapped BasicBlock pointer may not be nullptr!");
		if(numinputs > 0) assert(numinputs==PipeBase::num_inputs() &&
				"Specified input length does not match wrappee output length!");
	}
	BasicBlock(const BasicBlock<Res(Arg),Internal>& o)
	:PipeBase(o),
	 name(o.name),
	 wrappee(o.wrappee->Internal::derived_clone())
	{}

	BasicBlock(BasicBlock<Res(Arg),Internal>&& o)
	:PipeBase(o),
	 name(std::move(o.name)),
	 wrappee(std::move(o.wrappee))
	{};

public:
	typedef std::function<Result(BasicBlockInput&&)> FnSig;
	typedef std::function<Result(Input&&)> GlobalFnSig;

	virtual Result operator()(Input&& inputs) const override{
		// no virtual lookup for wrappee
		typedef typename Internal::Result ThisArg;
		ThisArg res(wrappee->Internal::operator()(std::move(inputs)));
		if(PipeBase::num_inputs()) assert(check_size<ThisArg>(res,PipeBase::num_inputs()) && "Unexpected number of inputs");
		return process(std::move(res));
	}

	virtual Result process(BasicBlockInput&& inputs) const=0;

	virtual void serialize(std::ostream& os) const override{
		wrappee->serialize(os);
		os << name << "<" << ToStr<Res>::get() << "(" << ToStr<Arg>::get() << ")>";
		os << std::endl << PipeBase::num_inputs() << " " << PipeBase::num_outputs();
		os << std::endl;
	}

	const Internal* internal() const{ return wrappee.get(); }

	virtual size_t internal_num_inputs() const override{
		return wrappee->internal_num_inputs();
	}

	virtual ~BasicBlock(){}
};

template <
	typename Res, 		// resulting type of functor
	typename Arg 		// argument type of functor
	>
class BasicBlock<Res(Arg),nullptr_t> : public Pipeline<Res(Arg)>{
public:
	typedef Arg BasicBlockInput;
	typedef Arg Input;
	typedef Res Result;
	typedef Pipeline<Res(Arg)> PipeBase;

protected:
	const std::string name;

	BasicBlock(std::string name, size_t inputs_len=0)
	:PipeBase(inputs_len),
	 name(name)
	{}

	BasicBlock(const BasicBlock<Res(Arg),nullptr_t> &o)
	:PipeBase(o),
	 name(o.name)
	{}

	BasicBlock(BasicBlock<Res(Arg),nullptr_t>&& o)
	:PipeBase(o),
	 name(std::move(o.name))
	{}

public:
	typedef std::function<Result(BasicBlockInput&&)> FnSig;
	typedef std::function<Result(Input&&)> GlobalFnSig;

	virtual Result operator()(Input&& inputs) const override{
		if(PipeBase::num_inputs()) assert(check_size<Input>(inputs,PipeBase::num_inputs()) && "Unexpected number of inputs");
		return process(std::move(inputs));
	}

	virtual Result process(BasicBlockInput&& inputs) const=0;

	virtual void serialize(std::ostream& os) const override{
		os << name << "<" << ToStr<Res>::get() << "(" << ToStr<Arg>::get() << ")>";
		os << std::endl << PipeBase::num_inputs() << " " << PipeBase::num_outputs();
		os << std::endl;
	}

	virtual ~BasicBlock(){}
};

/*************************************************************************************************/

namespace {

template <typename T>
struct BB_CRTP_helper;

template <class Res, class Arg, class Internal, template <class, class> class Derived>
struct BB_CRTP_helper<Derived<Res(Arg),Internal>>{
	Res operator()(const Derived<Res(Arg),Internal> *bb, typename Derived<Res(Arg),Internal>::PipeBase::argument_type inputs){
		typedef typename Internal::Result ThisArg;
		ThisArg res(bb->internal()->Internal::operator()(std::move(inputs)));
		if(bb->num_inputs()) assert(check_size<ThisArg>(res,bb->num_inputs())
				&& "Unexpected number of inputs");
		return bb->Derived<Res(Arg),Internal>::process(std::move(res));
	}
};

template <class Res, class Arg, template <class, class> class Derived>
struct BB_CRTP_helper<Derived<Res(Arg),nullptr_t>>{
	Res operator()(const Derived<Res(Arg),nullptr_t> *bb, Arg inputs){
		if(bb->num_inputs()) assert(check_size<Arg>(inputs,bb->num_inputs())
				&& "Unexpected number of inputs");
		return bb->Derived<Res(Arg),nullptr_t>::process(std::move(inputs));
	}
};

} // anonymous namespace

/*************************************************************************************************/

template <typename T, class Enable=void>
class BB_CRTP;

/**
 * CRTP to do the wiring of cloning, serialization and operator() for us.
 */
template <	typename Res,							// functor result type
			typename Arg,							// functor argument type
			template <class, class> class Derived,	// the derived class that was used
			typename Internal						// wrapped pipeline
		 >
class BB_CRTP<
	Derived<Res(Arg),Internal>,
	typename std::enable_if<	// block instantiations that do not meet the following criteria:
			std::is_same<Internal,nullptr_t>::value ||				// internal is either a nullptr
			std::is_base_of<AbstractPipe,Internal>::value>::type	// or a pipeline
	> : public BasicBlock<Res(Arg),Internal> {
public:
	typedef BasicBlock<Res(Arg),Internal> BaseClass;
	typedef Derived<Res(Arg),Internal> DerivedClass;
	typedef typename BaseClass::FnSig FnSig;
	typedef typename BaseClass::GlobalFnSig GlobalFnSig;

	typedef typename BaseClass::PipeBase PipeBase;
	typedef typename PipeBase::argument_type argument_type;
	typedef typename PipeBase::result_type result_type;

protected:
	BB_CRTP(const BB_CRTP<DerivedClass> &o)
	:BaseClass(o)
	{}

	template<typename U = Internal, class = typename std::enable_if<!std::is_same<U, nullptr_t>::value>::type>
	BB_CRTP(std::unique_ptr<Internal> wrappee, int inputs_len=0)
	:BaseClass(ToStr<Derived<Res(Arg),Internal>>::get(),std::move(wrappee),inputs_len)
	 {}

	template<typename U = Internal, class = typename std::enable_if<std::is_same<U, nullptr_t>::value>::type>
	BB_CRTP(int inputs_len=0)
	:BaseClass(ToStr<Derived<Res(Arg),Internal>>::get(),inputs_len)
	 {}

	BB_CRTP(BB_CRTP<DerivedClass>&& o)
	:BaseClass(std::move(o))
	{}

	BB_CRTP() = delete;

protected:
	virtual void write_data(std::ostream& os) const=0;

public:
	Res operator()(argument_type&& inputs) const override final{
		BB_CRTP_helper<Derived<Res(Arg),Internal>> h;
		return h(static_cast<const Derived<Res(Arg),Internal>*>(this),inputs);
	}

	std::unique_ptr<PipeBase> clone() const override final{
		return std::unique_ptr<PipeBase>(
			static_cast<PipeBase*>(new DerivedClass(static_cast<DerivedClass const&>(*this)))
		);
	}

	std::unique_ptr<DerivedClass> derived_clone() const{
		return std::unique_ptr<DerivedClass>(new DerivedClass(static_cast<DerivedClass const&>(*this)));
	}

	void serialize(std::ostream& os) const override final{
		BaseClass::serialize(os);
		write_data(os);
	}

	virtual ~BB_CRTP(){}

	template<typename U = Internal, class = typename std::enable_if<!std::is_same<U, nullptr_t>::value>::type>
	static std::unique_ptr<DerivedClass> deserialize(std::istream& is, std::unique_ptr<Internal> internal){

		std::string bin;
		getline(is,bin);
		std::istringstream iss(bin);

		size_t num_inputs, num_outputs;
		if(!iss) pipeline_error("Error reading Pipeline from input stream.");
		iss >> num_inputs;
		if(!iss) pipeline_error("Error reading Pipeline from input stream.");
		iss >> num_outputs;

		Factory<Derived<Res(Arg),nullptr_t>> f;
		auto t=DerivedClass::deserialize(is,num_inputs,num_outputs);

		Pipeline<Res(typename Internal::Input)> *p=construct_pipe(f,std::move(internal),t);
		return std::unique_ptr<DerivedClass>(static_cast<DerivedClass*>(p));
	}

	template<typename U = Internal, class = typename std::enable_if<std::is_same<U, nullptr_t>::value>::type>
	static std::unique_ptr<DerivedClass> deserialize(std::istream& is){

		std::string bin;
		getline(is,bin);
		std::istringstream iss(bin);

		size_t num_inputs, num_outputs;
		if(!iss) pipeline_error("Error reading Pipeline from input stream.");
		iss >> num_inputs;
		if(!iss) pipeline_error("Error reading Pipeline from input stream.");
		iss >> num_outputs;

		Factory<Derived<Res(Arg),nullptr_t>> f;
		auto t=DerivedClass::deserialize(is,num_inputs,num_outputs);

		Pipeline<Res(Arg)> *p=construct_pipe(f,t);
		return std::unique_ptr<DerivedClass>(static_cast<DerivedClass*>(p));
	}
};

/*************************************************************************************************/

#define Write_Name(Type) 					\
template<typename T, typename N> 			\
struct ToStr<pipeline::Type<T,N>>{ 			\
	static string get(){ 					\
		string type{#Type}, t("pipeline::");\
		t.append(type); 					\
		return t; 							\
	} 										\
};

// macro to define new basicblocks, takes care of a bunch of stuff
#define Derive_BB_CRTP(Type) template<typename T,typename N=nullptr_t> class Type;	\
		Write_Name(Type) 															\
template <typename Ret, typename Arg, typename Internal> 							\
class Type<Ret(Arg),Internal> final : public BB_CRTP<Type<Ret(Arg),Internal>>

// macro for easy typedef in new basicblock, defines all base classes and functor types
#define Typedefs(Type) typedef BasicBlock<Ret(Arg),Internal> BaseClass;	\
		typedef BB_CRTP<Type<Ret(Arg),Internal> > CRTPClass;			\
		typedef typename BaseClass::Input Input;						\
		typedef Type<Ret(Arg),Internal> DerivedClass;					\
		typedef typename CRTPClass::PipeBase PipeBase;					\
		typedef Ret result_type;										\
		typedef Arg argument_type;

/*************************************************************************************************/

/**
 * Factory class to build Pipeline objects.
 */
template <template <class, class=nullptr_t> class BB, typename Ret, typename Arg>
struct Factory<BB<Ret(Arg)>>{

	/**
	 * Factory method for multistage pipelines.
	 *
	 * Pipeline execution order is:
	 *		Arg -> Internal.operator() -> BB.operator() -> Ret.
	 */
	template <class Internal, typename... CtorArgs>
	std::unique_ptr<BB<Ret(Arg),Internal>> operator()(std::unique_ptr<Internal> internal, CtorArgs... args) const{
		static_assert(std::is_same<typename Internal::Result,Arg>::value ||
				std::is_assignable<Arg,typename Internal::Result>::value,
				"Invalid parameters: Internal::Result is incompatible with Arg.");
		return std::unique_ptr<BB<Ret(Arg),Internal>>(new BB<Ret(Arg),Internal>(std::move(internal),args...));
	}

	/**
	 * Factory method for a single stage pipeline.
	 *
	 * Pipeline execution order is:
	 * 		Arg -> BB.operator() -> Ret.
	 */
	template <typename... CtorArgs>
	std::unique_ptr<BB<Ret(Arg)>> operator()(CtorArgs... args) const{
		return std::unique_ptr<BB<Ret(Arg),nullptr_t>>(new BB<Ret(Arg)>(args...));
	}

	/**
	 * Constructs a multistage pipeline by deserializing is.
	 */
	template <typename Internal>
	static std::unique_ptr<BB<Ret(Arg),Internal>> deserialize(std::istream& is, std::unique_ptr<Internal> internal){
		// todo: implement verification of blocks and associated types
		std::string line;
		getline(is,line);
		return BB_CRTP<BB<Ret(Arg),Internal>>::deserialize(is,std::move(internal));
	}

	/**
	 * Constructs a single stage pipeline by deserializing is.
	 */
	static std::unique_ptr<BB<Ret(Arg)>> deserialize(std::istream& is){
		// todo: implement verification of blocks and associated types
		std::string line;
		getline(is,line);
		return BB_CRTP<BB<Ret(Arg)>>::deserialize(is);
	}
};

/*************************************************************************************************/

template <typename T>
class MultistagePipe;

/**
 * Class used to model multistage pipelines. These pipelines can deserialize automatically.
 */
template <typename Res, typename Arg>
class MultistagePipe<Res(Arg)> : public Pipeline<Res(Arg)>{
public:
	MultistagePipe(size_t input_len, size_t output_len)
	:Pipeline<Res(Arg)>(input_len,output_len)
	{}

	MultistagePipe(size_t input_len)
	:Pipeline<Res(Arg)>(input_len)
	{}

	MultistagePipe(const MultistagePipe<Res(Arg)>& o)
	:Pipeline<Res(Arg)>(o)
	{}

	MultistagePipe(MultistagePipe<Res(Arg)>&& o)
	:Pipeline<Res(Arg)>(std::move(o))
	{}

	virtual ~MultistagePipe(){}
};

/*************************************************************************************************/

} // pipeline namespace

/*************************************************************************************************/

#endif /* CORE_HPP_ */
