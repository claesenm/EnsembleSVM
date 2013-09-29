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
 * blocks.cpp
 *
 *      Author: Marc Claesen
 */

/*************************************************************************************************/

#include <iostream>
#include "pipeline/pipelines.hpp"
#include "Models.hpp"

/*************************************************************************************************/

using std::string;
using std::vector;
using namespace ensemble::pipeline;

typedef std::vector<double> Vector;

/*************************************************************************************************/

template <typename Res, typename Arg>
void failure(const Pipeline<Res(Arg)>& pipe, const char* test){
	std::cerr << test << " test failed for " << std::endl;
	std::cerr << pipe;
}

template <typename T, typename... CtorArgs>
bool test_io(const Factory<T>& factory, CtorArgs... args){

	auto pipe = factory(args...);
	std::stringbuf buffer1, buffer2;
	std::iostream stream1(&buffer1), stream2(&buffer2);
	std::string str1, str2;

	stream1 << *pipe;
	str1=buffer1.str();

	auto deserialized = Factory<T>::deserialize(stream1);
	stream2 << *deserialized;
	str2=buffer2.str();

	bool error = (str1.compare(str2)!=0);
	if(error) failure(*pipe,"io");
	return error;
}

template <typename Res, typename Arg>
bool test_io_multistage(const Pipeline<Res(Arg)>& pipe){
	std::stringbuf buffer1, buffer2;
	std::iostream stream1(&buffer1), stream2(&buffer2);
	std::string str1, str2;

	stream1 << pipe;
	str1=buffer1.str();

	deserializer<Res(Arg)> d;
	std::unique_ptr<Pipeline<Res(Arg)>> deserialized = d(stream1);

	stream2 << *deserialized;
	str2=buffer2.str();

	bool error = (str1.compare(str2)!=0);
	if(error) failure(*pipe,"io_multi");
	return error;
}

template <typename Res, typename Arg>
bool test_functor(const Pipeline<Res(Arg)>* pipe, Res expected, Arg input){
	Res output = (*pipe)(input);
	bool error = expected!=output;
	if(error) failure(*pipe,"functor");
	return error;
}

/*************************************************************************************************/

void test_scale(bool& globalerr){
	// scalar
	{
		Factory<Scale<double(double)>> f;
		std::unique_ptr<Scale<double(double)>> pipe = f(5.0,1);
		globalerr = globalerr | test_io(f,5.0,1);
		globalerr = globalerr | test_functor(pipe.get(),5.0,1.0);
	}
	// vector
	{
		Factory<Scale<Vector(Vector)>> f;
		std::unique_ptr<Scale<Vector(Vector)>> pipe = f(5.0,3);
		globalerr = globalerr | test_io(f,5.0,3);
		globalerr = globalerr | test_functor(pipe.get(),{5,10,15},{1,2,3});
	}
	// vector 2
	{
		Factory<Scale<Vector(Vector)>> f;
		Vector coef={1.0,2.0,3.0};
		{
			// should compile with explicit size
			std::unique_ptr<Scale<Vector(Vector)>> pipe = f(coef,3);
		}
		std::unique_ptr<Scale<Vector(Vector)>> pipe = f(coef);
		globalerr = globalerr | test_io(f,coef,3);
		globalerr = globalerr | test_functor(pipe.get(),{2,4,6},{2,2,2});
	}
}

/*************************************************************************************************/

void test_offset(bool& globalerr){
	// scalar
	{
		Factory<Offset<double(double)>> f;
		std::unique_ptr<Offset<double(double)>> pipe = f(5.0,1);
		globalerr = globalerr | test_io(f,5.0,1);
		globalerr = globalerr | test_functor(pipe.get(),6.0,1.0);
	}
	// vector
	{
		Factory<Offset<Vector(Vector)>> f;
		std::unique_ptr<Offset<Vector(Vector)>> pipe = f(5.0,3);
		globalerr = globalerr | test_io(f,5.0,3);
		globalerr = globalerr | test_functor(pipe.get(),{5,10,15},{0,5,10});
	}
	// vector 2
	{
		Factory<Offset<Vector(Vector)>> f;
		Vector offsets={1.0,2.0,3.0};
		std::unique_ptr<Offset<Vector(Vector)>> pipe = f(offsets);
		globalerr = globalerr | test_io(f,offsets);
		globalerr = globalerr | test_functor(pipe.get(),{1,2,3},{0,0,0});
	}
}

/*************************************************************************************************/

void test_logistic(bool& globalerr){
	// scalar
	{
		Factory<Logistic<double(double)>> f;
		std::unique_ptr<Logistic<double(double)>> pipe = f(1);
		globalerr = globalerr | test_io(f,1);
		globalerr = globalerr | test_functor(pipe.get(),1.0/(1+exp(-1.0)),1.0);
	}
	// vector
	{
		Factory<Logistic<Vector(Vector)>> f;
		std::unique_ptr<Logistic<Vector(Vector)>> pipe = f(2);
		globalerr = globalerr | test_io(f,2);
		globalerr = globalerr | test_functor(pipe.get(),{1.0/(1+exp(-1.0)),1.0/(1+exp(-2.0))},{1.0,2.0});
	}
}

/*************************************************************************************************/

void test_threshold(bool& globalerr){
	// scalar
	{
		Factory<Threshold<double(double)>> f;
		std::unique_ptr<Threshold<double(double)>> pipe = f(0.0,1.0,0.0);
		globalerr = globalerr | test_io(f,0.0,1.0,0.0);
		globalerr = globalerr | test_functor(pipe.get(),0.0,-1.0);
		globalerr = globalerr | test_functor(pipe.get(),1.0,1.0);
	}
	// scalar 2
	{
		Factory<Threshold<bool(double)>> f;
		std::unique_ptr<Threshold<bool(double)>> pipe = f(0.0,true,false);
		globalerr = globalerr | test_io(f,0.0,true,false);
		globalerr = globalerr | test_functor(pipe.get(),true,1.0);
		globalerr = globalerr | test_functor(pipe.get(),false,-1.0);
	}
	// vector
	{
		Factory<Threshold<Vector(Vector)>> f;
		std::unique_ptr<Threshold<Vector(Vector)>> pipe = f(0.0,1.0,0.0,3);
		globalerr = globalerr | test_io(f,0,1.0,0.0,3);
		globalerr = globalerr | test_functor(pipe.get(),{0.0,1.0,1.0},{-1.0,1.0,1.5});
	}
	// vector 2
	{
		typedef std::vector<bool> BoolVec;
		typedef std::vector<int> IntVec;
		Factory<Threshold<BoolVec(IntVec)>> f;
		std::unique_ptr<Threshold<BoolVec(IntVec)>> pipe = f(0,true,false,3);
		globalerr = globalerr | test_io(f,0,true,false,3);
		globalerr = globalerr | test_functor(pipe.get(),{false,true,true},{-1,1,1});
	}
}

/*************************************************************************************************/

void test_average(bool& globalerr){
	// same type
	{
		Factory<Average<double(std::vector<double>)>> f;
		std::unique_ptr<Average<double(std::vector<double>)>> pipe = f(3);
		globalerr = globalerr | test_io(f,3);
		globalerr = globalerr | test_functor(pipe.get(),0.0,{-1.0,0.0,1.0});
	}
	// with divisor, unconstrained size
	{
		Factory<Average<double(std::vector<double>)>> f;
		std::unique_ptr<Average<double(std::vector<double>)>> pipe = f(1.0);
		globalerr = globalerr | test_io(f,1.0);
		globalerr = globalerr | test_functor(pipe.get(),0.0,{-1.0,0.0,1.0});
		globalerr = globalerr | test_functor(pipe.get(),0.0,{-2.0-1.0,0.0,1.0,2.0});
	}
	// with divisor
	{
		Factory<Average<double(std::vector<double>)>> f;
		std::unique_ptr<Average<double(std::vector<double>)>> pipe = f(4.0,3);
		globalerr = globalerr | test_io(f,4.0,3);
		globalerr = globalerr | test_functor(pipe.get(),1.0,{1.0,2.0,1.0});
	}
	// different return type
	{
		Factory<Average<int(std::vector<double>)>> f;
		std::unique_ptr<Average<int(std::vector<double>)>> pipe = f(2);
		globalerr = globalerr | test_io(f,2);
		globalerr = globalerr | test_functor(pipe.get(),1,{0.0,2.0});
	}
}

/*************************************************************************************************/

void test_sum(bool& globalerr){
	// vector
	{
		Factory<Sum<double(Vector)>> f;
		std::unique_ptr<Sum<double(Vector)>> pipe = f(3);
		globalerr = globalerr | test_io(f,3);
		globalerr = globalerr | test_functor(pipe.get(),6.0,{1.0,2.0,3.0});
	}
	// unconstrained num of inputs
	{
		Factory<Sum<double(Vector)>> f;
		std::unique_ptr<Sum<double(Vector)>> pipe = f();
		globalerr = globalerr | test_io(f);
		globalerr = globalerr | test_functor(pipe.get(),6.0,{1.0,2.0,3.0});
		globalerr = globalerr | test_functor(pipe.get(),10.0,{1.0,2.0,3.0,4.0});
	}
	// switch types
	{
		Factory<Sum<int(std::list<double>)>> f;
		std::unique_ptr<Sum<int(std::list<double>)>> pipe = f(3);
		globalerr = globalerr | test_io(f,3);
		globalerr = globalerr | test_functor(pipe.get(),6,{1.0,2.0,3.0});
	}
}

/*************************************************************************************************/

void test_median(bool& globalerr){
	// vector
	{
		Factory<Median<double(Vector)>> f;
		std::unique_ptr<Median<double(Vector)>> pipe = f(3);
		globalerr = globalerr | test_io(f,3);
		globalerr = globalerr | test_functor(pipe.get(),1.0,{0.0,1.0,2.0});
	}
	// unconstrained
	{
		Factory<Median<double(std::deque<double>)>> f;
		std::unique_ptr<Median<double(std::deque<double>)>> pipe = f();
		globalerr = globalerr | test_io(f);
		globalerr = globalerr | test_functor(pipe.get(),1.0,{0.0,1.0,2.0});
		globalerr = globalerr | test_functor(pipe.get(),1.0,{-1.0,0.0,1.0,2.0,1.5});
	}
	// list fixme
//	{
//		Factory<Median<double(std::list<double>)>> f;
//		std::unique_ptr<Median<double(std::list<double>)>> pipe = f();
//		globalerr = globalerr | test_io(f);
//		globalerr = globalerr | test_functor(pipe.get(),1.0,{0.0,1.0,2.0});
//		globalerr = globalerr | test_functor(pipe.get(),1.0,{-1.0,0.0,1.0,2.0,1.5});
//	}
}

/*************************************************************************************************/

using ensemble::SVMModel;
using ensemble::SparseVector;
using ensemble::Kernel;
using ensemble::LinearKernel;

std::unique_ptr<SVMModel> makeSVM(){
	SVMModel::SV_container SVs;
	{
		std::vector<double> v={1.0,0.0,2.0};
		SVs.emplace_back(new SparseVector(v));
	}
	{
		std::vector<double> v={-1.0,1.0};
		SVs.emplace_back(new SparseVector(v));
	}

	SVMModel::Classes classes;
	classes.emplace_back("positive",1);
	classes.emplace_back("negative",1);
	std::unique_ptr<Kernel> linear(new LinearKernel());
	return std::unique_ptr<SVMModel>(
			new SVMModel(std::move(SVs),{1.0,-1.0},std::move(classes),{0.0},std::move(linear)));
}

void test_svm(bool& globalerr){
	// vector argument
	{

		Factory<SVM<double(Vector)>> f;

		{
			auto svm = makeSVM();
			globalerr = globalerr | test_io(f,svm.release());
		}

		auto svm = makeSVM();
		std::unique_ptr<SVM<double(Vector)>> pipe = f(svm.release());
		globalerr = globalerr | test_functor(pipe.get(),3.0,{0.0,1.0,2.0});
	}
	// sparsevector argument
	{

		Factory<SVM<double(SparseVector)>> f;

		{
			auto svm = makeSVM();
			globalerr = globalerr | test_io(f,svm.release());
		}

		auto svm = makeSVM();
		std::unique_ptr<SVM<double(SparseVector)>> pipe = f(svm.release());
		std::vector<double> vec={0.0,1.0,2.0};
		SparseVector sparse(vec);
		globalerr = globalerr | test_functor(pipe.get(),3.0,sparse);
	}
}

/*************************************************************************************************/


int main(int argc, char **argv)
{
	bool globalerr=false;

	std::cout << "Testing Scale BasicBlock." << std::endl;
	test_scale(globalerr);
	std::cout << "Testing Offset BasicBlock." << std::endl;
	test_offset(globalerr);
	std::cout << "Testing Logistic BasicBlock." << std::endl;
	test_logistic(globalerr);
	std::cout << "Testing Threshold BasicBlock." << std::endl;
	test_threshold(globalerr);
	std::cout << "Testing Average BasicBlock." << std::endl;
	test_average(globalerr);
	std::cout << "Testing Sum BasicBlock." << std::endl;
	test_sum(globalerr);
	std::cout << "Testing Median BasicBlock." << std::endl;
	test_median(globalerr);
	std::cout << "Testing SVM BasicBlock." << std::endl;
	test_svm(globalerr);

	if(globalerr) exit(EXIT_FAILURE);
	else exit(EXIT_SUCCESS);
}
