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
 * pipelines.cpp
 *
 *      Author: Marc Claesen
 */

/*************************************************************************************************/

#include <iostream>
#include <cstdlib>
#include "Executable.hpp"
#include "pipeline/pipelines.hpp"
#include "Models.hpp"

/*************************************************************************************************/

using std::string;
using std::vector;
using namespace ensemble::pipeline;

typedef std::vector<double> Vector;

/*************************************************************************************************/

template <typename Res, typename Arg>
void failure(const MultistagePipe<Res(Arg)>& pipe, const char* test){
	std::cerr << test << " test failed for " << std::endl;
	std::cerr << pipe;
}

template <typename Res, typename Arg>
bool test_io_multistage(const MultistagePipe<Res(Arg)>* pipe){
	std::stringbuf buffer1, buffer2;
	std::iostream stream1(&buffer1), stream2(&buffer2);
	std::string str1, str2;

	stream1 << *pipe;
	str1=buffer1.str();

	deserializer<Res(Arg)> d;
	std::unique_ptr<MultistagePipe<Res(Arg)>> deserialized = d(stream1);


	stream2 << *deserialized;
	str2=buffer2.str();

	bool error = (str1.compare(str2)!=0);
	if(error) failure(*pipe,"io_multi");
	return error;
}

template <typename Res, typename Arg>
bool test_functor(const MultistagePipe<Res(Arg)>* pipe, Res expected, Arg input){
	Res output = (*pipe)(input);
	bool error = expected!=output;
	if(error) failure(*pipe,"functor");
	return error;
}

/*************************************************************************************************/

void test_majorityvote(bool& globalerr){
	// set num inputs
	{
		Factory<MajorityVote> f;
		std::unique_ptr<MultistagePipe<double(std::vector<double>)>> pipe = f(5);
		globalerr = globalerr | test_io_multistage(pipe.get());
		globalerr = globalerr | test_functor(pipe.get(),3.0/5,{-1.0,-2.0,0.1,1.0,2.1});
	}
	// unconstrained
	{
		Factory<MajorityVote> f;
		std::unique_ptr<MultistagePipe<double(std::vector<double>)>> pipe = f();
		globalerr = globalerr | test_io_multistage(pipe.get());
		globalerr = globalerr | test_functor(pipe.get(),3.0/5,{-1.0,-2.0,0.1,1.0,2.1});
		globalerr = globalerr | test_functor(pipe.get(),1.0/3,{-1.0,-2.0,0.1});
	}
	// vector 1
	{
		Factory<MajorityVote> f;
		std::unique_ptr<MultistagePipe<double(std::vector<double>)>> pipe = f({1.0,1.0,1.0});
		globalerr = globalerr | test_io_multistage(pipe.get());
		globalerr = globalerr | test_functor(pipe.get(),2.0/3,{-0.4,1.0,0.8});
	}
	// vector with non-zero threshold
	{
		Factory<MajorityVote> f;
		std::unique_ptr<MultistagePipe<double(std::vector<double>)>> pipe = f({1.0,1.0,1.0},0.5);
		globalerr = globalerr | test_io_multistage(pipe.get());
		globalerr = globalerr | test_functor(pipe.get(),2.0/3,{0.4,1.0,0.8});
	}
	// vector with non-zero threshold and various scales
	{
		Factory<MajorityVote> f;
		std::unique_ptr<MultistagePipe<double(std::vector<double>)>> pipe = f({2.0,1.0,1.0},0.5);
		globalerr = globalerr | test_io_multistage(pipe.get());
		globalerr = globalerr | test_functor(pipe.get(),2.0/4,{0.4,1.0,0.8});
	}
}

/*************************************************************************************************/

void test_logisticregression(bool& globalerr){
	{
		Factory<LogisticRegression> f;
		std::unique_ptr<MultistagePipe<double(std::vector<double>)>> pipe=f({2.0,1.0,1.0},1);
		globalerr = globalerr | test_io_multistage(pipe.get());
		globalerr = globalerr | test_functor(pipe.get(),1.0/(1+exp(-7)),{1.0,2.0,2.0});
	}
}

/*************************************************************************************************/

void test_normalizelinear(bool& globalerr){
	{
		Factory<NormalizeLinear> f;
		std::unique_ptr<MultistagePipe<SparseVector(SparseVector)>> pipe=f({1.0,2.0,3.0},{-2.0,-1.0,1.0});
		globalerr = globalerr | test_io_multistage(pipe.get());

		Vector v1dat={0.0, 1.0, 2.0}, v2dat={1.0, 0.0, 2.0, 0.0, 4.0};
		SparseVector v1(v1dat), v2(v2dat);

		Vector v1res={-2.0,1.0,7.0}, v2res={-1.0,-1.0,7.0};
		SparseVector v1result(v1res), v2result(v2res);

		globalerr = globalerr | test_functor(pipe.get(),v1result,std::move(v1));
		globalerr = globalerr | test_functor(pipe.get(),v2result,std::move(v2));
	}
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

void test_binarysvm(bool& globalerr){
	{
		Factory<BinarySVMAggregation> f;
		auto svm = makeSVM();

		std::unique_ptr<MultistagePipe<double(std::vector<double>)>> pipe=f(std::move(svm));
		globalerr = globalerr | test_io_multistage(pipe.get());

		std::vector<double> vec={0.0,1.0,2.0};
		globalerr = globalerr | test_functor(pipe.get(),3.0,std::move(vec));
	}
}

void test_linearaggr(bool& globalerr){
	{
		Factory<LinearAggregation> f;
		std::unique_ptr<MultistagePipe<double(std::vector<double>)>>
				pipe = f({1.0,2.0,-1.0},0.5);

		globalerr = globalerr | test_io_multistage(pipe.get());

		std::vector<double> vec={0.0,1.0,2.0};
		globalerr = globalerr | test_functor(pipe.get(),-0.5,std::move(vec));
	}
}


/*************************************************************************************************/

int main(int argc, char **argv)
{
	bool globalerr=false;

	std::cout << "Testing MajorityVote Pipeline." << std::endl;
	test_majorityvote(globalerr);
	std::cout << "Testing LogisticRegression Pipeline." << std::endl;
	test_logisticregression(globalerr);
	std::cout << "Testing NormalizeLinear Pipeline." << std::endl;
	test_normalizelinear(globalerr);
	std::cout << "Testing BinarySVMAggregation Pipeline." << std::endl;
	test_binarysvm(globalerr);

	if(globalerr) exit(EXIT_FAILURE);
	else exit(EXIT_SUCCESS);
}
