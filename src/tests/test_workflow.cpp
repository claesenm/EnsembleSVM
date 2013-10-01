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
 * test_workflow.cpp
 *
 *      Author: Marc Claesen
 */

/*************************************************************************************************/

#include "BinaryWorkflow.hpp"
#include "SelectiveFactory.hpp"
#include "Executable.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>

/*************************************************************************************************/

using std::string;
using std::vector;
using namespace ensemble;

typedef std::vector<double> Vector;

/*************************************************************************************************/

template <typename T>
void failure(const T& t, const char* test){
	std::cerr << test << " test failed for " << std::endl;
	std::cerr << t;
}

bool test_io(const BinaryWorkflow& m){

	std::stringbuf buffer1, buffer2;
	std::iostream stream1(&buffer1), stream2(&buffer2);
	std::string str1, str2;

	stream1 << m;
	str1=buffer1.str();

	std::unique_ptr<BinaryModel> deserialized = BinaryModel::deserialize(stream1);
	stream2 << *deserialized;
	str2=buffer2.str();

	bool error = (str1.compare(str2)!=0);
	if(error) failure(m,"io");
	return error;
}

/*************************************************************************************************/

int main(int argc, char **argv)
{
	bool globalerr=false;

	SVMModel::SV_container SVs1, SVs2;
	{
		std::vector<double> v={1.0,0.0,2.0};
		SVs1.emplace_back(new SparseVector(v));
		SVs2.emplace_back(new SparseVector(v));
	}
	{
		std::vector<double> v={-1.0,1.0};
		SVs1.emplace_back(new SparseVector(v));
	}
	{
		std::vector<double> v={1.0,0.0,0.0,4.0};
		SVs2.emplace_back(new SparseVector(v));
	}

	{
		std::cout << "Testing BinaryWorkflow with SVM model." << std::endl;

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
		std::unique_ptr<BinaryModel> predictor(new SVMModel(std::move(SVs),{1.0,-1.0},
				std::move(classes),{0.0},std::move(linear)));

		std::unique_ptr<BinaryWorkflow> flow = defaultBinaryWorkflow(std::move(predictor));
		globalerr = globalerr | test_io(*flow);

		// add preprocessing
		std::unique_ptr<MultistagePipe<SparseVector(SparseVector)>> pre;
		{
			pipeline::Factory<pipeline::NormalizeLinear> f;
			std::vector<double> coefs={1.0,2.0,3.0};
			std::vector<double> offsets={0.0,1.0,-1.0};
			pre = f(std::move(coefs),std::move(offsets));
		}
		flow->set_preprocessing(std::move(pre));
		globalerr = globalerr | test_io(*flow);
	}

	std::vector<std::unique_ptr<SVMModel>> models;
	{
		SVMModel::Classes classes1, classes2;
		classes1.emplace_back("positive",1);
		classes1.emplace_back("negative",1);
		classes2=classes1;

		std::unique_ptr<Kernel> linear(new LinearKernel());
		models.emplace_back(new SVMModel(std::move(SVs1),{1.0,-1.0},std::move(classes1),{0.0},std::move(linear)));

		linear.reset(new LinearKernel());
		models.emplace_back(new SVMModel(std::move(SVs2),{1.0,-1.0},std::move(classes2),{0.0},std::move(linear)));

		std::cout << "Testing BinaryWorkflow with SVM ensemble." << std::endl;
		std::unique_ptr<BinaryModel> ensemble(new SVMEnsemble(std::move(models)));

		std::unique_ptr<BinaryWorkflow> flow = defaultBinaryWorkflow(std::move(ensemble));
		globalerr = globalerr | test_io(*flow);

		// add preprocessing
		std::unique_ptr<MultistagePipe<SparseVector(SparseVector)>> pre;
		{
			pipeline::Factory<pipeline::NormalizeLinear> f;
			std::vector<double> coefs={1.0,2.0,3.0};
			std::vector<double> offsets={0.0,1.0,-1.0};
			pre = f(std::move(coefs),std::move(offsets));
		}
		flow->set_preprocessing(std::move(pre));
		globalerr = globalerr | test_io(*flow);

		// explicitly set postprocessing to MV
		std::unique_ptr<MultistagePipe<double(std::vector<double>)>> post;
		{
			pipeline::Factory<pipeline::MajorityVote> f;
			post.reset(f(flow->num_predictor_outputs()).release());
		}
		flow->set_postprocessing(std::move(post));
		globalerr = globalerr | test_io(*flow);

		// set postprocessing to LR
		{
			pipeline::Factory<pipeline::LogisticRegression> f;
			post.reset(f(flow->num_predictor_outputs()).release());
		}

		flow->set_postprocessing(std::move(post));
		globalerr = globalerr | test_io(*flow);
	}

	if(globalerr) exit(EXIT_FAILURE);
	else exit(EXIT_SUCCESS);
}
