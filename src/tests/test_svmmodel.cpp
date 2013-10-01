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
 * test_svmmodel.cpp
 *
 *      Author: Marc Claesen
 */


#include "Executable.hpp"
#include "SelectiveFactory.hpp"
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

bool test_io(const SVMModel& m){

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
	{
		std::cout << "Testing SVMModel with linear kernel." << std::endl;
		std::unique_ptr<Kernel> linear(new LinearKernel());
		SVMModel model(std::move(SVs),{1.0,-1.0},std::move(classes),{0.0},std::move(linear));
		globalerr = globalerr | test_io(model);
	}
	{
		std::cout << "Testing SVMModel with RBF kernel." << std::endl;
		std::unique_ptr<Kernel> linear(new RBFKernel(0.5));
		SVMModel model(std::move(SVs),{1.0,-1.0},std::move(classes),{0.0},std::move(linear));
		globalerr = globalerr | test_io(model);
	}
	{
		std::cout << "Testing SVMModel with polynomial kernel." << std::endl;
		std::unique_ptr<Kernel> linear(new PolyKernel(3,1.0,0.5));
		SVMModel model(std::move(SVs),{1.0,-1.0},std::move(classes),{0.0},std::move(linear));
		globalerr = globalerr | test_io(model);
	}

	if(globalerr) exit(EXIT_FAILURE);
	else exit(EXIT_SUCCESS);
}
