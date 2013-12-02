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
 * test_userdefkernel.cpp
 *
 *      Author: Marc Claesen
 */


#include "Executable.hpp"
#include "SelectiveFactory.hpp"
#include "SparseVector.hpp"
#include "Kernel.hpp"
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

/*************************************************************************************************/

int main(int argc, char **argv)
{
	bool globalerr=false;

	// test construction
	unique_ptr<Kernel> kernel=KernelFactory(4,0,0,1);

	// test vectors
	SparseVector y{Vector{1,2,3,4}};
	SparseVector x1{Vector{1}}, x2{Vector{2}}, x3{Vector{3}};

	double result = kernel->k_function(&x1,&y);
	if(result!=2) globalerr=true;

	result = kernel->k_function(&x2,&y);
	if(result!=3) globalerr=true;

	result = kernel->k_function(&x3,&y);
	if(result!=4) globalerr=true;

	if(globalerr) exit(EXIT_FAILURE);
	else exit(EXIT_SUCCESS);
}
