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
 * test_sparsevector.cpp
 *
 *      Author: Marc Claesen
 */

#include "SparseVector.hpp"
#include "Util.hpp"
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

bool test_io(const SparseVector& m){

	std::stringbuf buffer1, buffer2;
	std::iostream stream1(&buffer1), stream2(&buffer2);
	std::string str1, str2;

	stream1 << m;
	str1=buffer1.str();

	auto deserialized = SparseVector::read(stream1,false);
	stream2 << *deserialized;
	str2=buffer2.str();

	bool error = (str1.compare(str2)!=0);
	if(error) failure(m,"io");
	return error;
}

bool test_eq(const SparseVector& v1, const SparseVector& v2){
	bool error = v1!=v2;
	if(error) failure(v1,"equality");
	return error;
}

/*************************************************************************************************/

int main(int argc, char **argv)
{
	bool globalerr=false;

	SparseVector::SparseSV adat={{1,1.0},{3,2.0}}, bdat={{1,2.0},{2,-1.0}};
	SparseVector a(std::move(adat)), b(std::move(bdat));

	std::vector<double> avdat={1.0,0.0,2.0}, bvdat={2.0,-1.0};
	SparseVector av(avdat), bv(bvdat);

	{
		std::cout << "Testing SparseVector constructors." << std::endl;
		test_eq(a,av);
		test_eq(b,bv);
	}
	{
		std::cout << "Testing SparseVector io." << std::endl;
		test_io(a);
		test_io(b);
		test_io(av);
		test_io(bv);
	}
	{
		std::cout << "Testing SparseVector operator+." << std::endl;
		std::vector<double> sum={3.0,-1.0,2.0};
		SparseVector solution(sum);
		SparseVector test=a+b;
		SparseVector test2=a+bvdat;
		test_eq(test,solution);
		test_eq(test2,solution);
	}
	{
		std::cout << "Testing SparseVector operator*." << std::endl;
		std::vector<double> product={2.0};
		SparseVector solution(product);
		SparseVector test=a*b;
		SparseVector test2=a*bvdat;
		test_eq(test,solution);
		test_eq(test2,solution);
	}
	{
		std::cout << "Testing SparseVector trimming." << std::endl;
		std::vector<double> data={2.0,0.0,3.0,5.0,1.0,0.0,1.0};
		SparseVector v(data);
		std::vector<double> data1={2.0,0.0,3.0,5.0,1.0};
		SparseVector v1(data1);

		v.trim(5);
		test_eq(v,v1);

		v.trim(3);
		v1.trim(3);
		test_eq(v,v1);
	}

	if(globalerr) exit(EXIT_FAILURE);
	else exit(EXIT_SUCCESS);
}


