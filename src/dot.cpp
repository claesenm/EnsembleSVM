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
 * dot.cpp
 *
 *      Author: Marc Claesen
 */

#include "dot.h"

// This file should only be included for compilation when BLAS is available.
// If BLAS is found by the configure script, the CPP macro HAVE_BLAS is defined.

typedef float real;
typedef double doublereal;
typedef long int integer;

namespace{

extern "C"{
	real sdot_(integer *n, real *x, integer *incx, real *y, integer *incy);
	doublereal ddot_(integer *n, doublereal *x, integer *incx, doublereal *y, integer *incy);
}

}

namespace ensemble{

float dot(const std::vector<float> &x, const std::vector<float> &y){
	assert(x.size()==y.size() && "Attempting to compute inner product of vectors of unequal lengths!");
	integer N=x.size();
	integer INC=1;

	// const-casting because BLAS wont change the vectors anyways, C++ <> FORTRAN
	std::vector<float> &xnoconst=const_cast<std::vector<float>& >(x);
	std::vector<float> &ynoconst=const_cast<std::vector<float>& >(y);

	float result=sdot_(&N,&xnoconst[0],&INC,&ynoconst[0],&INC);
	return result;
}

double dot(const std::vector<double> &x, const std::vector<double> &y){
	assert(x.size()==y.size() && "Attempting to compute inner product of vectors of unequal lengths!");
	integer N=x.size();
	integer INC=1;

	// const-casting because BLAS wont change the vectors anyways, C++ <> FORTRAN
	std::vector<double> &xnoconst=const_cast<std::vector<double>& >(x);
	std::vector<double> &ynoconst=const_cast<std::vector<double>& >(y);

	double result=ddot_(&N,&xnoconst[0],&INC,&ynoconst[0],&INC);
	return result;
}

} // ensemble namespace
