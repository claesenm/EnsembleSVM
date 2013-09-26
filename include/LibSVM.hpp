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
 * LibSVM.h
 *
 *      Author: Marc Claesen
 */

/*************************************************************************************************/

#ifndef LIBSVM_H_
#define LIBSVM_H_

/*************************************************************************************************/

#include "svm.h"
#include "Models.hpp"
#include <memory>
#include <iostream>
#include <vector>

/*************************************************************************************************/

using std::unique_ptr;
using std::vector;

/*************************************************************************************************/

namespace ensemble{

/*************************************************************************************************/

namespace LibSVM{

/*************************************************************************************************/

typedef std::pair<std::unique_ptr<svm_problem>, std::unique_ptr<svm_parameter> > full_svm_problem;

/**
 * Converts a LibSVM model into a generic SVMModel.
 * The LibSVM model is destroyed during conversion.
 */
unique_ptr<SVMModel> convert(unique_ptr<svm_model> libsvm);

/**
 * Reads a LibSVM model from <is>.
 */
unique_ptr<svm_model> readLibSVM(std::istream &is);

/**
 * Solves given instance-weighted SVM problem using LibSVM backend and returns generic SVMModel.
 *
 * This problem is of the following form:
 *
 * 	min  1/2 w^T w + Sum_{i=1}^n C_i \xi_i
 * 	s.t. y_i(w^T phi(x_i)+\rho >= 1-\xi_i		i=1:n
 * 		 \xi_i >= 0								i=1:n
 *
 * Where C_i=pospen*penalties[i] if labels[i] is true, otherwise C_i=negpen*penalties[i].
 *
 * trainsize specifies which part of the data, labels and penalties vectors is used (0:trainsize-1).
 * These vectors may be larger than trainsize to avoid reallocation when training several models.
 *
 * IMPORTANT: positive label must be value "true"!
 * IMPORTANT: ensure that the first data instance is POSITIVE to avoid a mess later on.
 * 		This is because LIBSVM always assigns y=1 to whatever label it sees first. This way f(x)>0 == positive.
 */
unique_ptr<SVMModel> trainBSVM(const Kernel *kernel, double pospen, double negpen,
		double cachesize, const vector<const SparseVector*> &data, const vector<bool> &labels,
		const vector<double> &penalties, unsigned trainsize, bool mutelibsvm=true);


std::unique_ptr<SVMModel> libsvm_train(full_svm_problem &&problem);

full_svm_problem construct_BSVM_problem
(const Kernel *kernel, double pospen, double negpen,
		double cachesize, const vector<const SparseVector*> &data, const vector<bool> &labels,
		const vector<double> &penalties, unsigned trainsize, bool mutelibsvm=true);

/*************************************************************************************************/

} // ensemble::LibSVM namespace

/*************************************************************************************************/

} // ensemble namespace

/*************************************************************************************************/

#endif /* LIBSVM_H_ */
