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
 * io.h
 *
 *      Author: Marc Claesen
 */

/*************************************************************************************************/

#ifndef IO_H_
#define IO_H_

/*************************************************************************************************/

#include "SparseVector.hpp"
#include <list>
#include <deque>
#include <memory>
#include <exception>

/*************************************************************************************************/

using std::unique_ptr;

/*************************************************************************************************/

namespace ensemble{

/*************************************************************************************************/

class invalid_file_exception: public std::exception{
	virtual const char* what() const throw()
			  {
		return "Attempting to read invalid model file.";
			  }
};


struct FileFormats{
	enum {
		DEFAULT = 0,
		CSV = 1,
		SparseCSV = 2
	};
};

/*************************************************************************************************/

/**
 * Reads individual misclassification penalties per training instance.
 *
 * Data file of the form:
 * <double>\n
 * <double>\n
 * ...
 *
 * Returns penalties, indexed by line number.
 */
unique_ptr< std::deque<double> > ReadIndividualPenaltiesFromFile(const std::string &fname);

/**
 * Reads the bootstrap mask from fname into mask.
 * Bootstrap masks are stored per model (rows), with integers marking the indices of data points that are used.
 * Number of models must equal predefined mask.size().
 *
 * Expects nullptr pointers in mask upon entry. Fills mask with correct pointers, these must be managed by the user.
 *
 * Indexing is 1-based, e.g. the first training instance has index 1.
 * File format is space-separated.
 */
void readBootstrapMask(const std::string &fname, std::vector< std::list<unsigned>*> &mask, char delim=*" ");

/**
 * Reads the weight mask from fname into mask.
 * Weight masks are stored per model (rows), in sparse format <unsigned: instanceidx>:<double: weight in model>
 * Number of models must equal predefined mask.size().
 *
 * Expects nullptr pointers in mask upon entry. Fills mask with correct pointers, these must be managed by the user.
 *
 * Indexing is 1-based, e.g. the first training instance has index 1.
 * File format is space-separated.
 */
void readWeightMask(const std::string &fname, std::vector< SparseVector* > &mask);

/**
 * Reads labels from datafname, stores correct indices in pos and neg.
 *
 * Labels must be stored in the first column, delimited by <delim>.
 * If posvall==true, all labels!=poslabel are considered negative.
 */
void readLabels(std::ifstream &file, char delim, const std::string &poslabel, const std::string &neglabel, std::deque<unsigned> &pos, std::deque<unsigned> &neg, bool posvall);

/**
 * Reads cross-validation mask from file.
 *
 * Returns the indices per fold in mask (key = folds, value = list of indices).
 */
void readCrossvalMask(const std::string &filename, std::map<unsigned, std::deque<unsigned> > &mask);

/*************************************************************************************************/

} // ensemble namespace

/*************************************************************************************************/

#endif /* IO_H_ */
