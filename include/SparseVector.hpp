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
 * SparseVector.h
 *
 *      Author: Marc Claesen
 */

#ifndef SPARSEVECTOR_H_
#define SPARSEVECTOR_H_

/*************************************************************************************************/

#include <memory>
#include <map>
#include <vector>
#include <iostream>

/*************************************************************************************************/

using std::unique_ptr;
using std::vector;
using std::pair;

/*************************************************************************************************/

class svm_node;

namespace ensemble{

/*************************************************************************************************/

/**
 * Sparse Vector Class
 */
class SparseVector final{
public:
	typedef std::vector<std::pair<unsigned,double> > SparseSV;
	typedef SparseSV::size_type size_type;
	typedef SparseSV::iterator iterator;
	typedef SparseSV::const_iterator const_iterator;
	typedef SparseSV::reverse_iterator reverse_iterator;
	typedef SparseSV::const_reverse_iterator const_reverse_iterator;
	typedef double value_type;

private:
	SparseVector();

protected:
	SparseSV sparseSV;	// LibSVM-style sparse representation

public:
	SparseVector(const std::vector<double>& v);
	SparseVector(SparseSV &&content);
	SparseVector(const SparseVector &v);
	SparseVector(const svm_node* x);

	SparseVector(SparseVector&& o);
	SparseVector &operator=(const SparseVector &v)=default;
	SparseVector &operator=(SparseVector&& v);

	SparseVector operator+(const std::vector<double>& offset) const;
	SparseVector operator+(const SparseVector& offset) const;

	SparseVector operator*(const std::vector<double>& scale) const;
	SparseVector operator*(const SparseVector& scale) const;

	~SparseVector()=default;

	/**
	 * Returns the amount of nonzero elements in this SV.
	 */
	size_type numNonzero() const;

	/**
	 * Returns the index of the last nonzero element in this SV.
	 */
	unsigned size() const;

	iterator begin();
	iterator end();
	const_iterator begin() const;
	const_iterator end() const;

	reverse_iterator rbegin();
	reverse_iterator rend();
	const_reverse_iterator rbegin() const;
	const_reverse_iterator rend() const;

	/**
	 * Trims the SparseVector to a maximum length of maxlen.
	 */
	void trim(size_t maxlen);

	bool operator==(const SparseVector& other) const;
	bool operator!=(const SparseVector& other) const;
	bool operator<(const SparseVector& other) const;

	friend std::ostream &operator<<(std::ostream &os, const SparseVector &v);

	/**
	 * Reads a SparseVector in default format (replace ws by ',' if csv=true)
	 * csv=true: reads sparsevector in SparseCSV format
	 *
	 * <idx1>:<value1> <idx2>:<value2> ...
	 */
	static unique_ptr<SparseVector> read(std::istream &iss, bool csv=false);

	/**
	 * Reads a SparseVector formatted in CSV from iss.
	 *
	 * <value1>,<value2>,...
	 */
	static unique_ptr<SparseVector> readCSV(std::istream &iss);

	static unique_ptr<SparseVector> readf(std::istream &iss, unsigned format=0);

	vector<pair<unsigned, double> > toVector() const;

	/**
	 * Gets the value at index idx.
	 * Using this is NOT efficient.
	 */
	double operator[](unsigned idx) const;

	/**
	 * Returns the density of the SparseVector.
	 *
	 * This equals #nonzeros/length.
	 */
	double density() const;

	std::vector<double> dense() const;
};

/*************************************************************************************************/

/**
 * Returns the inner product between x and y.
 */
double InnerProduct(const SparseVector &x, const SparseVector &y);

/**
 * Returns the inner product between x and y.
 */
double InnerProduct(const vector<pair<unsigned,double> > &x, const SparseVector &y);

/**
 * Returns the inner product between x and y.
 */
template <typename T>
double InnerProduct(const vector<T> &x, const SparseVector &y){
	double result=0;
	for(SparseVector::const_iterator I=y.begin(),E=y.end();I!=E;++I){
		if(I->first > x.size())
			break;
		result+=x[I->first-1]*I->second;
	}
	return result;
}

/**
 * Returns the inner product between x and y.
 */
template <typename T>
T InnerProduct(const vector<T> &x, const vector<T> &y){
	double result=0;
	unsigned s=x.size()<y.size() ? x.size() : y.size();
	for(unsigned i=0;i<s;++i)
		result+=x[i]*y[i];
	return result;
}

/**
 * Returns the inner product between x and y.
 */
double InnerProduct(const vector<pair<unsigned,double> > &x, const vector<pair<unsigned,double> > &y);

/**
 * Computes the elementwise product of x and y.
 */
void ElementWiseProduct(const vector<pair<unsigned,double> > &x, const SparseVector &y, vector<pair<unsigned,double> > &xy);

/**
 * Computes the elementwise product of x and y.
 */
template <typename T>
void ElementWiseProduct(const vector<T> &x, const vector<T> &y, vector<T> &xy);

double squaredNorm(const SparseVector &v);
double squaredNorm(const vector<pair<unsigned,double> > &v);

/*************************************************************************************************/

} // ensemble namespace

/*************************************************************************************************/

namespace pipeline{
namespace impl{

using ensemble::SparseVector;

/*************************************************************************************************/

SparseVector Offset(SparseVector&& sv, const std::vector<double>& offsets, size_t numoutputs);
SparseVector Scale(SparseVector&& sv, const std::vector<double>& scale, size_t numoutputs);

/*************************************************************************************************/

} // pipeline::impl namespace
} // pipeline namespace

/*************************************************************************************************/

#endif /* SPARSEVECTOR_H_ */
