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
 * SparseVector.cpp
 *
 *      Author: Marc Claesen
 */

#include "SparseVector.hpp"
#include "io.hpp"
#include "Util.hpp"
#include "svm.h"
#include <assert.h>
#include <math.h>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>

using std::string;

/*************************************************************************************************/

namespace{

/*************************************************************************************************/

std::streamsize PRECISION = 16;

typedef std::deque<std::pair<int,double> > SVDeque;

/*************************************************************************************************/

}

/*************************************************************************************************/

namespace ensemble{

/*************************************************************************************************/

SparseVector::SparseVector(){}
SparseVector::SparseVector(SparseSV&& content):sparseSV(std::move(content)){}
SparseVector::SparseVector(SparseVector&& o):sparseSV(std::move(o.sparseSV)){}
SparseVector::SparseVector(const std::vector<double>& v){
	sparseSV.reserve(v.size());
	unsigned i=1;
	for(auto e: v){
		if(e!=0)
			sparseSV.emplace_back(i,e);
		++i;
	}
}
SparseVector::SparseVector(const svm_node* x){
	// find size of the node
	int size=0;
	while(x[size].index!=-1)
		++size;

	sparseSV.reserve(size);
	for(int i=0;i<size;++i){
		sparseSV.emplace_back(x[i].index,x[i].value);
	}
}
SparseVector::size_type SparseVector::numNonzero() const{ return sparseSV.size(); }
unsigned SparseVector::size() const{
	if(sparseSV.empty()) return 0;
	return sparseSV.rbegin()->first;
}
SparseVector &SparseVector::operator=(SparseVector&& v){
	sparseSV=std::move(v.sparseSV);
	return *this;
}

SparseVector::SparseVector(const SparseVector &v):sparseSV(v.sparseSV){}

double SparseVector::density() const{
	return static_cast<double>(numNonzero())/rbegin()->first;
}

bool SparseVector::operator<(const SparseVector& other) const{
	if(this == &other) return false;
	unsigned thissize = numNonzero(), othersize=other.numNonzero();

	if(thissize!=othersize) // first factor is length of the vector
		return (thissize < othersize);

	for(const_iterator I=begin(),E=end(),Io=other.begin();I!=E;++I,++Io){
		if(I->first!=Io->first)
			return (I->first > Io->first);
		if(I->second!=Io->second)
			return (I->second < Io->second);
	}

	return false;
}


bool keycompare(int idx, const std::pair<int,double> &pair) {
	return (pair.first<idx);
}

double SparseVector::operator[](unsigned idx) const{
	for(SparseVector::const_iterator I=begin(),E=end();I!=E;++I)
		if(I->first==idx)
			return I->second;
	return 0;
}

SparseVector::iterator SparseVector::begin(){ return sparseSV.begin(); }
SparseVector::iterator SparseVector::end(){ return sparseSV.end(); }
SparseVector::const_iterator SparseVector::begin() const{ return sparseSV.begin(); }
SparseVector::const_iterator SparseVector::end() const{	return sparseSV.end(); }

SparseVector::reverse_iterator SparseVector::rbegin(){ return sparseSV.rbegin(); }
SparseVector::reverse_iterator SparseVector::rend(){ return sparseSV.rend(); }
SparseVector::const_reverse_iterator SparseVector::rbegin() const{ return sparseSV.rbegin(); }
SparseVector::const_reverse_iterator SparseVector::rend() const{	return sparseSV.rend(); }

bool SparseVector::operator==(const SparseVector& other) const{
	if(this==&other) return true;
	if(numNonzero()!=other.numNonzero()) return false;
	const_iterator I2=other.begin();
	for(const_iterator I1=begin(),E1=end();I1!=E1;++I1,++I2){
		if(I1->first != I2->first) return false;		// if index differs, SVs are not equal
		if(I1->second != I2->second) return false;		// if value differs, SVs are not equal
	}
	return true;
}
bool SparseVector::operator!=(const SparseVector& other) const{
	return !operator==(other);
}

std::vector<double> SparseVector::dense() const{
	if(numNonzero() == 0) return std::move(std::vector<double>(0));
	std::vector<double> result(rbegin()->first,0.0);
	for(auto& node: *this){
		result[node.first-1]=node.second;
	}
	return std::move(result);
}

void SparseVector::trim(size_t maxlen){
	if(maxlen==0){
		sparseSV.clear();
	}else{
		reverse_iterator I=rbegin();
		while(I->first > maxlen){
			sparseSV.erase(begin()+sparseSV.size()-1);
			I=rbegin();
		}
	}
}

/*************************************************************************************************/

SparseVector SparseVector::operator+(const std::vector<double>& offset) const{
	std::deque<std::pair<unsigned,double>> deque;
	SparseVector::const_iterator I=begin(), E=end();
	unsigned svidx=1;
	for(auto o: offset){
		if(I!=E && I->first==svidx){
			o+=I->second;
			++I;
		}
		if(o) deque.push_back(std::make_pair(svidx,o));
		++svidx;
	}
	for(;I!=E;++I){
		deque.push_back(*I);
	}

	SparseVector::SparseSV svdat(deque.begin(),deque.end());
	SparseVector sv(std::move(svdat));
	return std::move(sv);
}
SparseVector SparseVector::operator+(const SparseVector& offset) const{
	std::deque<std::pair<unsigned,double>> deque;
	SparseVector::const_iterator I1=begin(), I2=offset.begin(), E1=end(), E2=offset.end();

	double sum;
	while(I1!=E1 && I2!=E2){
		if(I1->first==I2->first){
			sum = I1->second + I2->second;
			if(sum!=0) deque.push_back(std::make_pair(I1->first,sum));
			++I1;
			++I2;
		}
		else if(I1->first < I2->first){
			if(I1->second) deque.push_back(*I1);
			++I1;
		}
		else{
			if(I2->second) deque.push_back(*I2);
			++I2;
		}
	}
	for(;I1!=E1;++I1){
		if(I1->second) deque.push_back(*I1);
	}
	for(;I2!=E2;++I2){
		if(I2->second) deque.push_back(*I2);
	}

	SparseVector::SparseSV svdat(deque.begin(),deque.end());
	SparseVector sv(std::move(svdat));
	return std::move(sv);
}

SparseVector SparseVector::operator*(const std::vector<double>& scale) const{
	std::deque<std::pair<unsigned,double>> deque;
	SparseVector::const_iterator I=begin(), E=end();
	unsigned svidx=1;
	for(auto s: scale){
		if(I==E)
			break;
		if(I->first==svidx){
			s*=I->second;
			++I;
			if(s) deque.push_back(std::make_pair(svidx,s));
		}
		// if index!=svidx, product is always zero
		++svidx;
	}
	// indices greater than scale.size() are assumed 0.

	SparseVector::SparseSV svdat(deque.begin(),deque.end());
	SparseVector sv(std::move(svdat));
	return std::move(sv);
}
SparseVector SparseVector::operator*(const SparseVector& scale) const{
	std::deque<std::pair<unsigned,double>> deque;
	SparseVector::const_iterator I1=begin(), I2=scale.begin(), E1=end(), E2=scale.end();

	double product;
	while(I1!=E1 && I2!=E2){
		if(I1->first==I2->first){
			product = I1->second * I2->second;
			if(product!=0) deque.push_back(std::make_pair(I1->first,product));
			++I1;
			++I2;
		}
		else if(I1->first < I2->first) ++I1;
		else ++I2;
	}

	SparseVector::SparseSV svdat(deque.begin(),deque.end());
	SparseVector sv(std::move(svdat));
	return std::move(sv);
}

/*************************************************************************************************/

double InnerProduct(const SparseVector &x, const SparseVector &y){
	double result=0.0;
	SparseVector::const_iterator Ix=x.begin(),Iy=y.begin(),Ex=x.end(),Ey=y.end();
	while(Ix!=Ex && Iy!=Ey){
		if(Ix->first==Iy->first){
			result += Ix->second*Iy->second;
			++Ix;
			++Iy;
		}else if(Ix->first < Iy->first){
			++Ix;
		}else{
			++Iy;
		}
	}
	return result;
}

double InnerProduct(const vector<pair<unsigned,double> > &x, const SparseVector &y){
	double result=0.0;
	vector<pair<unsigned,double> >::const_iterator Ix=x.begin(),Ex=x.end();
	SparseVector::const_iterator Iy=y.begin(),Ey=y.end();
	while(Ix!=Ex && Iy!=Ey){
		if(Ix->first==Iy->first){
			result += Ix->second*Iy->second;
			++Ix;
			++Iy;
		}else if(Ix->first < Iy->first){
			++Ix;
		}else{
			++Iy;
		}
	}
	return result;
}

double InnerProduct(const vector<pair<unsigned,double> > &x, const vector<pair<unsigned,double> > &y){
	double result=0.0;
	vector<pair<unsigned,double> >::const_iterator Ix=x.begin(),Ex=x.end(),Iy=y.begin(),Ey=y.end();
	while(Ix!=Ex && Iy!=Ey){
		if(Ix->first==Iy->first){
			result += Ix->second*Iy->second;
			++Ix;
			++Iy;
		}else if(Ix->first < Iy->first){
			++Ix;
		}else{
			++Iy;
		}
	}
	return result;
}

/*************************************************************************************************/

/**
 * IO FUNCTIONS
 */
std::ostream &operator<<(std::ostream &os, const SparseVector &v){
	os.precision(PRECISION);
	bool first=true;
	for(SparseVector::const_iterator I=v.begin(),E=v.end();I!=E;++I){
		if(!first)
			os << " ";

		if(first)
			first=false;

		os << I->first << ":" << I->second;
	}
	return os;
}

// todo make more efficient
unique_ptr<SparseVector> SparseVector::read(std::istream &iss, bool csv){
	unique_ptr<SparseVector> sv(new SparseVector());

	char junk;

	string line;
	getline(iss,line);
	if(csv)
		std::replace(line.begin(),line.end(),',',' ');

	std::istringstream linestr(line,std::istringstream::in);

	int key;
	double value;
	SVDeque svdeque;

	while(linestr.good()){
		key=-1;
		linestr >> key;

		if(key<0)
			break;

		linestr.get(junk);

		if(junk!=*":" && !linestr.eof()){
			// invalid model file!
			exit_with_err(string("Wrong format, expecting ':' but got '") + junk + "'.");
		}

		linestr >> value;

		svdeque.push_back(std::make_pair(key,value));

//		if(csv){ // if this is a csv-file, discard commas
//			if(linestr.good())
//				linestr.ignore(1,',');
//		}
	}

	sv->sparseSV.reserve(svdeque.size());
	std::copy(svdeque.begin(),svdeque.end(),std::back_inserter(sv->sparseSV));

	return sv;
}


unique_ptr<SparseVector> SparseVector::readCSV(std::istream &iss){
	unique_ptr<SparseVector> v(new SparseVector());
	string line;
	int key=1; // 1-based indexing
	double value;
	SVDeque svdeque;

	while(iss.good()){
		iss >> value;

		if(value!=0.0)
			svdeque.push_back(std::make_pair(key,value));
//			v->sparseSV.push_back(std::make_pair(key,value));

		++key;

		if(!iss.good()) // last value in line
			break;

		iss.ignore(1,','); // ignore comma
	}

	v->sparseSV.reserve(svdeque.size());
	std::copy(svdeque.begin(),svdeque.end(),std::back_inserter(v->sparseSV));

	return v;
}

unique_ptr<SparseVector> SparseVector::readf(std::istream &iss, unsigned format){

	switch(format){
	case FileFormats::DEFAULT:
	{
		return SparseVector::read(iss);
	}
	case FileFormats::CSV:
	{
		return SparseVector::readCSV(iss);
	}
	case FileFormats::SparseCSV:
	{
		return SparseVector::read(iss,true);
	}
	default:
	{
		exit_with_err("Invalid SparseVector format specified.");

		// unreachable
		return unique_ptr<SparseVector>(nullptr);
	}
	}
}

vector<pair<unsigned, double> > SparseVector::toVector() const{
	typedef vector<pair<unsigned, double> > svvector;
	svvector vector(numNonzero());
	svvector::iterator Iv=vector.begin();

	for(const_iterator I=begin(),E=end();I!=E;++I){
		*Iv=std::make_pair(I->first,I->second);
	}

	return vector;
}

/*************************************************************************************************/

void ElementWiseProduct(const vector<pair<unsigned,double> > &x, const SparseVector &y, vector<pair<unsigned,double> > &xy){
	typedef vector<pair<unsigned,double> > svvector;
	std::deque<pair<unsigned, double> > intermediate;

	svvector::const_iterator Ix=x.begin(),Ex=x.end();
	SparseVector::const_iterator Iy=y.begin(),Ey=y.end();

	while(Ix!=Ex && Iy!=Ey){
		if(Ix->first==Iy->first){
			if(Ix->second!=0 && Iy->second!=0)
				intermediate.push_back(std::make_pair(Ix->first,Ix->second*Iy->second));
			++Ix;
			++Iy;
		}
		if(Ix->first < Iy->first)
			++Ix;
		if(Iy->first < Ix->first)
			++Iy;
	}

	xy.resize(intermediate.size());
	std::copy(intermediate.begin(),intermediate.end(),xy.begin());
}

template <typename T>
void ElementWiseProduct(const vector<T> &x, const vector<T> &y, vector<T> &xy){
	unsigned s=(x.size()<y.size()) ? x.size() : y.size();
	xy.resize(s);
	for(unsigned i=0;i<s;++i)
		xy[i]=x[i]*y[i];
}


double squaredNorm(const SparseVector &v){
	double norm=0;
	for(SparseVector::const_iterator I=v.begin(),E=v.end();I!=E;++I)
		norm+=pow(I->second,2);
	return norm;
}
double squaredNorm(const vector<pair<unsigned,double> > &v){
	double norm=0;
	for(vector<pair<unsigned,double> >::const_iterator I=v.begin(),E=v.end();I!=E;++I)
			norm+=pow(I->second,2);
	return norm;
}

/*************************************************************************************************/

} // ensemble namespace

/*************************************************************************************************/

namespace pipeline{
namespace impl{

/*************************************************************************************************/

ensemble::SparseVector Offset(ensemble::SparseVector&& sv, const std::vector<double>& offsets, size_t numoutputs){
	sv = sv + offsets;
	if(numoutputs) sv.trim(numoutputs);
	return std::move(sv);
}

ensemble::SparseVector Scale(ensemble::SparseVector&& sv, const std::vector<double>& scale, size_t numoutputs){
	sv = sv * scale;
	if(numoutputs) sv.trim(numoutputs);
	return std::move(sv);
}

/*************************************************************************************************/

} // pipeline::impl namespace
} // pipeline namespace

/*************************************************************************************************/
