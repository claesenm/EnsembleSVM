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
 * Kernel.cpp
 *
 *      Author: Marc Claesen
 */

#include "Kernel.hpp"
#include "Util.hpp"
#include <cassert>
#include <math.h>
#include <cstring>
#include <math.h>
#include <typeinfo>
#include <string>
#include <iostream>
#include <sstream>
#include <iterator>
#include <numeric>
#include <cmath>

using std::string;

using std::endl;

namespace {

// todo
std::string LINEAR_STR{"linear"};
std::string POLY_STR{"polynomial"};
std::string RBF_STR{"rbf"};
std::string SIGMOID_STR{"sigmoid"};
std::string USERDEF_STR{"userdef"};

}


namespace ensemble{

// KERNEL FUNCTIONS
Kernel::Kernel(unsigned type):type(type){}
Kernel::~Kernel(){}
unsigned Kernel::getType() const{ return type; }
bool Kernel::operator==(const Kernel &other) const{
	return (typeid(*this)==typeid(other));
}
bool Kernel::operator!=(const Kernel &other) const{
	return !(operator ==(other));
}

// LINEARKERNEL FUNCTIONS
LinearKernel::LinearKernel():Kernel(KERNEL_TYPES::LINEAR){}
double LinearKernel::k_function(const SparseVector *x, const SparseVector *y) const{
	return InnerProduct(*x,*y);
}
double LinearKernel::k_function(const_iterator Ix, const_iterator Ex, const_iterator Iy, const_iterator Ey) const{
	if(distance(Ix,Ex) <= distance(Iy,Ey))
		return std::inner_product(Ix,Ex,Iy,0.0);
	return std::inner_product(Iy,Ey,Ix,0.0);
}
unique_ptr<Kernel> LinearKernel::clone() const{
	unique_ptr<Kernel> ptr(static_cast<Kernel*>(new LinearKernel()));
	return ptr;
}
LinearKernel::~LinearKernel(){}
bool LinearKernel::operator==(const Kernel &other) const{ return Kernel::operator==(other); }
bool LinearKernel::operator==(const LinearKernel &other) const{	return true; }

// POLYNOMIAL KERNEL (gamma*u'*v+coef0)^degree
PolyKernel::PolyKernel(unsigned degree, double coef0, double gamma):Kernel(KERNEL_TYPES::POLY),degree(degree),coef0(coef0),gamma(gamma){}
PolyKernel::PolyKernel(const PolyKernel &orig):Kernel(KERNEL_TYPES::POLY),degree(orig.getDegree()),coef0(orig.getCoef()),gamma(orig.getGamma()){}
unsigned PolyKernel::getDegree() const{ return degree; }
double PolyKernel::getCoef() const{ return coef0; }
double PolyKernel::getGamma() const{ return gamma; }
PolyKernel::~PolyKernel(){}
unique_ptr<Kernel> PolyKernel::clone() const{
	Kernel *kernel=new PolyKernel(getDegree(),getCoef(),getGamma());
	unique_ptr<Kernel> ptr(kernel);
	return ptr;
}
double PolyKernel::k_function(const SparseVector *x, const SparseVector *y) const{
	//	return powi(gamma*InnerProduct(*x,*y)+coef0,degree); // LibSVM code
	return pow(getGamma()*InnerProduct(*x,*y)+getCoef(),getDegree());
}
double PolyKernel::k_function(const_iterator Ix, const_iterator Ex, const_iterator Iy, const_iterator Ey) const{
	if(distance(Ix,Ex) <= distance(Iy,Ey))
		return pow(getGamma()*std::inner_product(Ix,Ex,Iy,0.0)+getCoef(),getDegree());
	return pow(getGamma()*std::inner_product(Iy,Ey,Ix,0.0)+getCoef(),getDegree());
}
bool PolyKernel::operator==(const PolyKernel &other) const{
	return (getGamma()==other.getGamma() && getCoef()==other.getCoef() && getDegree()==other.getDegree());
}
bool PolyKernel::operator==(const Kernel &other) const{
	if(!Kernel::operator ==(other))
		return false;

	const PolyKernel *casted = static_cast<const PolyKernel*>(&other); // will never fail
	return operator==(*casted);
}


// RBF KERNEL exp(-gamma*|u-v|^2)
RBFKernel::RBFKernel(double gamma):Kernel(KERNEL_TYPES::RBF),gamma(gamma){}
RBFKernel::RBFKernel(const RBFKernel &orig):Kernel(KERNEL_TYPES::RBF),gamma(orig.getGamma()){}
double RBFKernel::getGamma() const{ return gamma; }
RBFKernel::~RBFKernel(){}
unique_ptr<Kernel> RBFKernel::clone() const{
	Kernel *kernel=new RBFKernel(getGamma());
	unique_ptr<Kernel> ptr(kernel);
	return ptr;
}
double RBFKernel::k_function(const SparseVector *x, const SparseVector *y) const{
	double sum = 0;
	SparseVector::const_iterator Ix=x->begin(),Ex=x->end(),Iy=y->begin(),Ey=y->end();
	while(Ix!=Ex && Iy!=Ey){
		if(Ix->first == Iy->first){
			double d = Ix->second - Iy->second;
			sum += d*d;
			++Ix;
			++Iy;
		}else{
			if(Ix->first > Iy->first){
				sum += Iy->second * Iy->second;
				++Iy;
			}else{
				sum += Ix->second * Ix->second;
				++Ix;
			}
		}
	}

	while(Ix!=Ex){
		sum += Ix->second * Ix->second;
		++Ix;
	}

	while(Iy!=Ey){
		sum += Iy->second * Iy->second;
		++Iy;
	}

	return exp(-getGamma()*sum);
}
double RBFKernel::k_function(const_iterator Ix, const_iterator Ex, const_iterator Iy, const_iterator Ey) const{

	size_t distx = distance(Ix,Ex), disty = distance(Iy,Ey);
	if(distx == disty)
		return exp(-gamma*(std::inner_product(Ix,Ex,Iy,0.0,std::plus<double>(),
						[](double x, double y){ return std::pow(x-y,2); })));
	if(distx < disty){
		// used if length of x and y differ
		double nonoverlap=std::accumulate(Iy+distx,Ey,0.0,
				[](double x, double y){ return x+y*y;});

		// rbf function
		return exp(-gamma*(std::inner_product(Ix,Ex,Iy,nonoverlap,std::plus<double>(),
				[](double x, double y){ return std::pow(x-y,2); })));
	}

	// used if length of x and y differ
	double nonoverlap=std::accumulate(Ix+disty,Ex,0.0,
			[](double x, double y){ return x+y*y;});

	// rbf function
	return exp(-gamma*std::inner_product(Iy,Ey,Ix,nonoverlap,std::plus<double>(),
			[](double x, double y){ return std::pow(x-y,2); }));
}
bool RBFKernel::operator==(const RBFKernel &other) const{
	return (getGamma()==other.getGamma());
}
bool RBFKernel::operator==(const Kernel &other) const{
	if(!Kernel::operator ==(other))
		return false;

	const RBFKernel *casted = static_cast<const RBFKernel*>(&other); // will never fail
	return operator==(*casted);
}

// SIGMOID KERNEL
SigmoidKernel::SigmoidKernel(double coef0, double gamma):Kernel(KERNEL_TYPES::SIGMOID),coef0(coef0),gamma(gamma){}
SigmoidKernel::SigmoidKernel(const SigmoidKernel &orig):Kernel(KERNEL_TYPES::SIGMOID),coef0(orig.getCoef()),gamma(orig.getGamma()){}
double SigmoidKernel::getCoef() const{ return coef0; }
double SigmoidKernel::getGamma() const{ return gamma; }
SigmoidKernel::~SigmoidKernel(){}
unique_ptr<Kernel> SigmoidKernel::clone() const{
	Kernel *kernel=new SigmoidKernel(getCoef(),getGamma());
	unique_ptr<Kernel> ptr(kernel);
	return ptr;
}
double SigmoidKernel::k_function(const SparseVector *x, const SparseVector *y) const{
	return tanh(getGamma()*InnerProduct(*x,*y)+getCoef());
}
double SigmoidKernel::k_function(const_iterator Ix, const_iterator Ex, const_iterator Iy, const_iterator Ey) const{
	if(distance(Ix,Ex) <= distance(Iy,Ey))
		return tanh(getGamma()*std::inner_product(Ix,Ex,Iy,0.0)+getCoef());
	return tanh(getGamma()*std::inner_product(Iy,Ey,Ix,0.0)+getCoef());
}
bool SigmoidKernel::operator==(const SigmoidKernel &other) const{
	return (getGamma()==other.getGamma() && getCoef()==other.getCoef());
}
bool SigmoidKernel::operator==(const Kernel &other) const{
	if(!Kernel::operator ==(other))
		return false;

	const SigmoidKernel *casted = static_cast<const SigmoidKernel*>(&other); // will never fail
	return operator==(*casted);
}

// PRECOMPUTED KERNEL // todo
UserdefKernel::UserdefKernel():Kernel(KERNEL_TYPES::USERDEF){}
UserdefKernel::UserdefKernel(const UserdefKernel &orig):Kernel(KERNEL_TYPES::USERDEF){}
UserdefKernel::~UserdefKernel(){}
unique_ptr<Kernel> UserdefKernel::clone() const{
	Kernel *kernel=new UserdefKernel();
	unique_ptr<Kernel> ptr(kernel);
	return ptr;
}
double UserdefKernel::k_function(const SparseVector *x, const SparseVector *y) const{
	return 0; // todo
}
double UserdefKernel::k_function(const_iterator Ix, const_iterator Ex, const_iterator Iy, const_iterator Ey) const{
	return 0; // todo
}
bool UserdefKernel::operator==(const UserdefKernel &other) const{
	return false; // todo
}
bool UserdefKernel::operator==(const Kernel &other) const{
	if(!Kernel::operator ==(other))
		return false;

	const UserdefKernel *casted = static_cast<const UserdefKernel*>(&other); // will never fail
	return operator==(*casted);
}

// todo
unique_ptr<Kernel> KernelFactory(unsigned kfun, unsigned degree, double gamma, double coef0){
	switch(kfun){
	case KERNEL_TYPES::LINEAR:
	{
		return unique_ptr<Kernel>(new LinearKernel());
	}
	case KERNEL_TYPES::POLY:
	{
		return unique_ptr<Kernel>(new PolyKernel(degree,coef0,gamma));
	}
	case KERNEL_TYPES::RBF:
	{
		return unique_ptr<Kernel>(new RBFKernel(gamma));
	}
	case KERNEL_TYPES::SIGMOID:
	{
		return unique_ptr<Kernel>(new SigmoidKernel(coef0,gamma));
	}
	case KERNEL_TYPES::USERDEF:
	{
		break; // todo
	}
	default:
	{
		exit_with_err("Invalid kernel type specified.");
		break;
	}
	}

	// unreachable
	return unique_ptr<Kernel>(nullptr);
}


/**
 * IO FUNCTIONS
 */
/**
 * KERNEL FUNCTIONS
 */
// base Kernel class
void Kernel::print(std::ostream &os) const{	os << "kernel_type " << type << endl; }
unique_ptr<Kernel> Kernel::read(std::istream &is){
	unique_ptr<Kernel> kernel;

	string line, keyword;
	getline(is,line);
	std::istringstream linestr(line,std::istringstream::in);

	// read first line to get kernel type
	unsigned kernel_type;
	linestr >> keyword;
	if(keyword.compare("kernel_type")!=0)
		exit_with_err(std::string("Invalid kernel format: expecting kernel_type, got: ") + line);
	linestr >> kernel_type;

	// defer to specific kernel readers
	switch(kernel_type){
	case KERNEL_TYPES::LINEAR:
	{
		unique_ptr<LinearKernel> derived=LinearKernel::read(is);
		Kernel *kernelptr=derived.get();
		derived.release();
		kernel=unique_ptr<Kernel>(kernelptr);
		break;
	}
	case KERNEL_TYPES::POLY:
	{
		unique_ptr<PolyKernel> derived=PolyKernel::read(is);
		Kernel *kernelptr=derived.get();
		derived.release();
		kernel=unique_ptr<Kernel>(kernelptr);
		break;
	}
	case KERNEL_TYPES::RBF:
	{
		unique_ptr<RBFKernel> derived=RBFKernel::read(is);
		Kernel *kernelptr=derived.get();
		derived.release();
		kernel=unique_ptr<Kernel>(kernelptr);
		break;
	}
	case KERNEL_TYPES::SIGMOID:
	{
		unique_ptr<SigmoidKernel> derived=SigmoidKernel::read(is);
		Kernel *kernelptr=derived.get();
		derived.release();
		kernel=unique_ptr<Kernel>(kernelptr);
		break;
	}
	case KERNEL_TYPES::USERDEF:
	{
		unique_ptr<UserdefKernel> derived=UserdefKernel::read(is);
		Kernel *kernelptr=derived.get();
		derived.release();
		kernel=unique_ptr<Kernel>(kernelptr);
		break;
	}
	default:
		exit_with_err("Invalid kernel type!");
		break;
	}
	return kernel;
}

// LinearKernel
void LinearKernel::print(std::ostream &os) const{ Kernel::print(os); }
unique_ptr<LinearKernel> LinearKernel::read(std::istream &is){
	return unique_ptr<LinearKernel>(new LinearKernel());
}

// PolyKernel
void PolyKernel::print(std::ostream &os) const{
	Kernel::print(os);
	os << "degree " << getDegree() << endl;
	os << "coef0 " << getCoef() << endl;
	os << "gamma " << getGamma() << endl;
}
unique_ptr<PolyKernel> PolyKernel::read(std::istream &is){
	unique_ptr<PolyKernel> kernel;
	string line, keyword;

	// read degree
	getline(is,line);
	std::istringstream linestr(line,std::istringstream::in);
	linestr.str(line);
	unsigned degree;
	linestr >> keyword;
	if(keyword.compare("degree")!=0)
		exit_with_err("Invalid polynomial kernel: expecting degree.");
	linestr >> degree;
	linestr.clear();

	// read coef0
	getline(is,line);
	linestr.str(line);
	double coef0;
	linestr >> keyword;
	if(keyword.compare("coef0")!=0)
		exit_with_err("Invalid polynomial kernel: expecting coef0.");
	linestr >> coef0;
	linestr.clear();

	// read gamma
	getline(is,line);
	linestr.str(line);
	double gamma;
	linestr >> keyword;
	if(keyword.compare("gamma")!=0)
		exit_with_err("Invalid polynomial kernel: expecting gamma.");
	linestr >> gamma;

	kernel=unique_ptr<PolyKernel>(new PolyKernel(degree,coef0,gamma));
	return kernel;
}

// RBFKernel
void RBFKernel::print(std::ostream &os) const{
	Kernel::print(os);
	os << "gamma " << getGamma() << endl;
}
unique_ptr<RBFKernel> RBFKernel::read(std::istream &is){
	unique_ptr<RBFKernel> kernel;
	string line, keyword;

	// read gamma
	getline(is,line);
	std::istringstream linestr(line,std::istringstream::in);
	double gamma;
	linestr >> keyword;
	if(keyword.compare("gamma")!=0)
		exit_with_err("Invalid RBF kernel: expecting gamma.");
	linestr >> gamma;
	linestr.clear();

	kernel=unique_ptr<RBFKernel>(new RBFKernel(gamma));
	return kernel;
}

// SigmoidKernel
void SigmoidKernel::print(std::ostream &os) const{
	Kernel::print(os);
	os << "coef0 " << getCoef() << endl;
	os << "gamma " << getGamma() << endl;
}
unique_ptr<SigmoidKernel> SigmoidKernel::read(std::istream &is){
	unique_ptr<SigmoidKernel> kernel;
	string line, keyword;

	// read coef0
	getline(is,line);
	std::istringstream linestr(line,std::istringstream::in);
	double coef0;
	linestr >> keyword;
	if(keyword.compare("coef0")!=0)
		exit_with_err("Invalid sigmoid kernel: expecting coef0.");
	linestr >> coef0;
	linestr.clear();

	// read gamma
	getline(is,line);
	linestr.str(line);
	double gamma;
	linestr >> keyword;
	if(keyword.compare("gamma")!=0)
		exit_with_err("Invalid sigmoid kernel: expecting gamma.");
	linestr >> gamma;

	kernel=unique_ptr<SigmoidKernel>(new SigmoidKernel(coef0,gamma));
	return kernel;
}

// UserdefKernel
//todo
void UserdefKernel::print(std::ostream &os) const{
	// todo
}
unique_ptr<UserdefKernel> UserdefKernel::read(std::istream &is){
	unique_ptr<UserdefKernel> ptr(nullptr); // todo
	return ptr;
}



} // ensemble namespace
