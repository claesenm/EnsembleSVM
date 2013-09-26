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
 * Kernel.h
 *
 *      Author: Marc Claesen
 */

/*************************************************************************************************/

#ifndef KERNEL_H_
#define KERNEL_H_

/*************************************************************************************************/

#include <memory>
#include "SparseVector.hpp"

using std::unique_ptr;

/*************************************************************************************************/

namespace ensemble{

/*************************************************************************************************/

struct KERNEL_TYPES{
	enum {
		LINEAR = 0,
		POLY = 1,
		RBF = 2,
		SIGMOID = 3,
		USERDEF = 4
	};
};

/*************************************************************************************************/

/**
 * Class to model a kernel. Specialized subclasses define common kernels.
 */
class Kernel{
private:
	unsigned type;

	// prevent slicing
	Kernel& operator=(const Kernel &orig)=delete;
	Kernel& operator=(Kernel&& orig)=delete;

protected:
	Kernel(unsigned type);

	/**
	 * Prints this Kernel to <os>.
	 *
	 * Used to achieve virtual operator<<.
	 */
	virtual void print(std::ostream &os) const=0;

public:
	typedef std::vector<double>::const_iterator const_iterator;
	/**
	 * Computes <*x,*y> using the current kernel.
	 */
	virtual double k_function(const SparseVector *x, const SparseVector *y) const=0;
	virtual double k_function(const_iterator Ix, const_iterator Ex, const_iterator Iy, const_iterator Ey) const=0;

	/**
	 * Returns the type of this kernel as defined by the enum in this header.
	 */
	unsigned getType() const;

	/**
	 * Clones this Kernel.
	 */
	virtual unique_ptr<Kernel> clone() const=0;

	/**
	 * Destructor.
	 */
	virtual ~Kernel();

	/**
	 * Writes this Kernel to <os>.
	 */
	friend std::ostream &operator<<(std::ostream &os, const Kernel &k);

	/**
	 * Reads a Kernel from <is>. Reading must start at a line of the form 'kernel_type X'.
	 *
	 * Implemented in io.cpp.
	 */
	static unique_ptr<Kernel> read(std::istream &is);

	virtual bool operator==(const Kernel &other) const=0;
	bool operator!=(const Kernel &other) const;
};

inline std::ostream& operator<< (std::ostream &os, const Kernel &kernel){
	kernel.print(os); // delegate the work to a polymorphic member function.
	return os;
}

/*************************************************************************************************/

/**
 * Class to model the linear kernel u'*v.
 */
class LinearKernel:public Kernel{
protected:
	virtual void print(std::ostream &os) const;

	/**
	 * Reads a LinearKernel from <is>.
	 *
	 * Reading must begin at the first line below 'kernel_type X'.
	 * Reads no additional information as linear kernels have no parameters. Cannot fail.
	 */
	static unique_ptr<LinearKernel> read(std::istream &is);

public:
	LinearKernel();
	LinearKernel(const LinearKernel &orig);
	double k_function(const SparseVector *x, const SparseVector *y) const;
	double k_function(const_iterator Ix, const_iterator Ex, const_iterator Iy, const_iterator Ey) const;
	virtual unique_ptr<Kernel> clone() const;
	virtual ~LinearKernel();
	virtual bool operator==(const Kernel &other) const;
	bool operator==(const LinearKernel &other) const;

	friend class Kernel;
};

/*************************************************************************************************/

/**
 * Class to model polynomial kernels.
 *
 * These kernels are defined by the parameters degree, coef0 and gamma:
 * 		(gamma*u'*v+coef0)^degree
 */
class PolyKernel:public Kernel{
private:
	unsigned degree;
	double coef0;
	double gamma;

protected:
	virtual void print(std::ostream &os) const;

	/**
	 * Reads a PolyKernel from <is>.
	 *
	 * Reading must begin at the first line below 'kernel_type X'.
	 * Input must be of the following form:
	 * 		degree <unsigned>\n
	 * 		coef0 <double>\n
	 * 		gamma <double>\n
	 */
	static unique_ptr<PolyKernel> read(std::istream &is);

public:
	PolyKernel(unsigned degree, double coef0, double gamma);
	PolyKernel(const PolyKernel &orig);
	double k_function(const SparseVector *x, const SparseVector *y) const;
	double k_function(const_iterator Ix, const_iterator Ex, const_iterator Iy, const_iterator Ey) const;

	virtual bool operator==(const Kernel &other) const;
	bool operator==(const PolyKernel &other) const;

	// accessor functions
	double getCoef() const;
	unsigned getDegree() const;
	double getGamma() const;

	// clone
	virtual unique_ptr<Kernel> clone() const;

	// destructor
	virtual ~PolyKernel();

	friend class Kernel;
};

/*************************************************************************************************/

/**
 * Class to model Radial Basis Function kernels.
 *
 * These kernels are defined by the parameter gamma:
 * 		exp(-gamma*|u-v|^2)
 */
class RBFKernel:public Kernel{
private:
	double gamma;

protected:
	/**
	 * Polymorphic print function.
	 *
	 * Implemented in io.cpp.
	 */
	virtual void print(std::ostream &os) const;

	/**
	 * Reads a RBFKernel from <is>.
	 *
	 * Reading must begin at the first line below 'kernel_type X'.
	 */
	static unique_ptr<RBFKernel> read(std::istream &is);

public:
	RBFKernel(double gamma);
	RBFKernel(const RBFKernel &orig);
	double k_function(const SparseVector *x, const SparseVector *y) const;
	double k_function(const_iterator Ix, const_iterator Ex, const_iterator Iy, const_iterator Ey) const;

	virtual bool operator==(const Kernel &other) const;
	bool operator==(const RBFKernel &other) const;

	// accessor functions
	double getGamma() const;

	// clone
	virtual unique_ptr<Kernel> clone() const;

	// destructor
	virtual ~RBFKernel();

	friend class Kernel;
};

/*************************************************************************************************/

/**
 * Class to model sigmoid kernels.
 *
 * These kernels are defined by the parameters coef0 and gamma:
 * 		tanh(gamma*u'*v+coef0)
 */
class SigmoidKernel:public Kernel{
private:
	double coef0;
	double gamma;

protected:
	virtual void print(std::ostream &os) const;

	/**
	 * Reads a SigmoidKernel from <is>.
	 *
	 * Reading must begin at the first line below 'kernel_type X'.
	 */
	static unique_ptr<SigmoidKernel> read(std::istream &is);

public:
	SigmoidKernel(double coef0, double gamma);
	SigmoidKernel(const SigmoidKernel &orig);
	double k_function(const SparseVector *x, const SparseVector *y) const;
	double k_function(const_iterator Ix, const_iterator Ex, const_iterator Iy, const_iterator Ey) const;

	bool operator==(const SigmoidKernel &other) const;
	virtual bool operator==(const Kernel &other) const;

	// accessor functions
	double getCoef() const;
	double getGamma() const;

	// clone
	virtual unique_ptr<Kernel> clone() const;

	// destructor
	virtual ~SigmoidKernel();

	friend class Kernel;
};

/*************************************************************************************************/

/**
 * Class to model user-defined kernels.
 *
 * todo implement
 */
class UserdefKernel:public Kernel{
protected:
	virtual void print(std::ostream &os) const;

	/**
	 * Reads a UserdefKernel from <is>.
	 *
	 * Reading must begin at the first line below 'kernel_type X'.
	 */
	static unique_ptr<UserdefKernel> read(std::istream &is);

public:
	UserdefKernel();
	UserdefKernel(const UserdefKernel &orig);
	double k_function(const SparseVector *x, const SparseVector *y) const;
	double k_function(const_iterator Ix, const_iterator Ex, const_iterator Iy, const_iterator Ey) const;
	virtual bool operator==(const Kernel &other) const;
	bool operator==(const UserdefKernel &other) const;
	virtual unique_ptr<Kernel> clone() const;
	virtual ~UserdefKernel();

	friend class Kernel;
};

/*************************************************************************************************/

unique_ptr<Kernel> KernelFactory(unsigned kfun, unsigned degree, double gamma, double coef0);

/*************************************************************************************************/

} // ensemble namespace

/*************************************************************************************************/

#endif /* KERNEL_H_ */
