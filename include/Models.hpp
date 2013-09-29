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
 * Model.h
 *
 *      Author: Marc Claesen
 */

/*************************************************************************************************/

#ifndef MODEL_H_
#define MODEL_H_

/*************************************************************************************************/

#include <memory>
#include <deque>
#include <map>
#include <vector>
#include <iostream>

#include "PredicatedFactory.hpp"
#include "SparseVector.hpp"
#include "Kernel.hpp"
#include "svm.h"

/*************************************************************************************************/

using std::unique_ptr;
using std::string;

struct svm_node;

/*************************************************************************************************/

namespace ensemble{

/*************************************************************************************************/

class Prediction{
public:
	typedef string Label;
	typedef double Score;
	typedef std::vector<Score> ScoreCont;
	typedef ScoreCont::iterator iterator;
	typedef ScoreCont::const_iterator const_iterator;

private:
	Label label;
	ScoreCont scores;

public:
	Prediction(unsigned numdecisions);
	Prediction(const string &label, const ScoreCont &scores);
	Prediction(const string &label, ScoreCont&& scores);
	Prediction(string&& label, ScoreCont&& scores);
	Prediction(Prediction&& orig)=default;
	Prediction(const Prediction& o)=default;
	Prediction& operator=(Prediction&& orig)=default;
	Prediction& operator=(const Prediction& orig)=default;
	Prediction()=default;
	~Prediction()=default;
	void setLabel(const string &label);
	void setScore(Score score, unsigned idx);
	Score getScore(unsigned idx) const;
	Label getLabel() const;
	iterator begin();
	iterator end();
	const_iterator begin() const;
	const_iterator end() const;
	Score &operator[](unsigned idx);
	friend std::ostream &operator<<(std::ostream &os, const Prediction &pred);
};

/*************************************************************************************************/

/**
 * General Model class (abstract).
 * All models can be used to make predictions.
 */
class Model{
public:
	typedef string Label;
	typedef double Score;

private:
	Model &operator=(const Model &orig);

protected:
	Model();
	Model(const Model &orig);
public:
	/*
	 * Make prediction with the given Model.
	 */
	virtual Prediction predict(const SparseVector &i) const=0;
	virtual Prediction predict(const std::vector<double> &i) const=0;
	virtual Prediction predict(const struct svm_node *x) const;

	virtual std::vector<double> decision_value(const SparseVector &i) const=0;
	virtual std::vector<double> decision_value(const std::vector<double> &i) const=0;
	virtual std::vector<double> decision_value(const struct svm_node *x) const;


	virtual ~Model(){};

	/**
	 * Attempts to read a Model from <fname>.
	 */
//	static unique_ptr<Model> load(const string&fname);
};

/**
 * General binary model.
 */
class BinaryModel : public Model{
protected:
	BinaryModel();
	BinaryModel(const BinaryModel& orig);
public:
	virtual std::string positive_label() const=0;
	virtual std::string negative_label() const=0;
	virtual size_t num_outputs() const=0;
	virtual ~BinaryModel();

	/**
	 * Constructs a BinaryModel from the given stream.
	 *
	 * If the format does not match, a nullptr is returned.
	 */
	static std::unique_ptr<BinaryModel> deserialize(std::istream& is);

	/**
	 * Attempts to read a Model from <fname>.
	 */
	static unique_ptr<BinaryModel> load(const string&fname);

	virtual void serialize(std::ostream& os) const=0;
	friend std::ostream &operator<<(std::ostream &os, const BinaryModel &model);
};

/**
 * Helper macro to register a concrete binary model class in the factory for reading.
 *
 * After registration, BinaryModel::deserialize() will be able to read given
 * derived model via the derived's static deserialize() method, which must be present.
 * The following signature is required for Derived::deserialize:
 * std::unique_ptr<BinaryModel> deserialize(std::istream& is);
 *
 * In serialization, the derived class must output its name in the first line.
 */
#define REGISTER_BINARYMODEL_HPP(ClassName)			\
struct ClassName##_Registrar{						\
	static bool matches(const std::string& str);	\
	ClassName##_Registrar();						\
};

#define REGISTER_BINARYMODEL_CPP(ClassName)										\
bool ClassName##_Registrar::matches(const std::string& str){					\
	return str.compare(ClassName::NAME)==0;										\
}																				\
ClassName##_Registrar::ClassName##_Registrar(){									\
		PredicatedFactory<ensemble::BinaryModel,const std::string&,std::istream&>	\
		::registerPtr(&ClassName##_Registrar::matches,&ClassName::deserialize);	\
}																				\
ClassName##_Registrar ClassName##_registrar;

// todo document

/**
 * Helper macro to define the class' name within the class. Must be public.
 */
#define REGISTER_BINARYMODEL_IN_CLASS(ClassName) 				\
static constexpr const char* NAME=#ClassName;					\
static unique_ptr<BinaryModel> deserialize(std::istream& is);

/*************************************************************************************************/

class SVMEnsemble;
class SVMEnsembleImpl;

/**
 * General class for Support Vector Machine models from different libraries.
 */
class SVMModel : public BinaryModel{
public:
	typedef std::vector<std::shared_ptr<SparseVector>> SV_container;	// used to contain ptrs to SVs
	typedef SV_container::iterator iterator;							// iterators
	typedef SV_container::const_iterator const_iterator;
	typedef std::vector<double> Weights; 								// contains alpha values (SV weights)
	typedef Weights::iterator weight_iter;
	typedef Weights::const_iterator const_weight_iter;
	typedef std::vector<std::pair<std::string,unsigned> > Classes; // Classes[i] contains [label,numSV] for class i

private:
	SVMModel &operator=(const SVMModel &orig);
	unsigned getStartOfClass(unsigned classidx) const;

protected:
	const SVMEnsemble *ens;

	// support vectors
	SV_container SVs;

	// SV weights: alpha_i*y_i
	Weights weights;

	// classes[i] contains label and amount of SVs for class i in <weights>.
	// i:1..n sum(classes[i,2]) == weights.size()
	Classes classes;

	// constants in decision functions
	std::vector<double> constants;

	// kernel parameters
	Kernel *kernel;

	/**
	 * Do predictions with cached kernel evaluations.
	 */
	virtual double predict_by_cache(const std::vector<double> &cache) const;

	/**
	 * Based on LibSVM prediction implementation.
	 */
	double svm_predict_values(const std::vector<double> &kernelevals) const;

public:
	/**
	 * Constructs a new SVMModel with given support vectors, SV weights, class definitions and kernel parameters.
	 *
	 * The new model acquires ownership of all its arguments. Pointers used for large arguments to avoid copying.
	 */
	SVMModel(SV_container&& SVs, Weights&& weights, Classes&& classes, std::vector<double>&& constants, unique_ptr<Kernel> kernel);

	/**
	 * Constructs a new SVMModel with given support vectors, SV weights, class definitions and kernel parameters.
	 *
	 * The new model acquires ownership of all its arguments except <ens>.
	 * Kernel is directed to the kernel in <ens>.
	 * Pointers used for large arguments to avoid copying.
	 */
	SVMModel(SV_container&& SVs, Weights&& weights, Classes&& classes, std::vector<double>&& constants, const SVMEnsemble *ens);
	SVMModel(const SVMModel &m);
	SVMModel(SVMModel&& m);

	virtual Prediction predict(const SparseVector &i) const override;
	virtual Prediction predict(const std::vector<double> &i) const override;

	virtual std::vector<double> decision_value(const SparseVector &i) const override;
	virtual std::vector<double> decision_value(const std::vector<double> &i) const override;

	virtual size_t num_outputs() const override;

	/**
	 * Iterators over SVs.
	 */
	iterator begin();
	iterator end();
	const_iterator begin() const;
	const_iterator end() const;

	/**
	 * Iterators over SVs of specific class.
	 */
	iterator begin(unsigned classidx);
	iterator end(unsigned classidx);
	const_iterator begin(unsigned classidx) const;
	const_iterator end(unsigned classidx) const;

	/**
	 * Iterators over all weights in this model.
	 *
	 * Can iterate over getNumSV()*(getNumClasses()-1) weights.
	 */
	weight_iter weight_begin();
	weight_iter weight_end();
	const_weight_iter weight_begin() const;
	const_weight_iter weight_end() const;

	/**
	 * Iterators over weights for specific decision function.
	 *
	 * Each iterator can iterate over getNumSV() weights.
	 * A model for k classes has k-1 decision functions, index starting at 0.
	 */
	weight_iter weight_begin(unsigned decfunidx);
	weight_iter weight_end(unsigned decfunidx);
	const_weight_iter weight_begin(unsigned decfunidx) const;
	const_weight_iter weight_end(unsigned decfunidx) const;

	/**
	 * Returns the size of the model, ie. the amount of SVs.
	 */
	size_t size() const;

	/**
	 * Returns the amount of classes in the model.
	 */
	unsigned getNumClasses() const;

	/**
	 * Returns the amount of SV for class i.
	 */
	unsigned getNumSV(unsigned i) const;

	/**
	 * Returns the label of class <i>.
	 */
	string getLabel(unsigned i) const;

	virtual std::string positive_label() const override;
	virtual std::string negative_label() const override;

	/**
	 * Replaces the label <current> by <replacement>. Used to update LIBSVM unsigned labels.
	 */
	void updateLabel(const std::string &current, const std::string &replacement);

	/**
	 * Returns the constants used in decision functions ('b').
	 */
	const std::vector<double> &getConstants() const;
	double getConstant(unsigned i) const;

	virtual const SparseVector& operator[](int idx) const;

	/**
	 * Redirects the support vector ptr at idx to newtarget.
	 * Errors when *newtarget!=SVMModel[idx].
	 */
	void redirectSV(iterator &I, std::shared_ptr<SparseVector> newtarget);

	/**
	 * Terminates the model. If the model is not part of an SVMEnsemble, all SVs inside are purged from memory.
	 */
	virtual ~SVMModel();

	static unique_ptr<SVMModel> read(std::istream &is, SVMEnsemble *ens=nullptr);

	virtual const Kernel *getKernel() const;
	friend class SVMEnsembleImpl;

	virtual void serialize(std::ostream& os) const override;
	friend std::ostream &operator<<(std::ostream &os, const SVMModel &model);

	static unique_ptr<SVMModel> load(const string&fname);

	REGISTER_BINARYMODEL_IN_CLASS(SVMModel)

};

REGISTER_BINARYMODEL_HPP(SVMModel)

/*************************************************************************************************/

} // ensemble namespace

/*************************************************************************************************/

#endif /* MODEL_H_ */
