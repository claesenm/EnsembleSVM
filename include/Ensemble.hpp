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
 * Ensemble.h
 *
 *      Author: Marc Claesen
 */

#ifndef ENSEMBLE_H_
#define ENSEMBLE_H_

/*************************************************************************************************/

//#include <set>
//#include <vector>
//#include <deque>
//#include <map>
#include <memory>
//#include <list>
//#include <cassert>
#include <iostream>
#include "Models.hpp"
#include "Kernel.hpp"
#include "config.h"
#include "any_iterator/any_iterator.hpp"

/*************************************************************************************************/

//#define CACHE_THRESHOLD 1

using std::unique_ptr;
using std::string;
using namespace ensemble;
using IteratorTypeErasure::any_iterator;
using IteratorTypeErasure::make_any_iterator_type;

/*************************************************************************************************/

namespace ensemble{

/*************************************************************************************************/

class Ensemble : public BinaryModel {
private:
	Ensemble &operator=(const Ensemble &e);
public:
	Ensemble(const Ensemble &e);
	Ensemble();

	virtual ~Ensemble(){};
};

/*************************************************************************************************/

class SVSort{
public:
	bool operator()(const SparseVector *v1, const SparseVector *v2) const{
		return (*v1)<(*v2);
	}
};

/*************************************************************************************************/

class SVMEnsembleImpl;

class SVMEnsemble final : public Ensemble{
private:
	SVMEnsemble &operator=(const SVMEnsemble &e);

public:

	typedef make_any_iterator_type<
			std::deque<std::shared_ptr<SparseVector>>::iterator
			>::type sv_iterator;
	typedef make_any_iterator_type<
			std::deque<std::shared_ptr<SparseVector>>::const_iterator
			>::type sv_const_iterator;
	typedef make_any_iterator_type<
			std::map<SVMModel*,int>::iterator
			>::type iterator;
	typedef make_any_iterator_type<
			std::map<SVMModel*,int>::const_iterator
			>::type const_iterator;

	typedef std::map< std::string, std::string > LabelMap;

private:

	std::unique_ptr<SVMEnsembleImpl> pImpl;

	SVMEnsemble(unique_ptr<SVMEnsembleImpl> impl);

public:
	/**
	 * Constructs an SVMEnsemble using <kernel>. The newly created object acquires ownership of <kernel>.
	 */
	SVMEnsemble(unique_ptr<Kernel> kernel);
	SVMEnsemble(unique_ptr<Kernel> kernel, const LabelMap &labelmap);
	SVMEnsemble(std::vector<unique_ptr<SVMModel>>&& models);
	SVMEnsemble(std::vector<unique_ptr<SVMModel>>&& models,
			const std::string& positive, const std::string& negative);

	/**
	 * Ensemble prediction: returns the predicted label and scores.
	 *
	 * First score is the ensemble consensus for the given label.
	 * Subsequent scores are decision values per base model.
	 */
	virtual Prediction predict(const SparseVector &i) const override;

	/**
	 * Dense prediction.
	 */
	virtual Prediction predict(const std::vector<double> &v) const override;

	/**
	 * Returns the base model decision values for prediction of the test instance.
	 */
	virtual std::vector<double> decision_value(const SparseVector &i) const override final;
	virtual std::vector<double> decision_value(const std::vector<double> &i) const override final;

	// Adds SVM model *m to the SVMEnsemble.
	virtual void add(std::unique_ptr<SVMModel> m);

	unsigned getSVindex(unsigned ensembleidx) const;
	unsigned getSVindex(unsigned localidx, const SVMModel * const mod) const;

	/**
	 * Returns the SV at <ensidx> within the ensemble.
	 */
	std::shared_ptr<SparseVector> getSV(unsigned ensidx);
	const SparseVector* getSV(unsigned ensidx) const;

	iterator begin();
	iterator end();
	const_iterator begin() const;
	const_iterator end() const;

	sv_iterator sv_begin();
	sv_iterator sv_end();
	sv_const_iterator sv_begin() const;
	sv_const_iterator sv_end() const;

	size_t size() const;
	size_t numDistinctSV() const;
	size_t numTotalSV() const;

	virtual ~SVMEnsemble();

	const Kernel *getKernel() const;

	virtual void printSV(std::ostream &os, int SVidx) const;
	friend std::ostream &operator<<(std::ostream &os, const SVMEnsemble &v);
	friend class SVMEnsembleImpl;

	virtual void serialize(std::ostream& os) const override;
	static unique_ptr<SVMEnsemble> read(std::istream &iss);
	static unique_ptr<SVMEnsemble> load(const string &fname);

	double density() const;

	/**
	 * Translate specified base-model label to an output label.
	 */
	std::string translate(const std::string &label) const;

	virtual std::string positive_label() const override;
	virtual std::string negative_label() const override;
	virtual size_t num_outputs() const override;

	REGISTER_BINARYMODEL_IN_CLASS(SVMEnsemble)
};

REGISTER_BINARYMODEL_HPP(SVMEnsemble)

/*************************************************************************************************/

} // ensemble namespace

/*************************************************************************************************/

#endif /* ENSEMBLE_H_ */
