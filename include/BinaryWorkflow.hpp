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
 * BinaryWorkflow.hpp
 *
 *      Author: Marc Claesen
 */

#ifndef WORKFLOW_HPP_
#define WORKFLOW_HPP_

/*************************************************************************************************/

#include "SparseVector.hpp"
#include "Models.hpp"
#include "Ensemble.hpp"
#include "pipeline/pipelines.hpp"
#include <vector>
#include <memory>

using ensemble::pipeline::MultistagePipe;

/*************************************************************************************************/

namespace ensemble{

/*************************************************************************************************/

/**
 * Class to model a generic workflow for a binary classifier.
 *
 * We follow the strategy pattern, with the following configurable steps:
 * 1. Preprocessing: SparseVector -> SparseVector.
 * 		- ex: scaling, bias correction, ...
 * 2. BinaryModel: SparseVector -> std::vector<double> (decision value(s)).
 * 		- ex: SVM model, SVM ensemble
 * 3. Postprocessing: std::vector<double> -> double (global decision value).
 * 		- ex: ensemble aggregation
 * 4. Thresholding to obtain binary (string) label.
 */
class BinaryWorkflow final : public BinaryModel{
public:
	typedef std::vector<double> Vector;
	typedef MultistagePipe<SparseVector(SparseVector)> Preprocessing;
	typedef MultistagePipe<double(Vector)> Postprocessing;

protected:
	std::unique_ptr<Preprocessing> preprocessing;
	std::unique_ptr<BinaryModel> predictor;
	std::unique_ptr<Postprocessing> postprocessing;
	double threshold;
	std::string positive;
	std::string negative;

public:
	BinaryWorkflow(
				std::unique_ptr<Preprocessing> preprocess,
				std::unique_ptr<BinaryModel> pred,
				std::unique_ptr<Postprocessing> postprocess,
				double threshold=0.0
			);
	BinaryWorkflow(
				std::unique_ptr<BinaryModel> pred,
				std::unique_ptr<Postprocessing> postprocess,
				double threshold=0.0
			);
	BinaryWorkflow(
				std::unique_ptr<BinaryModel> pred,
				double threshold=0.0
			);

	void set_preprocessing(std::unique_ptr<Preprocessing> pipe);
	void set_prediction(std::unique_ptr<BinaryModel> model);
	void set_postprocessing(std::unique_ptr<Postprocessing> pipe);
	void set_threshold(double threshold);

	void print_preprocessing(std::ostream& os) const;
	void print_predictor(std::ostream& os) const;
	void print_postprocessing(std::ostream& os) const;
	void print_threshold(std::ostream& os) const;

	virtual Prediction predict(const SparseVector& v) const override;
	virtual Prediction predict(const std::vector<double> &i) const override;

	virtual std::vector<double> decision_value(const SparseVector &i) const override;
	virtual std::vector<double> decision_value(const std::vector<double> &i) const override;

	virtual size_t num_inputs() const;
	virtual size_t num_outputs() const override;

	size_t num_predictor_outputs() const;

	virtual std::string positive_label() const override;
	virtual std::string negative_label() const override;

	const BinaryModel* get_predictor() const;
	std::unique_ptr<BinaryModel> release_predictor();

	virtual ~BinaryWorkflow();

	friend std::ostream& operator<<(std::ostream& os, const BinaryWorkflow& flow);
	virtual void serialize(std::ostream& os) const override;

	REGISTER_BINARYMODEL_IN_CLASS(BinaryWorkflow)
};

//REGISTER_BINARYMODEL_HPP(BinaryWorkflow)

/*************************************************************************************************/

/**
 * Constructs the default binary workflow around specified model.
 *
 * If model->num_outputs() > 1, aggregation may be selected as:
 * 1. (unweighted) MajorityVote
 * 2. (unweighted) LogisticRegression
 */
std::unique_ptr<BinaryWorkflow>
defaultBinaryWorkflow(std::unique_ptr<BinaryModel> model, bool majorityvote=true);


/*************************************************************************************************/

} // ensemble namespace

/*************************************************************************************************/

#endif /* WORKFLOW_HPP_ */
