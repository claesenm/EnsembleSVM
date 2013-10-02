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
 * Registration.hpp
 *
 *      Author: Marc Claesen
 */

#ifndef REGISTRATION_HPP_
#define REGISTRATION_HPP_

#include "pipeline/pipelines.hpp"
#include "Models.hpp"
#include "Ensemble.hpp"
#include "BinaryWorkflow.hpp"
#include "SelectiveFactory.hpp"

/*************************************************************************************************/

namespace ensemble{

/*************************************************************************************************/

void registerMultistagePipes(){
	MULTISTAGEPIPELINE_REGISTRATION(pipeline::MajorityVote)
	MULTISTAGEPIPELINE_REGISTRATION(pipeline::LogisticRegression)
	MULTISTAGEPIPELINE_REGISTRATION(pipeline::NormalizeLinear)
	MULTISTAGEPIPELINE_REGISTRATION(pipeline::BinarySVMAggregation)
	MULTISTAGEPIPELINE_REGISTRATION(pipeline::LinearAggregation)
}

void registerBinaryModels(){
	BINARYMODEL_REGISTRATION(SVMModel)
	BINARYMODEL_REGISTRATION(SVMEnsemble)
	BINARYMODEL_REGISTRATION(BinaryWorkflow)
}

/*************************************************************************************************/

namespace registration{

// force registration
struct Registrar{
	Registrar(){
		registerMultistagePipes();
		registerBinaryModels();
	}
} registrar;

} // registration namespace

}	// ensemble namespace

/*************************************************************************************************/

#endif /* REGISTRATION_HPP_ */
