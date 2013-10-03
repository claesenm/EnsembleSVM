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
 * pipelines.cpp
 *
 *      Author: Marc Claesen
 */

#include "pipeline/pipelines.hpp"

namespace ensemble{
namespace pipeline{

MULTISTAGEPIPELINE_POST_FACTORY(MajorityVote)
MULTISTAGEPIPELINE_POST_FACTORY(LogisticRegression)
MULTISTAGEPIPELINE_POST_FACTORY(NormalizeLinear)
MULTISTAGEPIPELINE_POST_FACTORY(BinarySVMAggregation)
MULTISTAGEPIPELINE_POST_FACTORY(LinearAggregation)

}
}
