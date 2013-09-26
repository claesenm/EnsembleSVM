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
 * LibSVM.cpp
 *
 *      Author: Marc Claesen
 */

#include "LibSVM.hpp"
#include "Util.hpp"
#include "assert.h"
#include "Kernel.hpp"
#include <sstream>
#include <stdlib.h>
#include <memory>

using std::unique_ptr;
using std::string;
using namespace ensemble;

// todo deleter for svm_node
// todo deleter for svm_problem
// todo deleter for svm_model

namespace{

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void print_nullptr(const char *s) {}

unique_ptr<SparseVector> Node2SV(const svm_node *node){
	unique_ptr<SparseVector::SparseSV > sparsesv(new SparseVector::SparseSV());
	for(int i=0;node[i].index!=-1;++i){
		sparsesv->push_back(std::make_pair(node[i].index,node[i].value));
	}
	unique_ptr<SparseVector> result(new SparseVector(std::move(*sparsesv)));
	return std::move(result);
}

/**
 * Creates a LibSVM svm_node from v.
 * The resulting svm_node is allocated using malloc and requires proper handling.
 */
svm_node *SV2Node(const SparseVector *v){
	svm_node *node=Malloc(svm_node,v->numNonzero()+1);
	unsigned idx=0;
	for(SparseVector::const_iterator I=v->begin(),E=v->end();I!=E;++I,++idx){
		node[idx].index=I->first;
		node[idx].value=I->second;
	}
	node[idx].index=-1;
	return node;
}


unique_ptr<Kernel> extractKernel(const svm_parameter &param){
	unique_ptr<Kernel> kernel;
	switch(param.kernel_type){
	case KERNEL_TYPES::LINEAR:	// linear kernel
		kernel.reset(new LinearKernel());
		break;
	case KERNEL_TYPES::POLY: // polynomial kernel
		kernel.reset(new PolyKernel(param.degree,param.coef0,param.gamma));
		break;
	case KERNEL_TYPES::RBF:	// RBF kernel
		kernel.reset(new RBFKernel(param.gamma));
		break;
	case KERNEL_TYPES::SIGMOID: // sigmoid kernel
		kernel.reset(new SigmoidKernel(param.coef0,param.gamma));
		break;
	case KERNEL_TYPES::USERDEF: // precomputed kernel
		// todo
		break;
	default:
		std::ostringstream ss(std::ostringstream::out);
		ss.str("Invalid LIBSVM kernel type: ");
		ss << param.kernel_type << ".";

		exit_with_err(ss.str());
		break;
	}
	return kernel;
}

SVMModel::SV_container extractSV(const svm_model &libsvm){
	unsigned numsv = libsvm.l;

	SVMModel::SV_container SVs;
	SVs.reserve(numsv);

	for(unsigned i=0;i<numsv;++i){
		// extract SV
		unique_ptr<SparseVector> sv(Node2SV(libsvm.SV[i]));
		assert(sv.get() && "Corrupt SV!");
		SVs.emplace_back(sv.release());
	}
	return std::move(SVs);
}

SVMModel::Classes extractClasses(const svm_model &libsvm){
	unsigned numclasses = libsvm.nr_class;

	SVMModel::Classes classes(numclasses);

	for(unsigned i=0;i<numclasses;++i){
		// get label and nr_sv
		unsigned nr_sv = libsvm.nSV[i];
		std::ostringstream os(std::ostringstream::out);
		os << libsvm.label[i];
		string label=os.str();
		classes.at(i)=std::make_pair(label,nr_sv);
	}
	return std::move(classes);
}

SVMModel::Weights extractWeights(const svm_model &libsvm){
	unsigned numsv=libsvm.l, numclasses=libsvm.nr_class;
	SVMModel::Weights weights(numsv*(numclasses-1),0);

	for(unsigned i=0;i<numclasses-1;++i){
		for(unsigned j=0;j<numsv;++j){
			unsigned idx=j+i*numsv;
			weights.at(idx)=libsvm.sv_coef[i][j];
		}
	}
	return std::move(weights);
}

std::vector<double> extractConstants(const svm_model &libsvm){
	unsigned numclasses = libsvm.nr_class;
	unsigned numconstants = numclasses*(numclasses-1)/2;

	std::vector<double> constants(&libsvm.rho[0],&libsvm.rho[numconstants]);
	return std::move(constants);
}

/**
 * Completes the kernel information in param.
 */
void completeSVMParameter(const Kernel *kernel, svm_parameter *param){
//	param->kernel_type=kernel->getType();

	switch(kernel->getType()){
	case KERNEL_TYPES::LINEAR:	// linear kernel
		if(dynamic_cast<const LinearKernel*>(kernel)){
			param->kernel_type=LINEAR; // in LIBSVM svm.h
			param->degree=0;
			param->coef0=0;
			param->gamma=0;
			return;
		}else{
			exit_with_err("Error building kernel.");
		}
		break;
	case KERNEL_TYPES::POLY: // polynomial kernel
		if(const PolyKernel *casted=dynamic_cast<const PolyKernel*>(kernel)){
			param->kernel_type=POLY; // in LIBSVM svm.h
			param->degree=casted->getDegree();
			param->coef0=casted->getCoef();
			param->gamma=casted->getGamma();
			return;
		}else{
			exit_with_err("Error building kernel.");
		}
		break;
	case KERNEL_TYPES::RBF:	// RBF kernel
		if(const RBFKernel *casted=dynamic_cast<const RBFKernel*>(kernel)){
			param->kernel_type=RBF; // in LIBSVM svm.h
			param->degree=0;
			param->coef0=0;
			param->gamma=casted->getGamma();
			return;
		}else{
			exit_with_err("Error building kernel.");
		}
		break;
	case KERNEL_TYPES::SIGMOID: // sigmoid kernel
		if(const SigmoidKernel *casted=dynamic_cast<const SigmoidKernel*>(kernel)){
			param->kernel_type=SIGMOID; // in LIBSVM svm.h
			param->degree=0;
			param->coef0=casted->getCoef();
			param->gamma=casted->getGamma();
			return;
		}else{
			exit_with_err("Error building kernel.");
		}
		break;
	case KERNEL_TYPES::USERDEF: // precomputed kernel
		if(dynamic_cast<const UserdefKernel*>(kernel)){
			param->kernel_type=PRECOMPUTED; // in LIBSVM svm.h
			// todo userdef
			return;
		}else{
			exit_with_err("Error building kernel.");
		}
		break;
	default:
		exit_with_err("Illegal kernel!");
		break;
	}
}

} // anonymous namespace

namespace ensemble{

namespace LibSVM{

unique_ptr<SVMModel> convert(unique_ptr<svm_model> libsvm){
	// extract kernel
	unique_ptr<Kernel> kernel = extractKernel(libsvm->param);

	// extract SV
	SVMModel::SV_container&& SVs = extractSV(*libsvm);

	// extract SV weights
	SVMModel::Weights&& weights = extractWeights(*libsvm);

	// extract class information
	SVMModel::Classes&& classes = extractClasses(*libsvm);

	// extract constants
	std::vector<double>&& constants = extractConstants(*libsvm);

	// build model
	unique_ptr<SVMModel> model(new SVMModel(std::move(SVs),std::move(weights),std::move(classes),std::move(constants),std::move(kernel)));

	// clean up
	svm_model *libsvmmodel = libsvm.get(); // todo: create libsvm deleter for unique_ptr
	svm_free_and_destroy_model(&libsvmmodel);
	libsvm.release();

	return std::move(model);
}

unique_ptr<SVMModel> trainBSVM(const Kernel *kernel, double pospen, double negpen,
		double cachesize, const vector<const SparseVector*> &data, const vector<bool> &labels,
		const vector<double> &penalties, unsigned trainsize, bool mutelibsvm){

	auto problem=construct_BSVM_problem(kernel, pospen, negpen, cachesize, data, labels, penalties, trainsize, mutelibsvm);
	return libsvm_train(std::move(problem));
}

//unique_ptr<SVMModel> trainBSVM(const Kernel *kernel, double pospen, double negpen,
//		double cachesize, const vector<const SparseVector*> &data, const vector<bool> &labels,
//		const vector<double> &penalties, unsigned trainsize, bool mutelibsvm){
//	if(mutelibsvm)
//		svm_set_print_string_function(&print_nullptr);
//
//	// construct LibSVM svm_parameters
//	svm_parameter *param=Malloc(svm_parameter,1);
//	param->svm_type=C_SVC; // C-SVC, defined in LibSVM's svm.h
//	param->cache_size = cachesize;
//
//	completeSVMParameter(kernel,param);
//	param->C = 1.0;	// weight is completely defined in pospen/negpen OR pointwise weights
//
//	// default values from LibSVM's svm-train.c
//	param->eps = 1e-3;
//	param->shrinking = 1;
//	param->probability = 0;
//
//	// weights are the CLASS-specific weights/labels
//	param->nr_weight=2;
//	param->weight = Malloc(double,2);
//	param->weight[0]=pospen;
//	param->weight[1]=negpen;
//	param->weight_label = Malloc(int,2); // internally uses integer labels +1 for positive class, -1 for negative
//	param->weight_label[0]=+1;
//	param->weight_label[1]=-1;
//
//	svm_problem *prob=Malloc(svm_problem,1);
//	prob->l = trainsize;
//
//	// insert training labels
//	prob->y = Malloc(double,trainsize);
//	for(unsigned idx=0;idx<trainsize;++idx){
//		if(labels[idx]) prob->y[idx] = +1;
//		else prob->y[idx] = -1;
//	}
//
//	// insert training data
//	prob->x=Malloc(svm_node*,trainsize);
//	for(unsigned idx=0;idx<trainsize;++idx){
//		prob->x[idx] = SV2Node(data[idx]);
//	}
//
//	// pointwise penalties
//	prob->W=Malloc(double,trainsize);
//	for(unsigned idx=0;idx<trainsize;++idx)
//		prob->W[idx]=penalties[idx];
//
//
//	// construct and convert svmmodel
//	unique_ptr<svm_model> libsvmmodel(svm_train(prob, param));
//
//	// clean up
//	svm_destroy_param(param);
//	free(prob->W);
//	free(prob->x);
//	free(prob->y);
//	free(prob);
//
//	return std::move(convert(std::move(libsvmmodel)));
//}



full_svm_problem construct_BSVM_problem
(const Kernel *kernel, double pospen, double negpen,
		double cachesize, const vector<const SparseVector*> &data, const vector<bool> &labels,
		const vector<double> &penalties, unsigned trainsize, bool mutelibsvm){
	if(mutelibsvm)
		svm_set_print_string_function(&print_nullptr);

	// construct LibSVM svm_parameters
	std::unique_ptr<svm_parameter> param(Malloc(svm_parameter,1));
	param->svm_type=C_SVC; // C-SVC, defined in LibSVM's svm.h
	param->cache_size = cachesize;

	completeSVMParameter(kernel,param.get());
	param->C = 1.0;	// weight is completely defined in pospen/negpen OR pointwise weights

	// default values from LibSVM's svm-train.c
	param->eps = 1e-3;
	param->shrinking = 1;
	param->probability = 0;

	// weights are the CLASS-specific weights/labels
	param->nr_weight=2;
	param->weight = Malloc(double,2);
	param->weight[0]=pospen;
	param->weight[1]=negpen;
	param->weight_label = Malloc(int,2); // internally uses integer labels +1 for positive class, -1 for negative
	param->weight_label[0]=+1;
	param->weight_label[1]=-1;

	std::unique_ptr<svm_problem> prob(Malloc(svm_problem,1));
	prob->l = trainsize;

	// insert training labels
	prob->y = Malloc(double,trainsize);
	for(unsigned idx=0;idx<trainsize;++idx){
		if(labels[idx]) prob->y[idx] = +1;
		else prob->y[idx] = -1;
	}

	// insert training data
	prob->x=Malloc(svm_node*,trainsize);
	for(unsigned idx=0;idx<trainsize;++idx){
		prob->x[idx] = SV2Node(data[idx]);
	}

	// pointwise penalties
	prob->W=Malloc(double,trainsize);
	for(unsigned idx=0;idx<trainsize;++idx)
		prob->W[idx]=penalties[idx];

	return std::make_pair(std::move(prob),std::move(param));
}


std::unique_ptr<SVMModel> libsvm_train(full_svm_problem &&problem){

	unique_ptr<svm_model> libsvmmodel(svm_train(problem.first.get(), problem.second.get()));

	// clean up
	svm_destroy_param(problem.second.get());
	free(problem.first->W);
	free(problem.first->x);
	free(problem.first->y);
	free(problem.first.get());

	problem.first.release();
	problem.second.release();

	return std::move(convert(std::move(libsvmmodel)));
}

} // ensemble::LibSVM namespace

} // ensemble namespace
