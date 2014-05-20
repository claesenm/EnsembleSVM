## Introduction

EnsembleSVM is a library providing an API to implement ensemble
learning use Support Vector Machine (SVM) base models. The package
contains some executable tools which behave similar to standard
SVM learning algorithms.

The package is self-contained in the sense that it contains most
necessary tools to build a pipeline for binary classification. Most
notable features include bootstrap sampling, cross-validation and
ensemble training/prediction.

The EnsembleSVM webpage contains all sorts of useful information at:
http://esat.kuleuven.be/stadius/ensemblesvm/

EnsembleSVM uses a divide-and-conquer strategy to handle large data
sets by training base models on (small) subsamples and aggregating
these base models into a strong ensemble.

![workflow](doc/workflow.png?raw=true "Workflow")

If you use EnsembleSVM, please cite our paper:
> M. Claesen, F. De Smet, J. Suykens, and B. De Moor, "*EnsembleSVM: A library for ensemble learning using support vector machines*" Journal of Machine Learning Research, vol. 15, pp. 141–145, 2014.

## Various

For installation instructions, please refer to the file INSTALL or
our webpage. Our webpage also contains an elaborate user manual and
some use cases to familiarise users with the software.

Please report bugs to marc.claesen@esat.kuleuven.be. 

EnsembleSVM is released under the General Lesser Public License 
version 3 (GPLv3+). See the file COPYING.LESSER for the license
agreement.

----

Copyright (C) 2013-2014 KU Leuven

Copying and distribution of this file, with or without modification,
are permitted in any medium without royalty provided the copyright
notice and this notice are preserved.  This file is offered as-is,
without warranty of any kind.
