# @configure_input@

# todo: why doesn't this work?
SUBDIRS = .
if ENABLE_DOXYGEN
	SUBDIRS += doc/
endif
#SUBDIRS = . doc

#########################
# COMPILATION FLAGS ETC #
#########################
#HEADER_INCLUDES=-Iincludes -I$(LIBSVMPATH)
if DEBUG
EXTRAFLAGS=-Wall -pedantic -g
else
EXTRAFLAGS=-Wall -pedantic
endif

AM_CXXFLAGS=$(EXTRAFLAGS)
AM_CPPFLAGS =-I$(top_srcdir)/include -I$(top_builddir)/src/libsvm -ffast-math

################
# DISTRIBUTION #
################

EXTRA_DIST = doc/doxygen.cfg.in doc/Makefile.in COPYING.LESSER LICENSE $(DEFAULT_LIBSVM_DIR)/svm.cpp \
	$(DEFAULT_LIBSVM_DIR)/svm.h $(DEFAULT_LIBSVM_DIR)/COPYRIGHT boost include/any_iterator

#####################
# AUTOTOOLS SCRIPTS #
#####################
ACLOCAL_AMFLAGS= -I m4

if DEFAULT_LIBSVM_PATH
ABS_PATH_TO_LIBSVM=$(top_srcdir)/$(LIBSVMPATH)
else
ABS_PATH_TO_LIBSVM=$(LIBSVMPATH)
endif

pkginclude_HEADERS = include/CLI.hpp  include/DataFile.hpp  include/Ensemble.hpp  include/io.hpp  include/Kernel.hpp \
	include/LibSVM.hpp  include/Models.hpp  include/SparseVector.hpp  include/Util.hpp include/ThreadPool.hpp \
	include/Type2str.hpp include/SelectiveFactory.hpp include/BinaryWorkflow.hpp include/Registration.hpp include/Executable.hpp

pipeline_includedir = $(pkgincludedir)/pipeline
pipeline_include_HEADERS = include/pipeline/core.hpp include/pipeline/blocks.hpp include/pipeline/pipelines.hpp

anyiter_includedir =  $(pkgincludedir)/any_iterator
anyiter_include_HEADERS = include/any_iterator/any_iterator.hpp

anyiter_detail_includedir = $(pkgincludedir)/any_iterator/detail
anyiter_detail_include_HEADERS = include/any_iterator/detail/any_iterator_abstract_base.hpp \
	include/any_iterator/detail/any_iterator_metafunctions.hpp \
	include/any_iterator/detail/any_iterator_wrapper.hpp

#######
# CONSTRUCT CORE LIBRARY
#######

src/libsvm/svm.cpp: $(ABS_PATH_TO_LIBSVM)/svm.cpp
	mkdir -p src/libsvm	
	cp -f $< $@
	cp $(ABS_PATH_TO_LIBSVM)/COPYRIGHT src/libsvm/
src/libsvm/svm.h: $(ABS_PATH_TO_LIBSVM)/svm.h
	mkdir -p src/libsvm
	cp -f $< $@

# copy libsvm source files we will end up using to make sure there is no mess
# clean up at all times and rebuild at all times
BUILT_SOURCES = src/libsvm/svm.cpp src/libsvm/svm.h src/libsvm/COPYRIGHT
CLEANFILES = src/libsvm/svm.cpp src/libsvm/svm.h src/libsvm/COPYRIGHT
DISTCLEANFILES = src/libsvm/svm.cpp src/libsvm/svm.h src/libsvm/COPYRIGHT doc/Makefile doc/doxygen.cfg
MAINTAINERCLEANFILES = src/libsvm/svm.cpp src/libsvm/svm.h src/libsvm/COPYRIGHT

# we include libsvm code in our shared library
# to avoid problems with platforms that dont support double linking of libraries
lib_LTLIBRARIES = lib/libensemblesvm.la

dist_lib_libensemblesvm_la_SOURCES = src/CLI.cpp \
	src/DataFile.cpp 		\
	src/Ensemble.cpp 		\
	src/io.cpp  			\
	src/Kernel.cpp 			\
	src/LibSVM.cpp 			\
	src/Models.cpp 			\
	src/SparseVector.cpp 	\
	src/Util.cpp			\
	src/pipeline/pipelines.cpp \
	src/BinaryWorkflow.cpp
	
nodist_lib_libensemblesvm_la_SOURCES = src/libsvm/svm.cpp

lib_libensemblesvm_la_LDFLAGS = -version-info $(SHLIBVER)

bin_PROGRAMS=
check_PROGRAMS=

BASELIBS=lib/libensemblesvm.la

## BOOTSTRAP
bin_PROGRAMS += $(top_builddir)/bin/bootstrap
__top_builddir__bin_bootstrap_SOURCES = src/tools/bootstrap.cpp
__top_builddir__bin_bootstrap_LDADD = $(BASELIBS)
## CROSS-VALIDATE
bin_PROGRAMS += $(top_builddir)/bin/cross-validate
__top_builddir__bin_cross_validate_SOURCES = src/tools/cross-validate.cpp
__top_builddir__bin_cross_validate_LDADD = $(BASELIBS)
## DUMMY-GENERATOR
bin_PROGRAMS += $(top_builddir)/bin/dummy-generator
__top_builddir__bin_dummy_generator_SOURCES = src/tools/dummy-generator.cpp
__top_builddir__bin_dummy_generator_LDADD = $(BASELIBS)
## ESVM-PREDICT
bin_PROGRAMS += $(top_builddir)/bin/esvm-predict
__top_builddir__bin_esvm_predict_SOURCES = src/tools/esvm-predict.cpp
__top_builddir__bin_esvm_predict_LDADD = $(BASELIBS)
## ESVM-TRAIN
bin_PROGRAMS += $(top_builddir)/bin/esvm-train
__top_builddir__bin_esvm_train_SOURCES = src/tools/esvm-train.cpp
__top_builddir__bin_esvm_train_LDADD = $(BASELIBS)
## MERGE-MODELS
bin_PROGRAMS += $(top_builddir)/bin/merge-models
__top_builddir__bin_merge_models_SOURCES = src/tools/merge-models.cpp
__top_builddir__bin_merge_models_LDADD = $(BASELIBS)
## SPARSE
bin_PROGRAMS += $(top_builddir)/bin/sparse
__top_builddir__bin_sparse_SOURCES = src/tools/sparse.cpp
__top_builddir__bin_sparse_LDADD = $(BASELIBS)
## SPLIT-DATA
bin_PROGRAMS += $(top_builddir)/bin/split-data
__top_builddir__bin_split_data_SOURCES = src/tools/split-data.cpp
__top_builddir__bin_split_data_LDADD = $(BASELIBS)
## esvm-edit
bin_PROGRAMS += $(top_builddir)/bin/esvm-edit
__top_builddir__bin_esvm_edit_SOURCES = src/tools/esvm-edit.cpp
__top_builddir__bin_esvm_edit_LDADD = $(BASELIBS)

## TEST BINARIES
check_PROGRAMS += $(top_builddir)/tests/blocks
__top_builddir__tests_blocks_SOURCES = src/tests/test_blocks.cpp
__top_builddir__tests_blocks_LDADD = $(BASELIBS)
check_PROGRAMS += $(top_builddir)/tests/pipelines
__top_builddir__tests_pipelines_SOURCES = src/tests/test_pipelines.cpp
__top_builddir__tests_pipelines_LDADD = $(BASELIBS)
check_PROGRAMS += $(top_builddir)/tests/sparsevector
__top_builddir__tests_sparsevector_SOURCES = src/tests/test_sparsevector.cpp
__top_builddir__tests_sparsevector_LDADD = $(BASELIBS)
check_PROGRAMS += $(top_builddir)/tests/svmmodel
__top_builddir__tests_svmmodel_SOURCES = src/tests/test_svmmodel.cpp
__top_builddir__tests_svmmodel_LDADD = $(BASELIBS)
check_PROGRAMS += $(top_builddir)/tests/svmensemble
__top_builddir__tests_svmensemble_SOURCES = src/tests/test_svmensemble.cpp
__top_builddir__tests_svmensemble_LDADD = $(BASELIBS)
check_PROGRAMS += $(top_builddir)/tests/workflow
__top_builddir__tests_workflow_SOURCES = src/tests/test_workflow.cpp
__top_builddir__tests_workflow_LDADD = $(BASELIBS)

# http://www.gnu.org/software/automake/manual/html_node/Scripts_002dbased-Testsuites.html#Scripts_002dbased-Testsuites
TESTS=$(check_PROGRAMS)

TESTS+=$(top_builddir)/tests/dummy-generator.test
EXTRA_DIST+=$(top_srcdir)/tests/dummy-generator.test $(top_srcdir)/tests/test_dummy-generator_input.txt \
	$(top_srcdir)/tests/test_dummy-generator_dummies.txt $(top_srcdir)/tests/test_dummy-generator_reference.txt

TESTS+=$(top_builddir)/tests/sparse.test
EXTRA_DIST+=$(top_srcdir)/tests/sparse.test $(top_srcdir)/tests/test_sparse_input.txt $(top_srcdir)/tests/test_sparse_output.txt