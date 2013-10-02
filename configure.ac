#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
AC_PREREQ([2.67])
AC_INIT([EnsembleSVM], [2.0], [marc.claesen@esat.kuleuven.be],[ensemblesvm],[http://esat.kuleuven.be/sista/ensemblesvm])
AC_CONFIG_SRCDIR([src/CLI.cpp])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([foreign no-define dist-bzip2 -Wall -Wno-portability subdir-objects 1.9 tar-ustar])

#####
# BASIC CONFIGURATION
#####

# location of prepackaged libsvm
AC_SUBST([DEFAULT_LIBSVM_DIR],[libsvm-weights-3.17])

# Library version, this is an LDFLAG for Libtool -version-info 0:0:0
# http://www.sourceware.org/autobook/autobook/autobook_91.html#SEC91
AC_SUBST([SHLIBVER],[2:0:0])

AC_SUBST([PACKAGE_VERSION])

####
# LIBTOOL STUFF
####
LT_INIT

# Checks for programs.
AC_PROG_CC
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_PROG_LIBTOOL

AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_FUNC_STRTOD

# Checks for header files.
AC_CHECK_HEADERS([float.h limits.h locale.h stdlib.h string.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE

# Check if we are working on Darwin
OS=shell uname
AM_CONDITIONAL(COMPILING_ON_DARWIN, [test x"$OS" = xDarwin])

# Extra flags
AC_ARG_ENABLE([debug], AS_HELP_STRING([--enable-debug], [turn on debugging]),
[case "${enableval}" in
  yes) debug=true ;;
  no)  debug=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-debug]) ;;
esac],[debug=false])
AM_CONDITIONAL([DEBUG], [test x$debug = xtrue])

AC_ARG_ENABLE([doxygen], AS_HELP_STRING([--enable-code-doc], [compiles html/pdf code docs using doxygen]),
[
	dnl action if present
	USE_DOXYGEN=1
],[
	dnl action if not present
	USE_DOXYGEN=0
])
AC_SUBST(USE_DOXYGEN)

# Declare precious variables
AC_ARG_VAR(LIBSVMPATH,[absolute path to custom LIBSVM folder])
AM_CONDITIONAL([DEFAULT_LIBSVM_PATH], [test -n $LIBSVMPATH])
if test -z "$LIBSVMPATH"; then
	LIBSVMPATH=$DEFAULT_LIBSVM_DIR
fi

if test -z "$CXXFLAGS"; then
	CXXFLAGS=-O2
fi

# Checks for library functions.
AC_CHECK_FUNCS([pow exp setlocale sqrt strdup strrchr strtol strtoul])

AX_CXX_COMPILE_STDCXX_11([],mandatory)
AX_PTHREAD

LIBS="$PTHREAD_LIBS $LIBS"
CFLAGS="$CFLAGS $PTHREAD_CFLAGS"
CXXFLAGS="$CXXFLAGS $PTHREAD_CFLAGS"
CC="$PTHREAD_CC"

AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile])

# http://www.sourceware.org/autobook/autobook/autobook_86.html#SEC86
Xsed="sed -e s/^X//"
LTLIBOBJS=`echo X"$LIBOBJS"|\
           [$Xsed -e "s,\.[^.]* ,.lo ,g;s,\.[^.]*$,.lo,"]`
AC_SUBST(LTLIBOBJS)

AC_OUTPUT
