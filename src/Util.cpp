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
 * Util.cpp
 *
 *      Author: Marc Claesen
 */

#include "Util.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <limits>

#include "config.h"

#ifndef PACKAGE_VERSION
#define PACKAGE_VERSION ""
#endif

#ifndef PACKAGE_URL
#define PACKAGE_URL ""
#endif

namespace ensemble{

std::string ENSEMBLESVM_VERSION=std::string(PACKAGE_VERSION);
std::string ENSEMBLESVM_URL=std::string(PACKAGE_URL);
std::string ENSEMBLESVM_LICENSE=std::string("Copyright (c) 2013, KU Leuven.\n"
		"License: GNU LGPL version 3 or later <http://www.gnu.org/licenses/lgpl.html>\n");

void exit_with_err(std::string err){
	std::cout << err << std::endl;
	exit(EXIT_FAILURE);
}

void exit_with_help(const std::deque<CLI::BaseArgument*> &args, std::string &header, std::string &footer, bool success){
	std::cout << header;
	for(std::deque<CLI::BaseArgument*>::const_iterator I=args.begin(),E=args.end();I!=E;++I)
		std::cout << **I;
	std::cout << footer;

	if(success)
		exit(EXIT_SUCCESS);
	else
		exit(EXIT_FAILURE);
}

void exit_with_version(std::string toolname){
	std::cout << toolname << " (part of EnsembleSVM v" << ENSEMBLESVM_VERSION << ")" << std::endl;
	std::cout << "Available at: " << ENSEMBLESVM_URL << std::endl << std::endl;
	std::cout << ENSEMBLESVM_LICENSE << std::endl;
	std::cout << "Written by Marc Claesen." << std::endl;

	exit(EXIT_SUCCESS);
}

}
