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
 * Util.h
 *
 *      Author: Marc Claesen
 */

/*************************************************************************************************/

#ifndef UTIL_H_
#define UTIL_H_

/*************************************************************************************************/

#include "CLI.hpp"
#include "config.h"
#include <string>

/*************************************************************************************************/

namespace ensemble{

/*************************************************************************************************/

extern std::string ENSEMBLESVM_VERSION;
extern std::string ENSEMBLESVM_LICENSE;

/*************************************************************************************************/

/**
 * Exits the program and displays given error on stderr.
 *
 * Exit status EXIT_FAILURE is used.
 */
__attribute__((noreturn)) void exit_with_err(std::string error);

/**
 * Exists the program and displays help based on arguments, header and footer on stdout.
 *
 * Format:
 * 	<header>
 * 	*args[0]\n
 * 	*args[1]\n
 * 	...
 * 	*args[args.size()]\n
 * 	<footer>
 *
 * If success=true, exits with EXIT_SUCCESS, otherwise exits with EXIT_FAILURE.
 */
__attribute__((noreturn)) void exit_with_help(const std::deque<CLI::BaseArgument*> &args, std::string &header, std::string &footer, bool success=false);

/**
 * Exits the program and displays the current version on stdout.
 *
 * Format:
 * 	toolname (EnsembleSVM v<ENSEMBLESVM_VERSION>)\n
 * 	<ENSEMBLESVM_LICENSE>\n
 * 	Written by Marc Claesen.\n
 *
 * Exit status EXIT_SUCCESS is used.
 */
__attribute__((noreturn)) void exit_with_version(std::string toolname);

/*************************************************************************************************/

}

/*************************************************************************************************/

#endif /* UTIL_H_ */
