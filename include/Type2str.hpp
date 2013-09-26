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
 * Type2str.hpp
 *
 *      Author: Marc Claesen
 */

/*************************************************************************************************/

#ifndef TYPE2STR_HPP_
#define TYPE2STR_HPP_

/*************************************************************************************************/

#include <string>
#include <vector>
#include <list>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <queue>

using std::string;

/*************************************************************************************************/

namespace ensemble{

/*************************************************************************************************/

class SparseVector;
class Model;
class SVMEnsemble;
class SVMModel;

/*************************************************************************************************/

}

/*************************************************************************************************/

namespace pipeline{

/*************************************************************************************************/

/**
 * Used to serialize types.
 *
 * A specialization must exist for any type that requires serialization.
 */
template <typename T>
struct ToStr{
	static string get(){
		return string("unknown");
	}
};

template <typename T>
struct ToStr<T*>{
	static string get(){
		string t=ToStr<T>::get();
		t.append("*");
		return t;
	}
};

template <typename T>
struct ToStr<T&>{
	static string get(){
		string t=ToStr<T>::get();
		t.append("&");
		return t;
	}
};

/*************************************************************************************************/

#define ToStrUntemplated(Type) template<> \
struct ToStr<Type>{ \
	static string get(){ return #Type; } \
};

ToStrUntemplated(int)
ToStrUntemplated(unsigned)
ToStrUntemplated(long)
ToStrUntemplated(bool)
ToStrUntemplated(float)
ToStrUntemplated(double)
ToStrUntemplated(std::string)
ToStrUntemplated(ensemble::Model)
ToStrUntemplated(ensemble::SVMModel)
ToStrUntemplated(ensemble::SVMEnsemble)
ToStrUntemplated(ensemble::SparseVector)

/*************************************************************************************************/

#define ToStrContainer(Container) template<typename T>	\
struct ToStr<Container<T>>{ \
	static string get(){ \
		string v{#Container}; \
		v.append("<"); \
		string t=ToStr<T>::get(); \
		v.append(t); \
		v.append(">"); \
		return v; \
	} \
};

ToStrContainer(std::vector)
ToStrContainer(std::queue)
ToStrContainer(std::list)
ToStrContainer(std::deque)
ToStrContainer(std::unique_ptr)
ToStrContainer(std::shared_ptr)

/*************************************************************************************************/

#define ToStrPairContainer(Container) template<typename T, typename N>	\
struct ToStr<Container<T,N>>{ \
	static string get(){ \
		string v{#Container}; \
		v.append("<"); \
		string t=ToStr<T>::get(); \
		v.append(t); \
		v.append(","); \
		t=ToStr<N>::get(); \
		v.append(t); \
		v.append(">"); \
		return v; \
	} \
};

ToStrPairContainer(std::map)
ToStrPairContainer(std::pair)

/*************************************************************************************************/

} // pipeline namespace

/*************************************************************************************************/

#endif /* TYPE2STR_HPP_ */
