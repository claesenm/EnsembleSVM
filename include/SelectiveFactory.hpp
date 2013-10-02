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
 * SelectiveFactory.hpp
 *
 *      Author: Marc Claesen
 */

/*************************************************************************************************/

#ifndef SELECTIVEFACTORY_HPP_
#define SELECTIVEFACTORY_HPP_

#include <vector>
#include <memory>
#include <map>

/*************************************************************************************************/

namespace ensemble{

/*************************************************************************************************/

// todo: enable variadic Criterion
// http://stackoverflow.com/questions/9831501/how-can-i-have-multiple-parameter-packs-in-a-variadic-template

/**
 * A meta-factory which forwards the input to a matching Base factory,
 * based on the given Criterion. A SelectiveFactory can generically instantiate
 * whatever derived class can be constructed that fits Criterion.
 *
 * The key advantage to this factory is that candidate constructors can be registered
 * from anywhere, for example using a global prior to main(). This works because the factory
 * is completely static. This allows registration to occur in library code.
 *
 * Since Predicates are user-defined, several derived classes may fit a given Criterion.
 * Therefore a SelectiveFactory can, in principle, return a collection of
 * constructed Base objects.
 *
 * Note that this implementation is not thread safe, but it's easy to fix that by adding a mutex
 * to container().
 */
template <	typename Base,		// the base class for instantiations
			typename Criterion,	// selection criterion for factories
			typename... Input	// input to base class factories
		 >
class SelectiveFactory{
public:
	// Predicate function pointers
	typedef bool(*Predicate)(Criterion);

	// Factory function pointers
	typedef std::unique_ptr<Base>(*Factory)(Input...);

	// the internal container holding Predicate and Factory pairs
	typedef typename std::map<Predicate,Factory> FunctionContainer;

private:

	/**
	 * The actual internal factory container.
	 *
	 * Wrapped inside a function to avoid the static
	 * initialization order fiasco.
	 *
	 * This is the Construct On First Use idiom.
	 */
	static FunctionContainer& container(){
		static FunctionContainer cont=FunctionContainer();
		return cont;
	}

public:

	/**
	 * This is a purely static class. No objects should be made.
	 */
	SelectiveFactory() = delete;

	/**
	 * Registers a new factory with given selection predicate.
	 */
	static void registerPtr(Predicate predicate, Factory factory){
		container().insert(std::make_pair(predicate,factory));
	}

	/**
	 * Constructs Base objects using all registered Factories for which
	 * predicate(criterion) holds.
	 */
	static std::vector<std::unique_ptr<Base>> Produce(Criterion criterion, Input... value){
		std::vector<std::unique_ptr<Base>> result;
		for(auto& f: container()){
			if((f.first)(criterion)){
				result.emplace_back((f.second)(value...));
			}
		}
		return std::move(result);
	}

	/**
	 * Returns the amount of factories currently registered.
	 */
	static size_t size(){ return container().size(); }
};

/*************************************************************************************************/

} // ensemble namespace

/*************************************************************************************************/

#endif /* SELECTIVEFACTORY_HPP_ */
