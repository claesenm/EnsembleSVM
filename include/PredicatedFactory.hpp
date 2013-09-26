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
 * PredicatedFactory.hpp
 *
 *      Author: Marc Claesen
 */

/*************************************************************************************************/

#ifndef PREDICATEDFACTORY_HPP_
#define PREDICATEDFACTORY_HPP_

/*************************************************************************************************/

#include <vector>
#include <memory>

/*************************************************************************************************/

namespace ensemble{

/*************************************************************************************************/

// todo: enable variadic Criterion
// http://stackoverflow.com/questions/9831501/how-can-i-have-multiple-parameter-packs-in-a-variadic-template

/**
 * A meta-factory which forwards the input to a matching Base factory,
 * based on the given Criterion. A PredicatedFactory can generically instantiate
 * whatever derived class can be constructed that fits Criterion.
 *
 * The key advantage to this factory is that candidate constructors can be registered
 * can happen anywhere, not just in main(). This works because the factory is completely
 * static. Users of this factory need not worry about registration in each executable.
 *
 * Since Predicates are user-defined, several derived classes may fit a given Criterion.
 * Therefore a PredicatedFactory can, in principle, return a collection of
 * constructed Base objects.
 */
template <	typename Base,		// the base class for instantiations
			typename Criterion,	// selection criterion for factories
			typename... Input	// input to base class factories
		 >
class PredicatedFactory{
public:
	typedef bool(*Predicate)(Criterion);
	typedef std::unique_ptr<Base>(*Factory)(Input...);
private:
	// use a map to ensure nothing gets registered twice in the compilation chain
	typedef typename std::map<Predicate,Factory> FunctionContainer;

	// must be a naked pointer to overcome the problem of initialization order
	// http://stackoverflow.com/questions/17443163/insert-content-in-static-containers-of-a-templated-class-prior-to-main
	static FunctionContainer *registeredFuns;

	// initializes the registered functions container if it is empty.
	static void create(){
		if(!registeredFuns) registeredFuns=new FunctionContainer;
	};

	// used to clean-up registeredFuns, which is otherwise not deleted
	// this causes a memory leak on program exit
	// which may not be cleaned up by all operating systems
	// http://stackoverflow.com/a/17443575/2148672
	class Deleter{
		~Deleter(){
			delete PredicatedFactory<Base,Criterion,Input...>::registeredFuns;
		}
	};
	static Deleter deleter;

public:
	/**
	 * Registers a new factory with given selection predicate.
	 */
	static void registerPtr(Predicate predicate, Factory factory){
		create();
		registeredFuns->insert(std::make_pair(predicate,factory));
	}

	/**
	 * Constructs Base objects using all registered Factories for which
	 * predicate(criterion) holds.
	 */
	static std::vector<std::unique_ptr<Base>> Produce(Criterion criterion, Input... value){
		create();
		std::vector<std::unique_ptr<Base>> result;
		for(auto& f: *registeredFuns){
			if((f.first)(criterion)){
				result.emplace_back((f.second)(value...));
			}
		}
		return std::move(result);
	}
};

/*************************************************************************************************/

// member initialization
template <typename Base, typename Criterion, typename... Input>
typename PredicatedFactory<Base,Criterion,Input...>::FunctionContainer *PredicatedFactory<Base,Criterion,Input...>::registeredFuns=nullptr;
template <typename Base, typename Criterion, typename... Input>
typename PredicatedFactory<Base,Criterion,Input...>::Deleter PredicatedFactory<Base,Criterion,Input...>::deleter;

/*************************************************************************************************/

} // ensemble namespace

/*************************************************************************************************/

#endif /* PREDICATEDFACTORY_HPP_ */
