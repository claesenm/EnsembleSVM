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
 * CLI.h
 *
 *      Author: Marc Claesen
 */

/*************************************************************************************************/

#ifndef CLI_H_
#define CLI_H_

/*************************************************************************************************/

#include <map>
#include <vector>
#include <deque>
#include <string>
#include <sstream>
#include <iostream>
#include <typeinfo>

/*************************************************************************************************/

using std::map;
using std::vector;
using std::deque;
using std::string;
using std::istringstream;
using std::pair;

/*************************************************************************************************/

namespace CLI{

/*************************************************************************************************/

template<typename T>
struct TName
{
	static const char* is()
	{
		return typeid(T).name();
	}
};

template<>
struct TName<bool>
{
	static const char* is() { return "bool"; }
};

template<>
struct TName<long double>
{
	static const char* is() { return "double"; }
};

template<>
struct TName<double>
{
	static const char* is() { return "double"; }
};


template<>
struct TName<int>
{
	static const char* is() { return "int"; }
};

template<>
struct TName<unsigned>
{
	static const char* is() { return "unsigned"; }
};

template<>
struct TName<string>
{
	static const char* is() { return "string"; }
};

template<>
struct TName<char>
{
	static const char* is() { return "char"; }
};

template <typename T>
const char* type_name_of(const T& instance)
{
	return TName<T>::is();
}

/*************************************************************************************************/

/**
 * Abstract base class to implement command line arguments with keywords.
 */
class BaseArgument{
	// part of safe bool idiom
	typedef void (BaseArgument::*bool_type)() const;
	void this_type_does_not_support_comparisons() const {};

protected:
	vector<string> description;
	string keyword;
	bool isconfigured;

	/**
	 * Indicates whether this argument should remain in the parse-list after finding a hit.
	 * Default value is false, but can be overwritten in derived classes.
	 */
	virtual bool parseAfterHit() const{ return false; };
	void printDescription(std::ostream &os) const;

public:
	// the length of the tab used before starting argument descriptions
	static const unsigned TABLENGTH=12;

	/**
	 * Constructs a BaseArgument with 1-line description with given keyword.
	 */
	BaseArgument(const string &description, const string &keyword):description(1,""),keyword(keyword),isconfigured(false){ this->description[0]=description; }

	/**
	 * Constructs a BaseArgument with multiline description (1 line per entry in the vector) with given keyword.
	 */
	BaseArgument(const vector<string> &description, const string &keyword):description(description),keyword(keyword),isconfigured(false){}

	/**
	 * Attempts to read the command line argument.
	 * Returns the first index after this argument. If reading failed, return value will equal idx.
	 *
	 * If the argument has a keyword, argv[idx] must equal the keyword.
	 */
	virtual unsigned read(char **argv, unsigned idx)=0;

	/**
	 * Parses command line, returns first index after this argument and  updates internal state of argument if it's found.
	 * If parsing failed, return.first will equal idx.
	 * Bool in returned pair indicates whether this argument should remain in the list of parse candidates.
	 *
	 * If the argument has a keyword, argv[idx] must equal the keyword. User-defined derived arguments may deviate from this rule.
	 */
	pair<unsigned,bool> parse(char **argv, unsigned idx);

	/**
	 * Returns true if this argument has been configured through CLI.
	 */
	bool configured() const{ return isconfigured; }

	/**
	 * Returns the amount of values in this argument.
	 */
	virtual unsigned size() const=0;
	virtual void print(std::ostream &os) const=0;

	/**
	 * Returns the length of this argument: keyword (optional) + amount of values to be read.
	 */
	unsigned length() const{
		return size()+1;
	}

	/**
	 * Prints this argument to os. Used in help functions.
	 */
	friend std::ostream &operator<<(std::ostream &os, const BaseArgument &arg){
		arg.print(os);
		return os;
	}

	/**
	 * Checks whether key matches the keyword of this argument.
	 */
	virtual bool operator==(const string &key) const{ return keyword.compare(key)==0; }
	virtual bool operator!=(const string &key) const{ return !operator==(key); }

	/**
	 * Returns true if the BaseArgument has been configured on CLI. Allows the following:
	 * 	BaseArgument t;
	 * 	if(t){}
	 */
	operator bool_type() const{ // safe bool idiom
		return isconfigured==true ? &BaseArgument::this_type_does_not_support_comparisons : 0;
	}
	/**
	 * Returns false if the BaseArgument has been configured on CLI. Allows the following:
	 * 	BaseArgument t;
	 * 	if(!t){}
	 */
	bool_type operator!() const{ // safe bool idiom
		return isconfigured==false ? &BaseArgument::this_type_does_not_support_comparisons : 0;
	}

	string key() const{ return keyword; }

	virtual ~BaseArgument(){};
};

/*************************************************************************************************/

/**
 * Used to implement boolean flag command line arguments with a default setting.
 *
 * Default setting is toggled when the keyword is present as a command line argument.
 */
class FlagArgument:public BaseArgument{

protected:
	bool val;

public:
	FlagArgument(const string &description, const string &keyword, bool defvalue=false);
	FlagArgument(const vector<string> &description, const string &keyword, bool defvalue=false);
	virtual unsigned read(char **argv, unsigned idx);
	virtual unsigned size() const{ return 0; };
	virtual void print(std::ostream &os) const;
	virtual bool value() const{ return val; }
	virtual ~FlagArgument(){};
};

/*************************************************************************************************/

/**
 * Used to implement boolean flags that do not appear in documentation.
 *
 * For example: standard flags --help and --version.
 */
class SilentFlagArgument:public FlagArgument{
public:
	SilentFlagArgument(const string &keyword, bool defvalue=false);
	virtual void print(std::ostream &os) const;
};

/*************************************************************************************************/

template <class T>
class Argument:public BaseArgument{
public:
	typedef deque<T> Content;

protected:
	Content content;

	// implement function here to stop compilation issues on some platforms
	// apparently it's not clear to some compilers that subclasses of Argument<T> are subclasses of BaseArgument
	const string &getKeyword() const{ return keyword; }
	bool config() const{ return configured(); }

public:
	Argument(const string &description, const string &keyword, const deque<T> &content);
	Argument(const vector<string> &description, const string &keyword, const deque<T> &content);

	virtual unsigned read(char **argv, unsigned idx);
	virtual unsigned size() const{ return content.size(); }
	T operator[](unsigned idx) const{ return content.at(idx); }

	virtual void print(std::ostream &os) const;
	virtual ~Argument(){}
};

template <class T>
Argument<T>::Argument(const string &description, const string &keyword, const deque<T> &content)
:BaseArgument(description,keyword),content(content){}

template <class T>
Argument<T>::Argument(const vector<string> &description, const string &keyword, const deque<T> &content)
:BaseArgument(description,keyword),content(content){}

template <class T>
unsigned Argument<T>::read(char **argv, unsigned idx){
	string current(argv[idx]);
	if(keyword.compare(current)!=0)
		return idx;

	++idx;

	// fill the argument with content of [argv[idx], argv[idx+contentSize())
	istringstream ss;
	for(unsigned i=0;i<size();++i,++idx){
		string value(argv[idx]);
		ss.clear();
		ss.str(value);
		ss >> content[i];
	}

	return idx;
}

template <class T>
void Argument<T>::print(std::ostream &os) const{
	if(isconfigured){
		os << keyword;
		for(unsigned i=0;i<content.size();++i)
			os << " " << content[i];

	}else{
		BaseArgument::printDescription(os);

		for(unsigned i=0;i<BaseArgument::TABLENGTH;++i)
			os << " ";
		os << size() << "x <";

		T obj;
		os << type_name_of(obj) <<  ">";
		os << std::endl;
	}
}

/*************************************************************************************************/

template <class T>
class RandomLengthArgument:public Argument<T>{
public:
	RandomLengthArgument(const string &description, const string &keyword, const deque<T> &content);
	RandomLengthArgument(const vector<string> &description, const string &keyword, const deque<T> &content);
	virtual unsigned read(char **argv, unsigned idx);
	virtual void print(std::ostream &os) const;
	virtual ~RandomLengthArgument(){}
};

template <class T>
RandomLengthArgument<T>::RandomLengthArgument(const string &description, const string &keyword, const deque<T> &content)
:Argument<T>(description,keyword,content){}

template <class T>
RandomLengthArgument<T>::RandomLengthArgument(const vector<string> &description, const string &keyword, const deque<T> &content)
:Argument<T>(description,keyword,content){}

template <class T>
unsigned RandomLengthArgument<T>::read(char **argv, unsigned idx){
	string current(argv[idx]);
	if(Argument<T>::getKeyword().compare(current)!=0)
		return idx;
	++idx;

	// read length of arguments
	unsigned len;
	istringstream ss;
	string lenstr(argv[idx]);
	ss.clear();
	ss.str(lenstr);
	ss >> len;
	Argument<T>::content.resize(len);
	++idx;

	// fill the argument with content of [argv[idx], argv[idx+contentSize())
	for(unsigned i=0;i<Argument<T>::size();++i,++idx){
		string value(argv[idx]);
		ss.clear();
		ss.str(value);
		ss >> Argument<T>::content[i];
	}

	return idx;
}

template <class T>
void RandomLengthArgument<T>::print(std::ostream &os) const{
	if(Argument<T>::config()){
		Argument<T>::print(os);
	}else{
		BaseArgument::printDescription(os);

		for(unsigned i=0;i<BaseArgument::TABLENGTH;++i)
			os << " ";
		os << "<unsigned=arglen> ";
		os << "<arglen>x <";

		T obj;
		os << type_name_of(obj) <<  ">";
		os << std::endl;
	}
}

/*************************************************************************************************/

/**
 * Attempts to parse the candidate arguments from the CLI arguments, starting at argv[idx].
 * Returns the index after which CLI parsing stopped.
 */
unsigned ParseCLI(char **argv, unsigned idx, unsigned argc, const deque<BaseArgument*> &candidates);

/*************************************************************************************************/

} // CLI namespace

/*************************************************************************************************/

#endif /* CLI_H_ */
