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
 * DataFile.h
 *
 *      Author: Marc Claesen
 */

/*************************************************************************************************/

#ifndef DATAFILE_H_
#define DATAFILE_H_

/*************************************************************************************************/

#include "SparseVector.hpp"
#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include <deque>
#include <set>
#include <vector>
#include <list>

/*************************************************************************************************/

#define MAX_BUFFER_SIZE 5000;

using std::unique_ptr;

/*************************************************************************************************/

namespace ensemble{

/*************************************************************************************************/

/**
 * Class to model files that allow fast retrieval of specified rows.
 * These files are never loaded into memory entirely and serve to handle big data problems.
 *
 * A file index saves the positions of lines in the backing file.
 * This allows fast on-demand retrieval of lines.
 */
class IndexedFile{
public:
	typedef std::deque<std::streamoff> RowIndex;

protected:
	mutable std::ifstream file;
	RowIndex index;

public:
	IndexedFile(const std::string &fname);

	/**
	 * Returns the specified row.
	 *
	 * row must be between 1 and size().
	 */
	std::string operator[](unsigned row) const;

	/**
	 * Returns the amount of rows in this indexed file.
	 */
	unsigned size() const;
	~IndexedFile();
};

/*************************************************************************************************/

class DataLine final{
private:
	unique_ptr<std::string> label;
	unique_ptr<SparseVector> sv;
	bool islabeled;

public:
	DataLine(unique_ptr<std::string> label, unique_ptr<SparseVector> sv);
	DataLine(unique_ptr<SparseVector> sv);
	DataLine(DataLine&& o);

	bool labeled() const;
	unique_ptr<SparseVector> getSV();
	unique_ptr<std::string> getLabel();

	const SparseVector *rawSV() const;
	const std::string *rawLabel() const;

	~DataLine();
};

class ConstDataLine final{
private:
	const std::string* label;
	const SparseVector* sv;
	bool islabeled;

public:
	ConstDataLine(const std::string* label, const SparseVector* sv);
	ConstDataLine(const SparseVector* sv);
	ConstDataLine(ConstDataLine&& o);

	bool labeled() const;

	const SparseVector *rawSV() const;
	const std::string *rawLabel() const;

	~ConstDataLine();
};

/*************************************************************************************************/

class DataFile{
public:
	typedef std::deque<std::unique_ptr<SparseVector>> Instances;
	typedef Instances::iterator iterator;
	typedef Instances::const_iterator const_iterator;

protected:

	/**
	 * Dummy constructor, used by subclasses.
	 * Does nothing.
	 */
	DataFile();
	Instances instances;

	DataFile &operator=(const DataFile &orig);
	DataFile(const DataFile &orig);

	static unique_ptr<DataFile> readf(std::istream &iss, int format=0);

public:
	DataFile(const std::string &filename);
	virtual const SparseVector *operator[](unsigned instance) const;
	virtual std::shared_ptr<ConstDataLine> getdataline(unsigned instance) const;

	/**
	 * Returns the amount of test instances in this file.
	 */
	virtual unsigned size() const;
	virtual ~DataFile();

	static unique_ptr<DataFile> readf(const std::string &fname, int format=0);

	/**
	 * Reads an unlabeled comma seperated file of the following format (p dimensional problem)
	 *
	 * <value 1>,...,<value p>\n
	 */
	static unique_ptr<DataFile> readCSV(std::istream &iss);

	/**
	 * Reads an unlabeled comma seperated file of the following format (p dimensional problem)
	 *
	 * <value idx>:<value>,...,<value idx>:<value>\n
	 * with 1 <= <value idx> <= p
	 */
	static unique_ptr<DataFile> readSparseCSV(std::istream &iss);

	static unique_ptr<DataLine> readline(const std::string &line, int format=0);
};

/*************************************************************************************************/

class LabeledDataFile:public DataFile{
public:
	typedef std::string Label;

	class LabelSort{
	public:
		bool operator()(const std::unique_ptr<Label>& v1, const std::unique_ptr<Label>& v2) const{
			return (*v1.get())<(*v2.get());
		}
	};

	typedef std::deque< std::pair<const SparseVector*,const Label*>> LabelMap;
	typedef std::set<std::unique_ptr<Label>,LabelSort> LabelSet;
	typedef LabelSet::iterator label_iterator;
	typedef LabelSet::const_iterator const_label_iterator;

protected:
	/**
	 * Dummy constructor, used by subclasses.
	 * Does nothing.
	 */
	LabeledDataFile();
	LabelSet labels;
	LabelMap labelmap;

	/**
	 * Attempts to add label to the set of labels. The LabeledDataFile acquires ownership of the label.
	 */
	const Label *addLabel(unique_ptr<Label> label);
	static unique_ptr<LabeledDataFile> readf(std::istream &iss, int format=0, const std::deque<unsigned> *indices=NULL);

public:
//	LabeledDataFile(const std::string &filename);

	/**
	 * Returns the label associated with instance. Returns NULL if instance is not part of the data file.
	 */
	virtual const Label *getLabel(unsigned instance) const;
	virtual ~LabeledDataFile();

	/**
	 * Reads a LabeledDataFile from fname with specified format.
	 *
	 * If a list of indices is specified (e.g. not NULL), only those line numbers are read.
	 */
	static unique_ptr<LabeledDataFile> readf(const std::string &fname, int format=0, const std::deque<unsigned> *indices=NULL);

	/**
	 * Reads a labeled comma seperated file of the following format (p dimensional problem)
	 *
	 * <label>,<value 1>,...,<value p>\n
	 */
	static unique_ptr<LabeledDataFile> readCSV(std::istream &iss);

	/**
	 * Reads a comma seperated file of the following format (p dimensional problem)
	 *
	 * <label>,<value idx>:<value>,...,<value idx>:<value>\n
	 * with 1 <= <value idx> <= p
	 */
	static unique_ptr<LabeledDataFile> readSparseCSV(std::istream &iss);

	/**
	 * Reads label and vector from specified line with given format.
	 */
	static unique_ptr<DataLine> readline(const std::string &line, int format=0);

	/**
	 * Returns the given DataLine.
	 */
	virtual std::shared_ptr<ConstDataLine> getdataline(unsigned instance) const;
};

/*************************************************************************************************/

}

/*************************************************************************************************/

#endif /* DATAFILE_H_ */
