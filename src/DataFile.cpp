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
 * Kernel.cpp
 *
 *      Author: Marc Claesen
 */

#include "DataFile.hpp"
#include "SparseVector.hpp"
#include "Util.hpp"
#include "io.hpp"
#include <string>
#include <iostream>
#include <sstream>
#include <memory>
#include <algorithm>

using std::endl;
using std::string;
using std::unique_ptr;
using std::istringstream;

namespace ensemble{

IndexedFile::IndexedFile(const string &fname){
	file.open(fname.c_str(),std::ios::in);

	// start of file
	index.push_back(static_cast<std::streamoff>(file.tellg()));

	std::string line;
	while(file.good()){
		getline(file,line);
		index.push_back(static_cast<std::streamoff>(file.tellg()));
	}

	// remove the last streamoff if it's -1 (e.g. EOF)
	if(*index.rbegin()==-1)
		index.pop_back();

	// remove last element if its distance to the end of file is zero
	std::streampos dist=file.seekg(0,std::ios::end).tellg()-file.seekg(*index.rbegin(),std::ios::beg).tellg();
	if(dist==0)
		index.pop_back();
}
IndexedFile::~IndexedFile(){ file.close(); }
unsigned IndexedFile::size() const{ return index.size(); }
std::string IndexedFile::operator[](unsigned row) const{
	file.clear();
	if(row>0 && size()>=row){
		file.seekg(index[row-1],std::ios::beg);
	}else{
		std::ostringstream oss;
		oss << "Invalid rowindex when reading IndexedFile: ";
		oss << row << " (size=" << size() << ").";
		exit_with_err(oss.str());
	}

	std::string line;
	getline(file,line);
	return std::move(line);
}

DataLine::DataLine(unique_ptr<std::string> label, unique_ptr<SparseVector> sv)
:label(std::move(label)),
 sv(std::move(sv)),
 islabeled(true)
{}
DataLine::DataLine(unique_ptr<SparseVector> sv)
:label(nullptr),
 sv(std::move(sv)),
 islabeled(false)
{}
DataLine::DataLine(DataLine&& o)
:label(std::move(o.label)),
 sv(std::move(o.sv)),
 islabeled(o.islabeled)
{}
DataLine::~DataLine(){}

bool DataLine::labeled() const{ return islabeled; }
unique_ptr<SparseVector> DataLine::getSV(){ return std::move(sv); }
unique_ptr<std::string> DataLine::getLabel(){ return std::move(label); }

const SparseVector *DataLine::rawSV() const{ return sv.get(); }
const std::string *DataLine::rawLabel() const{ return label.get(); }

ConstDataLine::ConstDataLine(const std::string* lab, const SparseVector* vec)
:label(lab),
 sv(vec),
 islabeled(true)
{}
ConstDataLine::ConstDataLine(const SparseVector* vec)
:label(nullptr),
 sv(vec),
 islabeled(false)
{}
ConstDataLine::ConstDataLine(ConstDataLine&& o)
:label(o.label),
 sv(o.sv),
 islabeled(o.islabeled)
{}
ConstDataLine::~ConstDataLine(){}

bool ConstDataLine::labeled() const{ return islabeled; }
const SparseVector *ConstDataLine::rawSV() const{ return sv; }
const std::string *ConstDataLine::rawLabel() const{ return label; }

DataFile::DataFile(){}
unsigned DataFile::size() const{ return instances.size(); }

DataFile::~DataFile(){}
DataFile::DataFile(const string &fname){
	istringstream liness;
	std::ifstream file;
	string line, junk;

	file.open(fname.c_str(),std::ios::in);

	while(file.good()){
		// first part of each line is the label, discard this for now
		file >> junk;

		line.clear();
		getline(file,line);
		if(line.empty())
			break;

		liness.clear();
		liness.str(line);
		instances.emplace_back(SparseVector::read(liness));
	}

	file.close();
}

const SparseVector *DataFile::operator[](unsigned idx) const{ return instances.at(idx).get(); }
std::shared_ptr<ConstDataLine> DataFile::getdataline(unsigned idx) const{
	return std::make_shared<ConstDataLine>(operator[](idx));
}

LabeledDataFile::LabeledDataFile():DataFile(){}

const LabeledDataFile::Label *LabeledDataFile::getLabel(unsigned instance) const{
	return labelmap.at(instance).second;
}
std::shared_ptr<ConstDataLine> LabeledDataFile::getdataline(unsigned idx) const{
	return std::make_shared<ConstDataLine>(getLabel(idx),operator[](idx));
}

LabeledDataFile::~LabeledDataFile(){}

const LabeledDataFile::Label *LabeledDataFile::addLabel(unique_ptr<Label> label){
	LabelSet::const_iterator F=labels.find(label);

	if(F==labels.end()){
		// label not present in set, clone it and add
		Label *ptr=label.get();
		labels.insert(std::move(label));
		return ptr;
	}

	// label already present in set, let unique_ptr destroy it
	return F->get();
}

/**
 * Data file reading
 */
unique_ptr<DataFile> DataFile::readf(const std::string &fname, int format){
	std::ifstream file(fname.c_str(),std::ios::in);

	unique_ptr<DataFile> datafile(DataFile::readf(file,format));
	file.close();
	return datafile;
}


unique_ptr<DataFile> DataFile::readf(std::istream &iss, int format){
	unique_ptr<DataFile> datafile(new DataFile());

	string line;
	while(getline(iss,line)){
		unique_ptr<DataLine> dataline(DataFile::readline(line,format));
		datafile->instances.emplace_back(dataline->getSV());
	}
	return datafile;
}


unique_ptr<DataFile> DataFile::readCSV(std::istream &iss){
	int format=FileFormats::CSV;
	return DataFile::readf(iss,format);
}
unique_ptr<DataFile> DataFile::readSparseCSV(std::istream &iss){
	int format=FileFormats::SparseCSV;
	return DataFile::readf(iss,format);
}

unique_ptr<DataLine> DataFile::readline(const std::string &line, int format){
	std::istringstream linestr(line,std::ios::in);
	unique_ptr<SparseVector> sv=SparseVector::readf(linestr,format);
	return unique_ptr<DataLine>(new DataLine(std::move(sv)));
}

unique_ptr<LabeledDataFile> LabeledDataFile::readf(const std::string &fname, int format, const std::deque<unsigned> *indices){
	std::ifstream file(fname.c_str(),std::ios::in);

	unique_ptr<LabeledDataFile> datafile(LabeledDataFile::readf(file,format,indices));
	file.close();
	return datafile;
}

unique_ptr<LabeledDataFile> LabeledDataFile::readf(std::istream &iss, int format, const std::deque<unsigned> *indices){
	unique_ptr<LabeledDataFile> datafile(new LabeledDataFile());

	std::deque<unsigned> sortedindices;
	std::deque<unsigned>::const_iterator Iind;
	if(indices!=nullptr){
		sortedindices=*indices;
		std::sort(sortedindices.begin(),sortedindices.end());
		Iind=sortedindices.begin();
	}

	string line;
	std::istringstream linestr;
	unsigned linenum=1;
	while(getline(iss,line)){
		if(indices!=nullptr && Iind==sortedindices.end())
			break;

		if(indices==nullptr || linenum==*Iind){
			if(indices!=nullptr)
				++Iind;

			unique_ptr<DataLine> dataline(LabeledDataFile::readline(line,format));
			unique_ptr<SparseVector> sv=dataline->getSV();
			unique_ptr<Label> label=dataline->getLabel();
			const SparseVector* ptr=sv.get();

			datafile->instances.emplace_back(std::move(sv));
			const Label *thislabel=datafile->addLabel(std::move(label));
			datafile->labelmap.push_back(std::make_pair(ptr,thislabel));

		}

		++linenum;
	}

	return datafile;
}
unique_ptr<DataLine> LabeledDataFile::readline(const std::string &line, int format){
	char delim=*",";
	if(format==FileFormats::DEFAULT)
		delim=*" ";

	std::istringstream liness(line);

	unique_ptr<Label> label(new Label());

	if(format!=FileFormats::DEFAULT && format!=FileFormats::CSV && format!=FileFormats::SparseCSV)
		exit_with_err("Unknown file format.");

	getline(liness,*label,delim);
	unique_ptr<SparseVector> sv=SparseVector::readf(liness,format);

	return unique_ptr<DataLine>(new DataLine(std::move(label),std::move(sv)));
}

unique_ptr<LabeledDataFile> LabeledDataFile::readCSV(std::istream &iss){
	int format=FileFormats::CSV;
	return LabeledDataFile::readf(iss,format);
}
unique_ptr<LabeledDataFile> LabeledDataFile::readSparseCSV(std::istream &iss){
	int format=FileFormats::SparseCSV;
	return LabeledDataFile::readf(iss,format);
}

} // ensemble namespace
