#include <iostream>
#include <string.h>
#include <fstream>
#include <cmath>
#include <stdlib.h>
#include "Types.hh"
#include "AdaBoost.hh"

#include <typeinfo>

using namespace std;
void readData(const char* filename, std::vector<Example>& data) {
	std::ifstream f(filename);
	u32 dimension, nObservations;
	f >> dimension;
	f >> nObservations;
	data.resize(nObservations);
	for (u32 i = 0; i < nObservations; i++) {
		f >> data.at(i).label;
		data.at(i).attributes.resize(dimension);
		for (u32 d = 0; d < dimension; d++) {
			f >> data.at(i).attributes.at(d);
		}
	}
	f.close();
}

int main(int argc, const char* argv[]) {
 	/*
	if (argc != 4) {
		std::cout << "usage: " << argv[0] << " <train-set> <test-set> <# iterations>" << std::endl;
		exit(1);
	}
	*/

	//const char* trainFile = argv[1];
	//const char* testFile = argv[2];
	//u32 adaBoostIterations = atoi(argv[3]);
	const char* trainFile = "splice/splice.train";
	const char* testFile = "splice/splice.test";
	
	std::ofstream f;
	f.open("result-AdaBoost.txt");
	for(u32 adaBoostIterations = 0; adaBoostIterations< 50; adaBoostIterations++){
		//u32 adaBoostIterations = atoi(argv[1]);
		std::vector<Example> trainingData;
		std::vector<Example> testData;
		readData(trainFile, trainingData);
		readData(testFile, testData);
		// train cascade of weak classifiers
		AdaBoost adaBoost(adaBoostIterations);
		adaBoost.initialize(trainingData);
		adaBoost.trainCascade(trainingData);
		// classification on test data
		u32 nClassificationErrors = 0;

		// TODO: compute classification error, your code here ...
		for(u32 i=0; i< testData.size(); i++)
			nClassificationErrors+= (adaBoost.classify(testData[i].attributes) != testData[i].label);
		f32 accuracy = 1.0 - (f32) nClassificationErrors / (f32) testData.size();

		std::cout << "Classified " << testData.size() << " examples." << std::endl;
		std::cout << "Accuracy: " << accuracy << " (" << testData.size() - nClassificationErrors << "/" << testData.size() << ")" << std::endl;
		f<<adaBoostIterations<<","<<accuracy<<"\n";
	}
	f.close();
	return 0;
}
