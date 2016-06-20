/*
 * AdaBoost.cc
 *
 *  Created on: Apr 25, 2014
 *      Author: richard
 */

#include "AdaBoost.hh"
#include <iostream>
#include <cmath>

AdaBoost::AdaBoost(u32 nIterations) :
	nIterations_(nIterations)
{}

void AdaBoost::normalizeWeights() {
	f32 sum_weights=0;
	for(u32 i =0; i < weights_.size(); i++)
		sum_weights+=weights_[i];
	for(u32 i =0; i < weights_.size(); i++)
		weights_[i]/=sum_weights;
}

	


void AdaBoost::updateWeights(const std::vector<Example>& data, const std::vector<u32>& classAssignments, u32 iteration) {
	for(u32 i = 0; i< data.size(); i++)
		weights_[i]*=std::pow(weightedErrorRate(data, classAssignments), static_cast<f32>(classAssignments[i]==data[i].label));
}

f32 AdaBoost::weightedErrorRate(const std::vector<Example>& data, const std::vector<u32>& classAssignments) {
	f32 errorRate = 0;
	for(u32 i = 0; i< data.size(); i++)
		errorRate+=(data[i].label != classAssignments[i]) * weights_[i];
	errorRate/=(1 - errorRate);
	return errorRate;
}

void AdaBoost::initialize(std::vector<Example>& data) {
	
	for(u32 i = 0; i < nIterations_; i++){
		weakClassifier_.push_back(Stump());						 // initialize weak classifiers
		classifierWeights_.push_back(0.0);     					 // initialize classifier weights
	}
	for(int i =0; i< data.size(); i++)
		weights_.push_back(1.0/static_cast<f32>(data.size())); 	 // initialize weights
	//for(auto a: weights_) std::cout<< a* data.size()<<"  ";
}

void AdaBoost::trainCascade(std::vector<Example>& data) {

	u32 dimension = data.at(0).attributes.size();
	std::vector<u32> classAssignments;

	for (u32 iteration = 0; iteration < nIterations_; iteration++) {
		weakClassifier_[iteration].train(data, weights_); 								// train weak classifier
		weakClassifier_[iteration].classify(data, classAssignments);  // classify training examples
		classifierWeights_[iteration]= std::log(1/weightedErrorRate(data, classAssignments));   // determine weight for weak classifier
		updateWeights( data, classAssignments, iteration);
		normalizeWeights();
	}
}

u32 AdaBoost::classify(const Vector& v) {
	if (confidence(v, 0) > confidence(v, 1))
		return 0;
	else
		return 1;
}

f32 AdaBoost::confidence(const Vector& v, u32 k) {
	f32 score = 0;
	for(u32 i=0; i< nIterations_; i++)
		score+=classifierWeights_[i]* (weakClassifier_[i].classify(v)==k);
	return score;
}
