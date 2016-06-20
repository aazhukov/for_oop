/*
 * NearestMeanClassifier.cc
 *
 *  Created on: Apr 25, 2014
 *      Author: richard
 */

#include "WeakClassifier.hh"
#include <cmath>
#include <iostream>
#include <limits>

/*
 * Stump
 */

Stump::Stump() :
		dimension_(0),
		splitAttribute_(0),
		splitValue_(0),
		classLabelLeft_(0),
		classLabelRight_(0)
{}

void Stump::initialize(u32 dimension) {
	dimension_ = dimension;
}


f32 Stump::weightedGain(const std::vector<Example>& data, const Vector& weights, u32 splitAttribute, f32 splitValue, u32 resultingLeftLabel) {
	f32 weight = 0;
	std::vector<u32> predictions;
	for(u32 i=0; i<data.size(); i++){
		weight+=weights[i]* (data[i].label ==  ((data[i].attributes[splitAttribute]<splitValue) ? resultingLeftLabel: (1 - resultingLeftLabel)));
	}
	return weight; 
}

void Stump::train(const std::vector<Example>& data, const Vector& weights) {
	f32 best_gain = std::numeric_limits<f32>::infinity();
	
	for (u32 curr_split_attr = 0; curr_split_attr< data[0].attributes.size(); curr_split_attr++){
		for(u32 curr_split_idx=0; curr_split_idx<data.size(); curr_split_idx++){
			f32 curr_split_value=data[curr_split_idx].attributes[curr_split_attr];
			for (auto left_label: {0, 1}){
				f32 curr_gain=weightedGain(data, weights, curr_split_attr, curr_split_value, left_label);
				if (curr_gain< best_gain){
					best_gain=curr_gain;
					classLabelLeft_=left_label;
					splitValue_=curr_split_value;
					splitAttribute_=curr_split_attr;
				}
			}
		}
	}
	classLabelRight_=1 -classLabelLeft_;

}

u32 Stump::classify(const Vector& v) {
	u32 label = 0;

	label= ((v[splitAttribute_] < splitValue_) ? classLabelLeft_ : classLabelRight_);
	return label;
}

void Stump::classify(const std::vector<Example>& data, std::vector<u32>& classAssignments) {
	classAssignments.resize(data.size());
	for (u32 i = 0; i < data.size(); i++) {
		classAssignments.at(i) = classify(data.at(i).attributes);
	}
}
