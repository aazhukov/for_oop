/*
 * nr3.cc
 *
 *  Created on: Apr 28, 2014
 *      Author: richard
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
 #include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <sstream>
#include "Types.hh"
#include "AdaBoost.hh"

#define _objectWindow_width 121
#define _objectWindow_height 61

#define _searchWindow_width 61
#define _searchWindow_height 61

// use 30/15 for overlapping negative examples and 120/60 for non-overlapping negative examples
#define _displacement_x 30 //120
#define _displacement_y 15 //60
using namespace cv;
using namespace std;
// helper function
void loadImage(const std::string& imageFile, cv::Mat& image) {
	image = cv::imread(imageFile, CV_LOAD_IMAGE_GRAYSCALE);
	//cout<<imageFile<<endl;
	//imshow("blah", image);
	//waitKey(0);
	if(!image.data ) {
		std::cout <<  "Could not open or find the image" << std::endl ;
		exit(1);
	}
}

void loadTrainFrames(const char* trainDataFile, std::vector<cv::Mat>& imageSequence,
		std::vector<cv::Point>& referencePoints) {
	imageSequence.clear();
	referencePoints.clear();
	std::ifstream f(trainDataFile);
	std::string line;
	while (getline(f, line)) {
		std::stringstream s(line);
		std::string imageFile;
		s >> imageFile;
		referencePoints.push_back(cv::Point());
		s >> referencePoints.back().x;
		s >> referencePoints.back().y;
		imageSequence.push_back(cv::Mat());
		loadImage(imageFile, imageSequence.back());

	}
	f.close();
}

void loadTestFrames(const char* testDataFile, std::vector<cv::Mat>& imageSequence, cv::Point& startingPoint) {
	imageSequence.clear();
	std::ifstream f(testDataFile);
	std::string line;
	getline(f, line);
	std::stringstream s(line);
	s >> startingPoint.x;
	s >> startingPoint.y;
	while (getline(f, line)) {
		imageSequence.push_back(cv::Mat());
		loadImage(line, imageSequence.back());
	}
	f.close();
}

// TODOS
void computeHistogram(const cv::Mat& image, const cv::Point& p, Vector& histogram) {
	histogram.clear();
	histogram.resize(256, 0);
	Rect region_of_interest = Rect(p.x-(_objectWindow_width/2), (p.y-_objectWindow_height/2), _objectWindow_width, _objectWindow_height);
	Mat image_roi = image(region_of_interest);	
	//Mat imcp=image.clone();
	//cv::rectangle(imcp, region_of_interest, cv::Scalar(255));
 	//imshow("blah", imcp);
	//waitKey(2);
 	for (int y=0; y < image_roi.rows; ++y) {
		for (int x=0;  x < image_roi.cols; ++x) {
			histogram[image_roi.at<uchar>(y,x)]+=1.0;
		}
	}
}

void generateTrainingData(std::vector<Example>& data, const std::vector<cv::Mat>& imageSequence, const std::vector<cv::Point>& referencePoints) {

	data.clear();

	for(auto i=0; i< imageSequence.size(); i++){
		Example ex;

		// generate positive examples ... compute histogram around each training point
		ex.label=1;
		computeHistogram(imageSequence[i], referencePoints[i], ex.attributes);
		data.push_back(ex);
		
		ex.label=0;	
		Point p;
		p=referencePoints[i];
		p.x+=60;
		p.y+=30;
		computeHistogram(imageSequence[i], p, ex.attributes);
		data.push_back(ex);
		p=referencePoints[i];
		p.x-=60;
		p.y+=30;
		computeHistogram(imageSequence[i], p, ex.attributes);
		data.push_back(ex);
		p=referencePoints[i];
		p.x+=60;
		p.y-=30;
		computeHistogram(imageSequence[i], p, ex.attributes);
		data.push_back(ex);
		p=referencePoints[i];
		p.x-=60;
		p.y-=30;
		computeHistogram(imageSequence[i], p, ex.attributes);
		data.push_back(ex);
	}
	
}

void findBestMatch(const cv::Mat& image, cv::Point& lastPosition, AdaBoost& adaBoost) {
	f32 maxConf = -100000;
	f32 curr_conf=0;
	cv::Point bestMatch, search_point;
	for(int x=-30; x<=30; x++){
		for(int y=-30; y<=30; y++){
			Vector v;
			search_point=lastPosition;
			search_point.x+=x;
			search_point.y+=y;
			computeHistogram(image, search_point, v);
			curr_conf=adaBoost.confidence(v, 1);
			if (curr_conf> maxConf){
				maxConf=curr_conf;
				bestMatch=search_point;
			}
		}
	}
	lastPosition = bestMatch;
}

void drawTrackedFrame(cv::Mat& image, cv::Point& position) {
	Mat imcp = image.clone();
	Rect r = Rect(position.x-(_objectWindow_width/2), (position.y-_objectWindow_height/2), _objectWindow_width, _objectWindow_height);
	cv::rectangle(imcp, r, cv::Scalar(255));
	imshow("Finding Nemo", imcp);
	waitKey(0);


}

int main( int argc, char** argv )
{
	if(argc != 4) {
		std::cout <<" Usage: " << argv[0] << " <training-file> <test-file> <# iterations for AdaBoost>" << std::endl;
		return -1;
	}

	u32 adaBoostIterations = atoi(argv[3]);

	// load the training frames
	std::vector<cv::Mat> imageSequence;
	std::vector<cv::Point> referencePoints;
	loadTrainFrames(argv[1], imageSequence, referencePoints);

	// generate gray-scale histograms from the training frames:
	// one positive example per frame (_objectWindow_width x _objectWindow_height window around reference point for object)
	// four negative examples per frame (with _displacement_{x/y} + small random displacement from reference point)
	std::vector<Example> trainingData;
	generateTrainingData(trainingData, imageSequence, referencePoints);
//	computeHistogram(imageSequence[0], referencePoints[0], histogram, a, b);
	
	// initialize AdaBoost and train a cascade with the extracted training data
	AdaBoost adaBoost(adaBoostIterations);
	adaBoost.initialize(trainingData);
	adaBoost.trainCascade(trainingData);

	u32 nClassificationErrors = 0;
	for(u32 i=0; i< trainingData.size(); i++)
			nClassificationErrors+= (adaBoost.classify(trainingData[i].attributes) != trainingData[i].label);

	// TODO: compute classification error, your code here ...

	std::cout << "Error rate on training set: " << (f32)nClassificationErrors / (f32)trainingData.size() << std::endl;

	// load the test frames and the starting position for tracking
	std::vector<Example> testImages;
	cv::Point lastPosition;
	loadTestFrames(argv[2], imageSequence, lastPosition);

	std::ofstream f;
	f.open("result-tracking.txt");
	f<<"frame_number,x_position,y_position\n";
	// for each frame...
	for (u32 i = 0; i < imageSequence.size(); i++) {
		// ... find the best match in a window of size
		// _searchWindow_width x _searchWindow_height around the last tracked position
		findBestMatch(imageSequence.at(i), lastPosition, adaBoost);
		// draw the result
		f<<i<<","<<lastPosition.x<<","<<lastPosition.y<<"\n";
		drawTrackedFrame(imageSequence.at(i), lastPosition);
	}
	f.close();
	return 0;
}
