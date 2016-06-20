/*
 * nr1.cc
 *
 *  Created on: May 5, 2014
 *      Author: richard
 */
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdlib.h>

using namespace std;
using namespace cv;

int main(int argc, const char* argv[]) {
	
	/*if (argc != 3) {
		std::cout << "usage: " << argv[0] << " <model> <image>" << std::endl;
		exit(1);
	}
	*/

	String cascade_name = argv[1];
	vector<String> image_names={"img1.jpg", "img2.jpg", "img3.jpg"};
	std::ofstream f;
	f.open("result-faceDetector.txt");
	for (auto image_name: image_names){
		//String image_name= argv[2];
		CascadeClassifier cascade;
		if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
		// detection
		Mat image=imread(image_name, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("Original image", image);
		waitKey(0);
		std::vector<Rect> faces;
		cascade.detectMultiScale( image, faces);
		f<<image_name<<": "<<faces.size()<<" faces found.\n";
		
		for( size_t i = 0; i < faces.size(); i++ ) {
			rectangle(image, faces[i], Scalar(255));		
		}


		imshow("Detections", image);
		waitKey(0);
	}
	f.close();
	return 0;
}
