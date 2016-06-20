all: sheet03_faceDetector sheet03_adaBoost sheet03_tracking

sheet03_faceDetector:
	g++ sheet03_faceDetector.cc -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -o sheet03_faceDetector

sheet03_adaBoost:
	g++ sheet03_adaBoost.cc AdaBoost.cc WeakClassifier.cc -o sheet03_adaBoost

sheet03_tracking:
	g++ sheet03_tracking.cc AdaBoost.cc WeakClassifier.cc -lopencv_core -lopencv_highgui -lopencv_imgproc -o sheet03_tracking
