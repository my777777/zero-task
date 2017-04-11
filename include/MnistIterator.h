#ifndef  MNIST_ITERATOR_H
#define MNIST_ITERATOR_H
#include "Resource.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;

struct MNISTImageFileHeader {
	unsigned char MagicNumber[4];
	unsigned char NumberOfImages[4];
	unsigned char NumberOfRows[4];
	unsigned char NumberOfColums[4];
};

struct MNISTLabelFileHeader {
	unsigned char MagicNumber[4];
	unsigned char NumberOfLabels[4];
};

const int MAGICNUMBEROFIMAGE = 2051;
const int MAGICNUMBEROFLABEL = 2049;


class MnistIterator {

private:
	cv::Mat mTrainImages;
	cv::Mat mTrainLabels;
	cv::Mat mTestImages;
	cv::Mat mTestLabels;
	int mTrainBatchSize;
	int mTestBatchSize;
	int mTotalTrainBatch;
	int mTotalTestBatch;
	int mTotalTestImages;
	int mTotalTrainImages;
	int mImageSize;
	int mNumberOfClass;

public:

	MnistIterator() ;
	MnistIterator(int trainBatchSize, int testBatchSize, string trainImagePath,string trainLabelPath, string testImagePath, string testLabelPath) ;
	~MnistIterator();

	cv::Mat ReadImages(const std::string& FileName);
	cv::Mat ReadLabels(const std::string& FileName);

	void nextTrainBatch(int index,double* nextBatch);
	void nextTestBatch(int index,double* nextBatch);
	void nextTrainLabelBatch(int index,double* lablel);
	void nextTestLabelBatch(int index,double* lablel);

	int ConvertCharArrayToInt(unsigned char* array, int LengthOfArray);
	bool IsImageDataFile(unsigned char* MagicNumber, int LengthOfArray);
	bool IsLabelDataFile(unsigned char* MagicNumber, int LengthOfArray);
	cv::Mat ReadData(std::fstream& DataFile, int NumberOfData,	int DataSizeInBytes);
	cv::Mat ReadImageData(std::fstream& ImageDataFile, int NumberOfImages);
	cv::Mat ReadLabelData(std::fstream& LabelDataFile, int NumberOfLabel);

};

#endif // MNIST_ITERATOR_H
