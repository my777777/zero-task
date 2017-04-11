#include "MnistIterator.h"


MnistIterator::MnistIterator() {

	mTrainBatchSize = gi_batchSize;
	mTestBatchSize = gi_batchSize;
	mTrainImages = ReadImages(gs_trainImagePath);
	mTrainLabels = ReadLabels(gs_trainLabelPath);
	mTestImages = ReadImages(gs_testImagePath);
	mTestLabels = ReadLabels(gs_testLabelPath);

	mImageSize = gi_imageWidth * gi_imageHeight;
	mTotalTestImages = gi_totalTestImages;
	mTotalTrainImages = gi_totalTrainImages;
	mNumberOfClass = gi_mnistNumberOfClass;
}

MnistIterator::~MnistIterator() {
}

MnistIterator::MnistIterator(int trainBatchSize, int testBatchSize, string trainImagePath, string trainLabelPath, string testImagePath, string testLabelPath) :
		mTrainBatchSize(trainBatchSize), mTestBatchSize(testBatchSize) {

	mTrainImages = ReadImages(trainImagePath);
	mTrainLabels = ReadLabels(trainLabelPath);
	mTestImages = ReadImages(testImagePath);
	mTestLabels = ReadLabels(testLabelPath);

	mImageSize = gi_imageWidth * gi_imageHeight;
	mTotalTestImages = gi_totalTestImages;
	mTotalTrainImages = gi_totalTrainImages;
	//mTotalTrainBatch=mTotalTrainImages/trainBatchSize + 1;
	//mTotalTestBatch=mTotalTestImages/testBatchSize + 1;

}

int MnistIterator::ConvertCharArrayToInt(unsigned char* array, int LengthOfArray) {
	if (LengthOfArray < 0) {
		return -1;
	}
	int result = static_cast<signed int>(array[0]);
	for (int i = 1; i < LengthOfArray; i++) {
		result = (result << 8) + array[i];
	}
	return result;
}

/**
 * @brief IsImageDataFile  Check the input MagicNumber is equal to
 *                         MAGICNUMBEROFIMAGE
 * @param MagicNumber      The array of the magicnumber to be checked
 * @param LengthOfArray    The length of the array
 * @return true, if the magcinumber is mathed;
 *         false, otherwise.
 */
bool MnistIterator::IsImageDataFile(unsigned char* MagicNumber, int LengthOfArray) {
	int MagicNumberOfImage = ConvertCharArrayToInt(MagicNumber, LengthOfArray);
	if (MagicNumberOfImage == MAGICNUMBEROFIMAGE) {
		return true;
	}
	return false;
}

/**
 * @brief IsImageDataFile  Check the input MagicNumber is equal to
 *                         MAGICNUMBEROFLABEL
 * @param MagicNumber      The array of the magicnumber to be checked
 * @param LengthOfArray    The length of the array
 * @return true, if the magcinumber is mathed;
 *         false, otherwise.
 */
bool MnistIterator::IsLabelDataFile(unsigned char *MagicNumber, int LengthOfArray) {
	int MagicNumberOfLabel = ConvertCharArrayToInt(MagicNumber, LengthOfArray);
	if (MagicNumberOfLabel == MAGICNUMBEROFLABEL) {
		return true;
	}
	return false;
}

/**
 * @brief ReadData  Read the data in a opened file
 * @param DataFile  The file which the data is read from.
 * @param NumberOfData  The number of the data
 * @param DataSizeInBytes  The size fo the every data
 * @return The Mat which rows is a data,
 *         Return a empty Mat if the file is not opened or the some flag was
 *                 seted when reading the  data.
 */
cv::Mat MnistIterator::ReadData(std::fstream& DataFile, int NumberOfData, int DataSizeInBytes) {
	cv::Mat DataMat;
	// read the data if the file is opened.
	if (DataFile.is_open()) {
		int AllDataSizeInBytes = DataSizeInBytes * NumberOfData;
		unsigned char* TmpData = new unsigned char[AllDataSizeInBytes];
		DataFile.read((char *) TmpData, AllDataSizeInBytes);

		DataMat = cv::Mat(NumberOfData, DataSizeInBytes, CV_8UC1, TmpData).clone();
		delete[] TmpData;
		DataFile.close();
	}
	return DataMat;
}

/**
 * @brief ReadImageData  Read the Image data from the MNIST file.
 * @param ImageDataFile  The file which contains the Images.
 * @param NumberOfImages The number of the images.
 * @return The mat contains the image and each row of the mat is a image.
 *         Return empty mat is the file is closed or the data is not matching
 *                the number.
 */
cv::Mat MnistIterator::ReadImageData(std::fstream& ImageDataFile, int NumberOfImages) {
	int ImageSizeInBytes = 28 * 28;
	return ReadData(ImageDataFile, NumberOfImages, ImageSizeInBytes);
}

/**
 * @brief ReadLabelData Read the label data from the MNIST file.
 * @param LabelDataFile The file contained the labels.
 * @param NumberOfLabel The number of the labels.
 * @return The mat contains the labels and each row of the mat is a label.
 *         Return empty mat is the file is closed or the data is not matching
 *                the number.
 */
cv::Mat MnistIterator::ReadLabelData(std::fstream& LabelDataFile, int NumberOfLabel) {
	int LabelSizeInBytes = 1;
	return ReadData(LabelDataFile, NumberOfLabel, LabelSizeInBytes);
}

/**
 * @brief ReadImages Read the Training images.
 * @param FileName  The name of the file.
 * @return The mat contains the image and each row of the mat is a image.
 *         Return empty mat is the file is closed or the data is not matched.
 */
cv::Mat MnistIterator::ReadImages(const std::string& FileName) {

	std::fstream File(FileName.c_str(), std::ios_base::in | std::ios_base::binary);
	if (!File.is_open()) {
		cout << "File:" << FileName << " open failed!" << endl;
		exit(-1);
	}

	MNISTImageFileHeader FileHeader;
	File.read((char *) (&FileHeader), sizeof(FileHeader));

	if (!IsImageDataFile(FileHeader.MagicNumber, 4)) {
		return cv::Mat();
	}

	int NumberOfImage = ConvertCharArrayToInt(FileHeader.NumberOfImages, 4);
	return ReadImageData(File, NumberOfImage);
}

/**
 * @brief ReadLabels  Read the label from the MNIST file.
 * @param FileName  The name of the file.
 * @return The mat contains the image and each row of the mat is a image.
 *         Return empty mat is the file is closed or the data is not matched.
 */
cv::Mat MnistIterator::ReadLabels(const std::string& FileName) {
	std::fstream File(FileName.c_str(), std::ios_base::in | std::ios_base::binary);

	if (!File.is_open()) {
		cout << "File:" << FileName << " open failed!" << endl;
		exit(-1);
	}

	MNISTLabelFileHeader FileHeader;
	File.read((char *) (&FileHeader), sizeof(FileHeader));

	if (!IsLabelDataFile(FileHeader.MagicNumber, 4)) {
		return cv::Mat();
	}

	int NumberOfImage = ConvertCharArrayToInt(FileHeader.NumberOfLabels, 4);

	return ReadLabelData(File, NumberOfImage);
}

void MnistIterator::nextTestBatch(int index, double* nextBatch) {
	int beginIndex = index * mTestBatchSize;
	//endIndex,but not inlcude endIndex
	int endIndex = (index + 1) * mTestBatchSize;
	if (endIndex > mTotalTestImages) {
		beginIndex = mTotalTestImages - mTestBatchSize;
		endIndex = mTotalTestImages;
	}
	int imageSize = gi_imageHeight * gi_imageWidth;
	for (int i = beginIndex; i < endIndex; i++) {
		uchar *data = mTestImages.ptr<uchar>(i);
		for (int px = 0; px < imageSize; px++) {
			nextBatch[(i - beginIndex) * imageSize + px] = ((int) data[px]) * 0.00392157;
		}
	}
}

void MnistIterator::nextTrainBatch(int index, double* nextBatch) {
	int beginIndex = index * mTrainBatchSize;
	//endIndex,but not inlcude endIndex
	int endIndex = (index + 1) * mTrainBatchSize;

	if (endIndex > mTotalTrainImages) {
		beginIndex = mTotalTrainImages - mTrainBatchSize;
		endIndex = mTotalTrainImages;
	}
	int imageSize = gi_imageHeight * gi_imageWidth;
	for (int i = beginIndex; i < endIndex; i++) {
		uchar *data = mTrainImages.ptr<uchar>(i);
		for (int px = 0; px < imageSize; px++) {
			//nextBatch[(i-beginIndex) * imageSize + px] = ((int)data[px])==0?0:1;
			//nextBatch[(i-beginIndex) * imageSize + px] =  ((int) data[px]) / (255.0);
			nextBatch[(i - beginIndex) * imageSize + px] = ((int) data[px]) * 0.00392157;

		}
	}
}

void MnistIterator::nextTrainLabelBatch(int index, double* label) {

	int beginIndex = index * mTrainBatchSize;
	//endIndex,but not inlcude endIndex
	int endIndex = (index + 1) * mTrainBatchSize;

	if (endIndex > mTotalTrainImages) {
		beginIndex = mTotalTrainImages - mTrainBatchSize;
		endIndex = mTotalTrainImages;
	}

	for (int i = beginIndex; i < endIndex; i++) {
		uchar *data = mTrainLabels.ptr<uchar>(i);
		int temp = (int) data[0];
		for (int j = 0; j < mNumberOfClass; j++) {
			label[(i - beginIndex) * mNumberOfClass + j] = temp == j ? 1 : 0;
		}
	}

}

void MnistIterator::nextTestLabelBatch(int index, double* label) {

	int beginIndex = index * mTestBatchSize;
	//endIndex,but not inlcude endIndex
	int endIndex = (index + 1) * mTestBatchSize;

	if (endIndex > mTotalTestImages) {
		beginIndex = mTotalTestImages - mTestBatchSize;
		endIndex = mTotalTestImages;
	}

	for (int i = beginIndex; i < endIndex; i++) {
		uchar *data = mTestLabels.ptr<uchar>(i);
		int temp = (int) data[0];
		for (int j = 0; j < mNumberOfClass; j++) {
			label[(i - beginIndex) * mNumberOfClass + j] = temp == j ? 1 : 0;
		}
	}
}
