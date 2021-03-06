#include "pch.h"

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "MarkovRandomField.h"

using namespace std;
using namespace cv;

const int MAX_DISPARITY = 24;
const int ITERATION = 20;
const int LAMBDA = 10;
const int SMOOTHNESS_PARAM = 2;

double countTime() {
	return static_cast<double>(clock());
}

int main() {
	const double beginTime = countTime();

	MarkovRandomField mrf;
	const string leftImgPath = "test-data/teddy/im2.png";
	const string rightImgPath = "test-data/teddy/im6.png";
	MarkovRandomFieldParam param;

	param.iteration = ITERATION;
	param.lambda = LAMBDA;
	param.maxDisparity = MAX_DISPARITY;
	param.smoothnessParam = SMOOTHNESS_PARAM;

	initializeMarkovRandomField(mrf, leftImgPath, rightImgPath, param);

	for (int i = 0; i < mrf.param.iteration; i++) {
		beliefPropagation(mrf, Left);
		beliefPropagation(mrf, Right);
		beliefPropagation(mrf, Up);
		beliefPropagation(mrf, Down);

		const energy_t energy = calculateMaxPosteriorProbability(mrf);

		cout << "Iteration: " << i << ";  Energy: " << energy << "." << endl;
	}

	Mat output = Mat::zeros(mrf.height, mrf.width, CV_8U);

	for (int i = mrf.param.maxDisparity; i < mrf.height - mrf.param.maxDisparity; i++) {
		for (int j = mrf.param.maxDisparity; j < mrf.width - mrf.param.maxDisparity; j++) {
			output.at<uchar>(i, j) = mrf.grid[i * mrf.width + j].bestAssignmentIndex * (256 / mrf.param.maxDisparity);
		}
	}

	const double endTime = countTime();

	cout << (endTime - beginTime) / CLOCKS_PER_SEC << endl;

	imshow("Output", output);
	waitKey();
	return 0;
}
