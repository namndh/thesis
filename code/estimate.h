//
// Created by Sam Hoang on 9/7/17.
//

#ifndef GR_ESTIMATE_H
#define GR_ESTIMATE_H

#include <time.h>
#include <stdlib.h>

// opencv
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv/cv.h"
#include "opencv/ml.h"
#include "ParallelSiftQuantization.h"


// cvxopt
#include "cvxopt.h"
#include "vl_dsift.h"


extern Mat accumulateFeature(Mat feature, Rect roiRect, Mat weight, int nFeatures);

extern Mat learn_to_count(int nFeatures, vector<Mat> features, vector<Mat> weights, Mat weightsMask, vector<Mat> densities,
                   float C,
                   int maxIter,
                   bool verbose);

extern bool isNonNegative(vector<Mat> *weights);

extern void addFeature(Mat *A, Mat *b, Mat feature, int rectIndex, int imageIndex, float sumDensity, int xLength);

extern PyObject *toPyMat(Mat m, int rows);

extern Mat toMat(PyObject *x, int cols);

extern Mat estimateDensity(Mat x, int nFeatures, Mat feature, Mat weight);

extern Mat estimateDensity(PyObject *x, int nFeatures, Mat feature, Mat weight);

#endif //GR_ESTIMATE_H
