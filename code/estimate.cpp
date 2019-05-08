//
// Created by Sam Hoang on 9/7/17.
//



#include "estimate.h"
#include "maxsubarray2D.h"

using namespace cv;


Mat estimateDensity(Mat x, int nFeatures, Mat feature, Mat weight) {
    Mat density = Mat(feature.size(), CV_64F);

    int rows = feature.rows;
    int cols = feature.cols;
    double minVal, maxVal;

    // make sure that the cluster index in feature matrix doesn't exceed nFeature
    minMaxLoc(feature, &minVal, &maxVal);
    if (maxVal > nFeatures) {
        cout << "Assertion failed: the cluster index in feature matrix exceeds nFeature" << endl;
        exit(7);
    }

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            int clusterIndex = (int) feature.at<float>(i, j);
            density.at<double>(i, j) = x.at<float>(0, clusterIndex)*weight.at<float>(i, j);
        }

    return density;
}

/**
 * calculate feature vector of a box
 * @param feature
 * @param roiRect
 * @param weight
 * @param nFeatures
 * @return
 */
Mat accumulateFeature(Mat feature, Rect roiRect, Mat weight, int nFeatures) {
    // init the featureVector to 0
    // indexes from 0 to 255
    Mat featureVector = Mat::zeros(1, nFeatures, CV_32F);
    // loop through the roiRect and add the weight of
    // pixels of the region having value != 0 to
    // featureVector
    for(int y = roiRect.y; y < roiRect.y + roiRect.height; y++)
        // y indexes row
        for(int x = roiRect.x; x < roiRect.x + roiRect.width; x++)
            // x indexes cols
        {
            // feature.at<float>(y, x) is the index of the cluster that the pixel belongs to.
            // And the index of weight starts from 0
            featureVector.at<float>(0, feature.at<float>(y, x)) += weight.at<float>(y-roiRect.y, x-roiRect.x);
        }

    return featureVector;
}

/**
 * check if all elements of all matrix of weights is non negative
 * @param weights
 * @return
 */
bool isNonNegative(vector<Mat> *weights) {
    int n = weights->size();
    double minVal, maxVal;

    for (int i = 0; i < n; i++) {
        minMaxLoc((*weights)[i], &minVal, &maxVal);
        // if the current matrix has the minimum value < 0, return false
        if (minVal < 0) return false;
    }

    // return true
    return true;
}

Mat learn_to_count(int nFeatures, vector<Mat> features, vector<Mat> weights, Mat weightsMask, vector<Mat> densities,
                   float C,
                   int maxIter,
                   bool verbose) {
    // get image size
    const int rows = features[0].rows;
    const int cols = features[0].cols;
    const unsigned int nImages = features.size();
    const int initial = 20; // initial box
    int preAlloc = 1 + initial + maxIter * nImages * 2;
    int nRects = 0;
    float value = 0;

    Mat rects = Mat::zeros(preAlloc, 5, CV_32F);
    Mat G = Mat::zeros(2 * preAlloc + nFeatures + nImages, nFeatures + nImages, CV_32F); // todo ??
    Mat h = Mat::zeros(2 * preAlloc + nFeatures + nImages, 1, CV_32F);
    Mat P;
    Mat q;
    Mat result;

    cout << "Add lower bound constraint" << endl;
    if (isNonNegative(&weights)) {
        cout << "Non-negative feature encoding assumed. w is constrained to be non-negative." << endl;
        Mat lbe = Mat::eye(nFeatures + nImages, nFeatures + nImages, CV_32F)*-1;
        lbe.copyTo(G(Rect(0, 0, lbe.cols, lbe.rows)));
    }

    cout << "draw rectangle randomly" << endl;

    // draw some rectangles randomly
    for (int r = 0; r < initial; r++) {

        int x1 = 0, x2 = 0, y1 = 0, y2 = 0;
        int xmax, xmin, ymax, ymin;

        int imageIndex = 0;

        while (abs(x2 - x1) < 1 || abs(y2 - y1) < 1 ) {
            x1 = rand()%cols;
            x2 = rand()%cols;
            y1 = rand()%rows;
            y2 = rand()%rows;
            imageIndex = rand()%nImages;
        }

        if (x2 > x1) {
            xmax = x2;
            xmin = x1;
        } else {
            xmax = x1;
            xmin = x2;
        }

        if (y2 > y1) {
            ymax = y2;
            ymin = y1;
        } else {
            ymax = y1;
            ymin = y2;
        }

        rects.at<float>(nRects, 0) = xmin; rects.at<float>(nRects, 1) = xmax;
        rects.at<float>(nRects, 2) = ymin; rects.at<float>(nRects, 3) = ymax;
        rects.at<float>(nRects, 4) = imageIndex;
        //cout << rects.row(nRects) << endl;

        Rect roiRect = Rect(xmin, ymin, xmax-xmin, ymax-ymin);
        // calculate density over the random box
        Scalar vl = sum(densities[imageIndex](roiRect));
        value = vl[0];
        // element wise multiplication of weightMask with weights[i] over roi Rect
        Mat weight = weightsMask(roiRect).mul(weights[imageIndex](roiRect));
        Mat feature = accumulateFeature(features[imageIndex], roiRect, weight, nFeatures);
        addFeature(&G, &h, feature, nRects, imageIndex, value, nFeatures+nImages);
        nRects++;
    } // end for

    int eyeSize = nFeatures+nImages;
    if (C >= 0) {
        cout << "Tikhonov regularization is used." << endl;
        P = 2*Mat::eye(eyeSize, eyeSize, CV_32F);
        q = Mat::zeros(eyeSize, 1, CV_32F);
        // set elements of images to 0
        for (int i = nFeatures; i < eyeSize; i++) {
            P.at<float>(i, i) = 0;
            q.at<float>(i, 1) = C;
        }
    } else {
        cout << "L1 regularization is used." << endl;
        q = Mat::ones(eyeSize, 1, CV_32F);
        for (int i = nFeatures; i < eyeSize; i++) {
            q.at<float>(i, 1) = C*(-1);
        }
    }

    // solve the quadratic programming using cvxopt
    // init python environment
    Py_Initialize();
    if (import_cvxopt() < 0) {
        fprintf(stderr, "error importing cvxopt");
        exit(1);
    }

    /* import cvxopt.solvers */
    PyObject *solvers = PyImport_ImportModule("cvxopt.solvers");
    if (!solvers) {
        fprintf(stderr, "error importing cvxopt.solvers");
        exit(2);
    }

    /* get reference to solvers.solvelp */
    PyObject *qp = PyObject_GetAttrString(solvers, "qp");
    if (!qp) {
        fprintf(stderr, "error referencing cvxopt.solvers.lp");
        Py_DECREF(solvers);
        exit(3);
    }

    PyObject *pArgs;
    PyObject *sol;
    PyObject *x;

    for (int k = 0; k < maxIter; k++) {
        cout << "iteration: " << k << endl;

        PyObject *pyP = toPyMat(P, 0);
        PyObject *pyq = toPyMat(q, 0);
        PyObject *pyh = toPyMat(h, nImages + nFeatures + nRects*2);
        PyObject *pyG = toPyMat(G, nImages + nFeatures + nRects*2);

        // pack arguments
        pArgs = PyTuple_New(4);

        if (!pyP || !pyq || !pyG || !pyh || !pArgs) {
            cout << "Error creating matrices" << endl;
            Py_DECREF(solvers); Py_DECREF(qp);
            Py_XDECREF(pyP); Py_XDECREF(pyq); Py_XDECREF(pyG); Py_XDECREF(pyh); Py_XDECREF(pArgs);
            exit(4);
        }

        /* pack matrices into an argument tuple - references are stolen*/
        PyTuple_SetItem(pArgs, 0, pyP);
        PyTuple_SetItem(pArgs, 1, pyq);
        PyTuple_SetItem(pArgs, 2, pyG);
        PyTuple_SetItem(pArgs, 3, pyh);

        // solve
        sol = PyObject_CallObject(qp, pArgs);
        if (!sol) {
            PyErr_Print();
            Py_DECREF(solvers); Py_DECREF(qp); Py_DECREF(pArgs);
            exit(5);
        }

        x = PyDict_GetItemString(sol, "x");

        if (verbose) {
            cout << "nRects = " << nRects << endl;
            printf("x = ");
            for (int i = 0; i < nFeatures; i++){
                printf("%5.4e   ", MAT_BUFD(x)[i]);
            }


            printf("\nslack = ");
            for (int i = nFeatures; i < nFeatures + nImages; i++){
                printf("%5.4e   ", MAT_BUFD(x)[i]);
            }

        }

        int maxX1, maxY1, minX1, minY1;
        int maxX2, maxY2, minX2, minY2;
        bool change = false;
        // since we had solutions, generating constraints
        for (int i = 0; i < nImages; i++) {
            Mat density = estimateDensity(x, nFeatures, features[i], weights[i]);
            // calculate the differences in ground truth density and estimate density
            Mat d2;
            Mat d1;
            density.convertTo(d1, CV_64F);
            densities[i].convertTo(d2, CV_64F);
            Mat diff = d1 - d2;
//            if (i == 0) {
//                writeCSV("loop_" + to_string(k) + ".csv", diff);
//            }


            double sumVal1 = maxSubarray2D(diff, &minX1, &minY1, &maxX1, &maxY1);
            double sumVal2 = maxSubarray2D(-1*diff, &minX2, &minY2, &maxX2, &maxY2);

            double maxSum = sumVal1 > sumVal2 ? sumVal1 : sumVal2;
            int pos = nFeatures+i;


            if (verbose) {
                cout << endl;
                cout << "maxSum = " << maxSum << " ";
                cout << "slack = " << MAT_BUFD(x)[pos] << " ";
                cout << "minX1 = " << minX1 << " minY1 = " << minY1 << " maxX1 = " << maxX1 << " maxY1 = " << maxY1 << " sum1 = " << sumVal1 << endl;
                cout << "minX2 = " << minX2 << " minY2 = " << minY2 << " maxX2 = " << maxX2 << " maxY2 = " << maxY2 << " sum2 = " << sumVal2 <<endl;
                cout << endl;
            }
            if (maxSum < MAT_BUFD(x)[pos]*1.01) {
                if (verbose) {
                    cout << "image " << i << " is sastisfy" << endl;
                }
                // the epsilon = 0.01 can be tweaked
                continue;
            }


            change = true;

            Rect roi;
            if (sumVal1 > sumVal2) {
                //roi = Rect(minY1, minX1, maxY1-minY1, maxX1-minX1);
                roi = Rect(minX1, minY1,   maxX1-minX1+1, maxY1-minY1+1); // oanhnt
            } else {
                //roi = Rect(minY2, minX2, maxY2-minY2, maxX2-minX2);
                roi = Rect(minX2, minY2,   maxX2-minX2+1, maxY2-minY2+1); // oanhnt
            }

            float sumDensity = (float)sum(densities[i](roi))[0];
            Mat weight = weightsMask(roi).mul(weights[i](roi));
            Mat feature = accumulateFeature(features[i], roi, weight, nFeatures);
            addFeature(&G, &h, feature, nRects, i, sumDensity, nFeatures+nImages);
            nRects++;

            if (nRects == preAlloc) {
                cout << "Warning: Number of pre-allocated rectangles reached! " <<
                     "Terminating the constraint generation" << endl;
                break;
            }
        }

        if (!change) {
            cout << "Constraint generation successfully converged" << endl;
            break;
        }
    }

    // convert the result to opencv mat
    result = toMat(x, nFeatures + nImages);
    // TODO: end for

    Py_DECREF(solvers);
    Py_DECREF(qp);
    Py_DECREF(pArgs);
    Py_DECREF(sol);

    Py_Finalize();

    return result;
}

/**
 * Calculate matrix A and b which is in the constrains:
 * Ax <= b
 * @param A
 * @param b
 * @param feature
 * @param rectIndex
 * @param nImages
 * @param xLength - the length of vector x need to be resolved
 */
void addFeature(Mat *A, Mat *b, Mat feature, int rectIndex, int imageIndex, float sumDensity, int xLength) {
    // first inequality formula at the index 2*rectIndex
    // copy calculated feature to A[0 -> nfeature]
    feature.copyTo((*A)(Rect(0, xLength + 2*rectIndex, feature.cols, feature.rows)));
    (*A).at<float>(xLength + 2*rectIndex, feature.cols + imageIndex) = -1;
    (*b).at<float>(xLength + 2*rectIndex, 0) = sumDensity;

    // second inequality formula at the index 2*rectIndex+1
    feature = feature*(-1);
    feature.copyTo((*A)(Rect(0, xLength + 2*rectIndex+1, feature.cols, feature.rows)));
    (*A).at<float>(xLength + 2*rectIndex+1, feature.cols + imageIndex) = -1;
    (*b).at<float>(xLength + 2*rectIndex+1, 0) = -1*sumDensity;
}

/**
 * Create a cvxopt matrix
 * m - opencv mat object
 * if the matrix is:
 * | 1   2 |
 * | 3   4 |
 * then values would be [1, 3, 2, 4]
 */
PyObject *toPyMat(Mat m, int rows) {
    if (m.empty()) {
        return NULL;
    }

    if (rows == 0) {
        // rows = 0 means that all rows of the mat is selected
        rows = m.rows;
    }

    int cols = m.cols;

    // create an empty matrix
    PyObject *M = (PyObject *)Matrix_New(rows, cols, DOUBLE);

    // assign values to the matrix
    int k = 0;
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            MAT_BUFD(M)[k] = m.at<float>(j, i);
            //printf("%7.1lf", MAT_BUFD(M)[k]);
            k++;
        }
        //cout << endl;
    }

    return M;
}

/**
 * Convert pymat to opencv mat object.
 * The pymats are always in the size of 1xn, so the opencv mat also has the same size
 * @param x - pymat pointer
 * @param cols - the number of cols of the pymat
 * @return
 */
Mat toMat(PyObject *x, int cols) {
    if (cols <= 0) {
        fprintf(stderr, "Error: invalid cols");
        exit(1);
    }
    Mat m = Mat::zeros(1, cols, CV_32F);
    for (int i = 0; i < cols; i++) {
        m.at<float>(0, i) = MAT_BUFD(x)[i];
    }

    return m;
}


/**
 * estimate the picture density based on the learned weight x
 * @param x - learned weight
 * @param nFeatures - the number of features
 * @param feature - feature matrix
 * @param weight - weight maxtrix
 * @return
 */
Mat estimateDensity(PyObject *x, int nFeatures, Mat feature, Mat weight) {
    Mat density = Mat(feature.size(), CV_64F);

    int rows = feature.rows;
    int cols = feature.cols;
    double minVal, maxVal;

    // make sure that the cluster index in feature matrix doesn't exceed nFeature
    minMaxLoc(feature, &minVal, &maxVal);
    if (maxVal > nFeatures) {
        cout << "Assertion failed: the cluster index in feature matrix exceeds nFeature" << endl;
        exit(7);
    }


    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            int clusterIndex = (int) feature.at<float>(i, j);
            density.at<double>(i, j) = MAT_BUFD(x)[clusterIndex]*weight.at<float>(i, j);
        }

    return density;
}
