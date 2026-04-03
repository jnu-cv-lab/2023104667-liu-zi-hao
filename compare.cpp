#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

// 自定义均值滤波
Mat myMeanFilter(const Mat& img, int ksize = 3) {
    Mat res = img.clone();
    int pad = ksize / 2;
    for (int i = pad; i < img.rows - pad; i++) {
        for (int j = pad; j < img.cols - pad; j++) {
            int sum = 0;
            for (int x = -pad; x <= pad; x++)
                for (int y = -pad; y <= pad; y++)
                    sum += img.at<uchar>(i + x, j + y);
            res.at<uchar>(i, j) = sum / (ksize * ksize);
        }
    }
    return res;
}

// 修复后的直方图绘制函数
Mat drawHist(const Mat& src) {
    Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange[] = {range}; // 必须是二级指针
    calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, histRange, true, false); // 补全7个参数
    normalize(hist, hist, 0, 400, NORM_MINMAX);

    Mat histImg(400, 512, CV_8UC3, Scalar(255,255,255));
    for (int i = 1; i < 256; i++) {
        line(histImg, Point(i*2, 400 - cvRound(hist.at<float>(i-1))),
             Point(i*2, 400 - cvRound(hist.at<float>(i))), Scalar(0,0,255), 2);
    }
    return histImg;
}

void process(string imgName, string saveName) {
    Mat img = imread(imgName, 0);
    if (img.empty()) {
        cout << "无法读取图片: " << imgName << endl;
        return;
    }

    Mat globalEq, claheEq, meanF, gaussF, medianF, sharp, eqFilter, filterEq;

    equalizeHist(img, globalEq);
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8,8));
    clahe->apply(img, claheEq);
    meanF = myMeanFilter(img);
    GaussianBlur(img, gaussF, Size(3,3), 0);
    medianBlur(img, medianF, 3);

    Mat lap;
    Laplacian(img, lap, CV_16S);
    convertScaleAbs(lap, lap);
    addWeighted(img, 1, lap, -0.5, 0, sharp); // 修复锐化计算溢出

    GaussianBlur(globalEq, eqFilter, Size(3,3), 0);
    equalizeHist(gaussF, filterEq);

    imwrite(saveName + "_origin.png", img);
    imwrite(saveName + "_global.png", globalEq);
    imwrite(saveName + "_clahe.png", claheEq);
    imwrite(saveName + "_mean.png", meanF);
    imwrite(saveName + "_gauss.png", gaussF);
    imwrite(saveName + "_median.png", medianF);
    imwrite(saveName + "_sharp.png", sharp);
    imwrite(saveName + "_eq_filter.png", eqFilter);
    imwrite(saveName + "_filter_eq.png", filterEq);
    imwrite(saveName + "_hist.png", drawHist(img));

    cout << saveName << " 处理完成" << endl;
}

int main() {
    process("book.jpg", "book");
    process("sea.jpg", "sea");
    process("sun.jpg", "sun");
    return 0;
}