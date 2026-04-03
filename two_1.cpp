#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main()
{
    cout << "程序开始运行！" << endl;
    Mat img = imread("cow.jpg");

    if (img.empty())
    {
        cout << " 图片未找到！" << endl;
        return -1;
    }

    // 2. 输出图像基本信息
    cout << "\n===== 图像基本信息 =====" << endl;
    cout << "图像宽度：" << img.cols << endl;
    cout << "图像高度：" << img.rows << endl;
    cout << "通道数：" << img.channels() << endl;
    cout << "数据类型：" << img.type() << endl;
    cout << "========================\n" << endl;

    // 4. 转灰度图
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // 3+4. 合并显示原图+灰度图
    Mat gray_bgr;
    cvtColor(gray, gray_bgr, COLOR_GRAY2BGR);
    Mat combined;
    hconcat(img, gray_bgr, combined);

    namedWindow("原图 | 灰度图", WINDOW_NORMAL);
    imshow("原图 | 灰度图", combined);

    // 仅按q/ESC才关闭，截图不触发
    while (true) {
        int key = waitKey(100);
        if (key == 'q' || key == 27) {
            break;
        }
    }

    // 5. 保存灰度图
    imwrite("result_gray.jpg", gray);
    cout << "灰度图已保存：result_gray.jpg" << endl;

    // 6. NumPy操作
    Mat roi = gray(Rect(0, 0, 100, 100));
    imwrite("roi_100x100.jpg", roi);
    cout << "100x100 区域已保存：roi_100x100.jpg" << endl;

    uchar pixel = gray.at<uchar>(50, 50);
    cout << "灰度图 (50,50) 像素值：" << (int)pixel << endl;

    destroyAllWindows();
    cout << "\n全部任务完成！" << endl;

    return 0;
}