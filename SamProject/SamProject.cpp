// SamProject.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include "sam.h"
#include "SJSegmentAnything.h"

#include "opencv2/opencv.hpp"
#include <Windows.h>
using namespace std;
void SamOriginal()
{
    Sam::Parameter param("..\\\models\\sam_onnx_preprocess_vit_h.onnx", "..\\models\\sam_onnx_example_vit_h.onnx", std::thread::hardware_concurrency());
    param.providers[0].deviceType = 0; // cpu for preprocess
    param.providers[1].deviceType = 0; // CUDA for sam
    Sam sam(param);
    auto inputSize = sam.getInputSize();
    
    cv::Mat image = cv::imread("..\\Data\\000.png");
    cv::resize(image, image, inputSize);
    
    sam.loadImage(image);
    
    //cv::Mat mask = sam.autoSegment({ 10, 10 });
    //cv::Mat mask = sam.getMask({ 361, 306 }); // 533 * 1024 / 960, 286 * 1024 / 960
    cv::Mat mask = sam.getMask({ 568, 305 });
    cv::imwrite("output_original.png", mask);
}
void SamCPU()
{
    double res;
    cv::Size inputSize;
    cv::Mat image;
    cv::Mat image1;

    std::vector<cv::Point> points;
    cv::Mat mask;

    image = cv::imread("..\\Data\\000.png");
    points.clear();
    points.push_back(cv::Point(568, 305));
    //points.push_back(cv::Point(2132, 1144));
    //points.push_back(cv::Point(338, 287));
    SJSegmentAnything* samcpu;
    samcpu = new SJSegmentAnything();
    samcpu->InitializeSamModel("..\\\models\\sam_onnx_preprocess_vit_h.onnx", "..\\models\\sam_onnx_example_vit_h.onnx", image.cols, image.rows);
    inputSize = samcpu->GetInputSize();
    samcpu->SamLoadImage(image);
    printf("here!!\n");
    mask = cv::Mat(image.rows, image.cols, CV_8UC1);
    samcpu->GetMask(points, {}, {}, mask, res);
    printf("here!!\n");
    cv::imwrite("output_cpu.png", mask);
    printf("here!!\n");
    delete samcpu;

}
int main()
{   
    //SamOriginal();
    //SamCPU();
    //getchar();
    SamCPU();
    
    //SamTRT();
    //SamGPUHQ();
    //PerfomanceTest();

    /*Sam::Parameter param("..\\\models\\sam_onnx_preprocess.onnx", "..\\models\\sam_onnx_example.onnx", std::thread::hardware_concurrency());
    param.providers[0].deviceType = 1; // cpu for preprocess
    param.providers[1].deviceType = 1; // CUDA for sam
    Sam sam(param);
    printf("here!!\n");
    auto inputSize = sam.getInputSize();
    printf("here!!\n");

    cv::Mat image = cv::imread("..\\Data\\000.png");
    printf("here!!\n");

    cv::resize(image, image, inputSize);
    printf("here!!\n");

    sam.loadImage(image);
    printf("Finish!!\n");

    //cv::Mat mask = sam.autoSegment({ 10, 10 });
    cv::Mat mask = sam.getMask({ 568, 305 }); // 533 * 1024 / 960, 286 * 1024 / 960

    cv::imwrite("output.png", mask);*/
    //SJSegmentAnything sam;
    
}