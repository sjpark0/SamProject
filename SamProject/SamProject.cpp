// SamProject.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include "sam.h"
#include "SJSegmentAnything.h"

#ifdef ENABLE_CUDA
#include "SJSegmentAnythingGPU.h"
#include "SJSegmentAnythingGPUHQ.h"
#endif

#include "opencv2/opencv.hpp"
#include <Windows.h>

using namespace std;
#ifdef ENABLE_CUDA
void PerfomanceTest()
{
    LARGE_INTEGER tickFreq;
    LARGE_INTEGER tickStart;
    LARGE_INTEGER tickEnd;
    QueryPerformanceFrequency(&tickFreq);

    double res;
    cv::Size inputSize;
    cv::Mat image;
    cv::Mat image1;

    std::vector<cv::Point> points;
    cv::Mat mask;

    image = cv::imread("..\\Data\\000.png");
    points.clear();
    //points.push_back(cv::Point(568, 305));
    points.push_back(cv::Point(2132, 1144));

    SJSegmentAnything* samcpu;
    SJSegmentAnythingGPU* samgpu;
    SJSegmentAnythingGPU* samquntized;
    QueryPerformanceCounter(&tickStart);
    samcpu = new SJSegmentAnything();
    samcpu->InitializeSamModel("..\\\models\\sam_onnx_preprocess_vit_h.onnx", "..\\models\\sam_onnx_example_vit_h.onnx", image.cols, image.rows);
    inputSize = samcpu->GetInputSize();
    cv::resize(image, image1, inputSize);
    mask = cv::Mat(image.rows, image.cols, CV_8UC1);
    samcpu->SamLoadImage(image1);
    samcpu->GetMask(points, {}, {}, mask, res);
    delete samcpu;
    QueryPerformanceCounter(&tickEnd);
    cout << "Warming up (CPU) : " << (double)(tickEnd.QuadPart - tickStart.QuadPart) / (double)(tickFreq.QuadPart) << "sec" << endl;

    QueryPerformanceCounter(&tickStart);
    samcpu = new SJSegmentAnything();
    samcpu->InitializeSamModel("..\\\models\\sam_onnx_preprocess_vit_h.onnx", "..\\models\\sam_onnx_example_vit_h.onnx", image.cols, image.rows);
    inputSize = samcpu->GetInputSize();
    cv::resize(image, image1, inputSize);
    mask = cv::Mat(image.rows, image.cols, CV_8UC1);
    samcpu->SamLoadImage(image1); 
    samcpu->GetMask(points, {}, {}, mask, res);
    QueryPerformanceCounter(&tickEnd);
    cout << "Second (CPU): " << (double)(tickEnd.QuadPart - tickStart.QuadPart) / (double)(tickFreq.QuadPart) << "sec" << endl;

    QueryPerformanceCounter(&tickStart);
    for (int i = 0; i < 100; i++) {
        samcpu->GetMask(points, {}, {}, mask, res);
    }
    QueryPerformanceCounter(&tickEnd);

    cout << "GetMask Only (CPU) " << (double)(tickEnd.QuadPart - tickStart.QuadPart) / (double)(tickFreq.QuadPart) << "sec" << endl;
    delete samcpu;


    //points[0] = cv::Point(2132, 1144);
    QueryPerformanceCounter(&tickStart);
    samgpu = new SJSegmentAnythingGPU();
    samgpu->InitializeSamModel("..\\\models\\sam_onnx_preprocess_vit_h.onnx", "..\\models\\sam_onnx_example_vit_h.onnx", image.cols, image.rows);
    inputSize = samgpu->GetInputSize();
    //cv::resize(image, image, inputSize);
    mask = cv::Mat(image.rows, image.cols, CV_8UC1);
    samgpu->SamLoadImage(image);
    samgpu->GetMask(points, {}, {}, mask, res);
    delete samgpu;
    QueryPerformanceCounter(&tickEnd);
    cout << "Warming up (GPU) : " << (double)(tickEnd.QuadPart - tickStart.QuadPart) / (double)(tickFreq.QuadPart) << "sec" << endl;

    QueryPerformanceCounter(&tickStart);
    samgpu = new SJSegmentAnythingGPU();
    samgpu->InitializeSamModel("..\\\models\\sam_onnx_preprocess_vit_h.onnx", "..\\models\\sam_onnx_example_vit_h.onnx", image.cols, image.rows);
    inputSize = samgpu->GetInputSize();
    //cv::resize(image, image, inputSize);
    mask = cv::Mat(image.rows, image.cols, CV_8UC1);
    samgpu->SamLoadImage(image);
    samgpu->GetMask(points, {}, {}, mask, res);
    QueryPerformanceCounter(&tickEnd);
    cout << "Second (GPU): " << (double)(tickEnd.QuadPart - tickStart.QuadPart) / (double)(tickFreq.QuadPart) << "sec" << endl;

    QueryPerformanceCounter(&tickStart);
    for (int i = 0; i < 100; i++) {
        samgpu->GetMask(points, {}, {}, mask, res);
    }
    QueryPerformanceCounter(&tickEnd);
    cout << "GetMask Only (GPU) " << (double)(tickEnd.QuadPart - tickStart.QuadPart) / (double)(tickFreq.QuadPart) << "sec" << endl;
    delete samgpu;    
}

void SamGPU()
{
    double res;
    cv::Size inputSize;
    cv::Mat image;
    std::vector<cv::Point> points;
    cv::Mat mask;

    image = cv::imread("..\\Data\\000.png");
    points.clear();
    points.push_back(cv::Point(568, 305));
    //points.push_back(cv::Point(2132, 1144));
    //points.push_back(cv::Point(338, 287));
    SJSegmentAnythingGPU* samgpu;
    samgpu = new SJSegmentAnythingGPU();
    samgpu->InitializeSamModel("..\\\models\\sam_onnx_preprocess_vit_h.onnx", "..\\models\\sam_onnx_example_vit_h.onnx", image.cols, image.rows);
    inputSize = samgpu->GetInputSize();
    mask = cv::Mat(image.rows, image.cols, CV_8UC1);

    samgpu->SamLoadImage(image);
    samgpu->GetMask(points, {}, {}, mask, res);
    cv::imwrite("output_gpu.png", mask);

    delete samgpu;
}
void SamGPUHQ()
{
    double res;
    cv::Size inputSize;
    cv::Mat image;
    std::vector<cv::Point> points;
    cv::Mat mask;

    image = cv::imread("..\\Data\\000.png");
    points.clear();
    points.push_back(cv::Point(568, 305));

    SJSegmentAnythingGPUHQ* samgpuhq;
    samgpuhq = new SJSegmentAnythingGPUHQ();
    samgpuhq->InitializeSamModel("..\\\modelsHQ\\sam_onnx_preprocess.onnx", "..\\modelsHQ\\sam_onnx_example_vit_h.onnx");
    inputSize = samgpuhq->GetInputSize();
    cv::resize(image, image, inputSize);
    mask = cv::Mat(inputSize.height, inputSize.width, CV_8UC1);
    samgpuhq->SamLoadImage(image);
    samgpuhq->GetMask(points, {}, {}, mask, res);
    cv::imwrite("output_gpu_hq.png", mask);

    delete samgpuhq;
}

#endif
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
    SamOriginal();
    SamCPU();
#ifdef ENABLE_CUDA    
    SamGPU(); 
#endif

}