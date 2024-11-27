// SamProject.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include "sam.h"
#include "SJSegmentAnything.h"
#include "SJSegmentAnythingCPU.h"
#include "SJSegmentAnythingGPU.h"
#include "SJSegmentAnythingGPUHQ.h"

#include "SJSegmentAnything2.h"
#include "SJSegmentAnythingCPU2.h"
#include "SJSegmentAnythingGPU2.h"

#include "SJSegmentAnythingTRT.h"
#include "opencv2/opencv.hpp"
#include <Windows.h>
using namespace std;
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
    SJSegmentAnything* samgpu;
    SJSegmentAnything* samquntized;
    QueryPerformanceCounter(&tickStart);
    samcpu = new SJSegmentAnythingCPU();
    samcpu->InitializeSamModel("..\\\models\\sam_onnx_preprocess_vit_h.onnx", "..\\models\\sam_onnx_example_vit_h.onnx", 1, image.cols, image.rows);
    inputSize = samcpu->GetInputSize();
    cv::resize(image, image1, inputSize);
    mask = cv::Mat(image.rows, image.cols, CV_8UC1);
    samcpu->SamLoadImage(image1, 0);
    samcpu->GetMask(points, {}, {}, mask, res);
    delete samcpu;
    QueryPerformanceCounter(&tickEnd);
    cout << "Warming up (CPU) : " << (double)(tickEnd.QuadPart - tickStart.QuadPart) / (double)(tickFreq.QuadPart) << "sec" << endl;

    QueryPerformanceCounter(&tickStart);
    samcpu = new SJSegmentAnythingCPU();
    samcpu->InitializeSamModel("..\\\models\\sam_onnx_preprocess_vit_h.onnx", "..\\models\\sam_onnx_example_vit_h.onnx", 1, image.cols, image.rows);
    inputSize = samcpu->GetInputSize();
    cv::resize(image, image1, inputSize);
    mask = cv::Mat(image.rows, image.cols, CV_8UC1);
    samcpu->SamLoadImage(image1, 0); 
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
    samgpu->InitializeSamModel("..\\\models\\sam_onnx_preprocess_vit_h.onnx", "..\\models\\sam_onnx_example_vit_h.onnx", 1, image.cols, image.rows);
    inputSize = samgpu->GetInputSize();
    //cv::resize(image, image, inputSize);
    mask = cv::Mat(image.rows, image.cols, CV_8UC1);
    samgpu->SamLoadImage(image, 0);
    samgpu->GetMask(points, {}, {}, mask, res);
    delete samgpu;
    QueryPerformanceCounter(&tickEnd);
    cout << "Warming up (GPU) : " << (double)(tickEnd.QuadPart - tickStart.QuadPart) / (double)(tickFreq.QuadPart) << "sec" << endl;

    QueryPerformanceCounter(&tickStart);
    samgpu = new SJSegmentAnythingGPU();
    samgpu->InitializeSamModel("..\\\models\\sam_onnx_preprocess_vit_h.onnx", "..\\models\\sam_onnx_example_vit_h.onnx", 1, image.cols, image.rows);
    inputSize = samgpu->GetInputSize();
    //cv::resize(image, image, inputSize);
    mask = cv::Mat(image.rows, image.cols, CV_8UC1);
    samgpu->SamLoadImage(image, 0);
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

void PerfomanceTest2()
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

    image = cv::imread("..\\Data\\100.png");
    points.clear();
    points.push_back(cv::Point(533, 286));
    //points.push_back(cv::Point(568, 305));
    //points.push_back(cv::Point(2132, 1144));

    SJSegmentAnything2* samcpu;
    SJSegmentAnything2* samgpu;
    QueryPerformanceCounter(&tickStart);
    samcpu = new SJSegmentAnythingCPU2();
    samcpu->InitializeSamModel("..\\\sam2_models\\sam2.1_hiera_large_encoder.onnx", "..\\sam2_models\\decoder.onnx", 1, image.cols, image.rows);
    mask = cv::Mat(image.rows, image.cols, CV_8UC1);
    samcpu->SamLoadImage(image, 0);
    samcpu->GetMask(points, {}, {}, mask, res);
    delete samcpu;
    QueryPerformanceCounter(&tickEnd);
    cout << "Warming up (CPU) : " << (double)(tickEnd.QuadPart - tickStart.QuadPart) / (double)(tickFreq.QuadPart) << "sec" << endl;

    QueryPerformanceCounter(&tickStart);
    samcpu = new SJSegmentAnythingCPU2();
    samcpu->InitializeSamModel("..\\\sam2_models\\sam2.1_hiera_large_encoder.onnx", "..\\sam2_models\\decoder.onnx", 1, image.cols, image.rows);
    mask = cv::Mat(image.rows, image.cols, CV_8UC1);
    samcpu->SamLoadImage(image, 0);
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
    samgpu = new SJSegmentAnythingGPU2();
    samgpu->InitializeSamModel("..\\\sam2_models\\sam2.1_hiera_large_encoder.onnx", "..\\sam2_models\\decoder.onnx", 1, image.cols, image.rows);
    mask = cv::Mat(image.rows, image.cols, CV_8UC1);
    samgpu->SamLoadImage(image, 0);
    samgpu->GetMask(points, {}, {}, mask, res);
    delete samgpu;
    QueryPerformanceCounter(&tickEnd);
    cout << "Warming up (GPU) : " << (double)(tickEnd.QuadPart - tickStart.QuadPart) / (double)(tickFreq.QuadPart) << "sec" << endl;

    QueryPerformanceCounter(&tickStart);
    samgpu = new SJSegmentAnythingGPU2();
    samgpu->InitializeSamModel("..\\\sam2_models\\sam2.1_hiera_large_encoder.onnx", "..\\sam2_models\\decoder.onnx", 1, image.cols, image.rows);
    mask = cv::Mat(image.rows, image.cols, CV_8UC1);
    samgpu->SamLoadImage(image, 0);
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
void SamOriginal()
{
    Sam::Parameter param("..\\\models\\sam_onnx_preprocess_vit_h.onnx", "..\\models\\sam_onnx_example_vit_h.onnx", std::thread::hardware_concurrency());
    param.providers[0].deviceType = 0; // cpu for preprocess
    param.providers[1].deviceType = 0; // CUDA for sam
    Sam sam(param);
    auto inputSize = sam.getInputSize();
    
    cv::Mat image = cv::imread("..\\Data\\007.png");
    cv::imwrite("test.png", image);
    cv::resize(image, image, inputSize);
    
    sam.loadImage(image);
    
    //cv::Mat mask = sam.autoSegment({ 10, 10 });
    cv::Mat mask = sam.getMask({ 361, 306 }); // 533 * 1024 / 960, 286 * 1024 / 960

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

    image = cv::imread("..\\Data\\007.png");
    points.clear();
    //points.push_back(cv::Point(568, 305));
    //points.push_back(cv::Point(2132, 1144));
    points.push_back(cv::Point(1354, 1150));
    SJSegmentAnything* samcpu;
    samcpu = new SJSegmentAnythingCPU();
    samcpu->InitializeSamModel("..\\\models\\sam_onnx_preprocess_vit_h.onnx", "..\\models\\sam_onnx_example_vit_h.onnx", 1, image.cols, image.rows);
    inputSize = samcpu->GetInputSize();
    samcpu->SamLoadImage(image, 0);
    printf("here!!\n");
    mask = cv::Mat(image.rows, image.cols, CV_8UC1);
    samcpu->GetMask(points, {}, {}, mask, res);
    printf("here!!\n");
    cv::imwrite("output_cpu.png", mask);
    printf("here!!\n");
    delete samcpu;

}
void SamGPU()
{
    double res;
    cv::Size inputSize;
    cv::Mat image;
    std::vector<cv::Point> points;
    cv::Mat mask;

    image = cv::imread("..\\Data\\007.png");
    points.clear();
    //points.push_back(cv::Point(2132, 1144));
    points.push_back(cv::Point(1354, 1150));
    SJSegmentAnything* samgpu;
    samgpu = new SJSegmentAnythingGPU();
    samgpu->InitializeSamModel("..\\\models\\sam_onnx_preprocess_vit_h.onnx", "..\\models\\sam_onnx_example_vit_h.onnx", 1, image.cols, image.rows);
    inputSize = samgpu->GetInputSize();
    mask = cv::Mat(image.rows, image.cols, CV_8UC1);

    samgpu->SamLoadImage(image, 0);
    samgpu->GetMask(points, {}, {}, mask, res);
    cv::imwrite("output_gpu.png", mask);

    delete samgpu;
}
void SamCPU2()
{
    double res;
    cv::Size inputSize;
    cv::Mat image;
    cv::Mat image1;

    std::vector<cv::Point> points;
    cv::Mat mask;

    image = cv::imread("..\\Data\\100.png");
    points.clear();
    points.push_back(cv::Point(533, 286));
    //points.push_back(cv::Point(2132, 1144));
    //points.push_back(cv::Point(1354, 1150));
    SJSegmentAnything2* samcpu;
    samcpu = new SJSegmentAnythingCPU2();
    samcpu->InitializeSamModel("..\\\sam2_models\\sam2.1_hiera_large_encoder.onnx", "..\\sam2_models\\decoder.onnx", 1, image.cols, image.rows);
    inputSize = samcpu->GetInputSize();
    samcpu->SamLoadImage(image, 0);
    mask = cv::Mat(image.rows, image.cols, CV_8UC1);
    samcpu->GetMask(points, {}, {}, mask, res);
    cv::imwrite("output_cpu2.png", mask);
    delete samcpu;

}
void SamGPU2()
{
    double res;
    cv::Size inputSize;
    cv::Mat image;
    std::vector<cv::Point> points;
    cv::Mat mask;

    image = cv::imread("..\\Data\\100.png");
    points.clear();
    points.push_back(cv::Point(533, 286));
    //points.push_back(cv::Point(2132, 1144));
    //points.push_back(cv::Point(1354, 1150));
    SJSegmentAnything2* samgpu;
    samgpu = new SJSegmentAnythingGPU2();
    samgpu->InitializeSamModel("..\\\sam2_models\\sam2.1_hiera_large_encoder.onnx", "..\\sam2_models\\decoder.onnx", 1, image.cols, image.rows);
    inputSize = samgpu->GetInputSize();
    mask = cv::Mat(image.rows, image.cols, CV_8UC1);

    samgpu->SamLoadImage(image, 0);
    samgpu->GetMask(points, {}, {}, mask, res);
    cv::imwrite("output_gpu2.png", mask);

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
void SamTRT()
{
    double res;
    cv::Size inputSize;
    cv::Mat image;
    std::vector<cv::Point> points;
    cv::Mat mask;

    image = cv::imread("..\\Data\\000.png");
    points.clear();
    points.push_back(cv::Point(568, 305));

    SJSegmentAnythingTRT *samtrt;
    samtrt = new SJSegmentAnythingTRT();
    //printf("%d\n", samtrt->InitializeSamModel("..\\\models\\sam_onnx_preprocess.onnx", "..\\models\\sam_onnx_example.onnx"));
    printf("%d\n", samtrt->InitializeSamModel("..\\\models\\sam_onnx_preprocess_vit_h_quantized.onnx", "..\\models\\sam_onnx_example_vit_h_quantized.onnx"));
    /*inputSize = samtrt->GetInputSize();
    cv::resize(image, image, inputSize);
    mask = cv::Mat(inputSize.height, inputSize.width, CV_8UC1);
    samtrt->SamLoadImage(image);
    samtrt->GetMask(points, {}, {}, mask, res);
    cv::imwrite("output_trt.png", mask);*/

    delete samtrt;
}
int main()
{   
    //SamOriginal();
    //SamCPU();
    //getchar();
    //SamCPU();
    
    //SamCPU();
    //SamGPU();
    
    //SamCPU2();
    //SamGPU2();
    PerfomanceTest2();
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