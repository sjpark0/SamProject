#include "SJSegmentAnythingGPU.h"
#include <codecvt>
#include <fstream>
#include <iostream>
#include <locale>
#include <opencv2/opencv.hpp>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace cv;
using namespace std;

SJSegmentAnythingGPU::SJSegmentAnythingGPU()
{
	m_pSessionPre = NULL;
	m_pSessionSam = NULL;
	m_pImageCUDATemp = NULL;
	m_pImageCUDA = NULL;
	m_pPreprocessing = NULL;
	m_pSegmentation = NULL;
	m_pSegmentationFloat = NULL;
	m_maskInputValue = NULL;
	m_pMemoryInfo = NULL;
	m_pInputTensorPre = NULL;
	m_pOutputTensorSam = NULL;
	m_hasMaskValueCUDA = NULL;
	m_orig_im_size_values = NULL;
	m_orig_im_size_valuesCUDA = NULL;
}
SJSegmentAnythingGPU::~SJSegmentAnythingGPU()
{
	if (m_pSessionPre) {
		delete m_pSessionPre;
		m_pSessionPre = NULL;
	}
	if (m_pSessionSam) {
		delete m_pSessionSam;
		m_pSessionSam = NULL;
	}
	if (m_pImageCUDATemp) {
		cudaFree(m_pImageCUDATemp);
		m_pImageCUDATemp = NULL;
	}
	if (m_pImageCUDA) {
		cudaFree(m_pImageCUDA);
		m_pImageCUDA = NULL;
	}
	if (m_pPreprocessing) {
		cudaFree(m_pPreprocessing);
		m_pPreprocessing = NULL;
	}
	if (m_pSegmentation) {
		cudaFree(m_pSegmentation);
		m_pSegmentation = NULL;
	}
	if (m_pSegmentationFloat) {
		cudaFree(m_pSegmentationFloat);
		m_pSegmentationFloat = NULL;
	}
	if (m_maskInputValue) {
		cudaFree(m_maskInputValue);
		m_maskInputValue = NULL;
	}
	if (m_pMemoryInfo) {
		delete m_pMemoryInfo;
		m_pMemoryInfo = NULL;
	}
	if (m_pInputTensorPre) {
		delete m_pInputTensorPre;
		m_pInputTensorPre = NULL;
	}
	if (m_pOutputTensorSam) {
		delete m_pOutputTensorSam;
		m_pOutputTensorSam = NULL;
	}
	if (m_hasMaskValueCUDA) {
		cudaFree(m_hasMaskValueCUDA);
		m_hasMaskValueCUDA = NULL;
	}
	if (m_orig_im_size_values) {
		delete[]m_orig_im_size_values;
		m_orig_im_size_values = NULL;
	}
	if (m_orig_im_size_valuesCUDA) {
		cudaFree(m_orig_im_size_valuesCUDA);
		m_orig_im_size_valuesCUDA = NULL;
	}
}

int SJSegmentAnythingGPU::iDivUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

int SJSegmentAnythingGPU::InitializeSamModel(const char* preModelPath, const char* samModelPath, int width, int height)
{
	
	m_pMemoryInfo = new Ort::MemoryInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);

	OrtCUDAProviderOptions option;
	option.device_id = 0;
	OrtCUDAProviderOptions option1;
	option1.device_id = 0;
	m_pSessionOptions[0].SetInterOpNumThreads(std::thread::hardware_concurrency());
	m_pSessionOptions[0].SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	m_pSessionOptions[0].AppendExecutionProvider_CUDA(option);

	m_pSessionOptions[1].SetInterOpNumThreads(std::thread::hardware_concurrency());
	m_pSessionOptions[1].SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	m_pSessionOptions[1].AppendExecutionProvider_CUDA(option1);
	
	std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
	auto wpreModelPath = converter.from_bytes(preModelPath);
	auto wsamModelPath = converter.from_bytes(samModelPath);

	m_pSessionPre = new Ort::Session(m_env, wpreModelPath.c_str(), m_pSessionOptions[0]);
	m_pSessionSam = new Ort::Session(m_env, wsamModelPath.c_str(), m_pSessionOptions[1]);

	m_InputShapePre = m_pSessionPre->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	m_OutputShapePre = m_pSessionPre->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	m_OutputShapeSam = m_InputShapePre;
	m_OutputShapeSam[1] = 1;

	cudaMalloc((void**)&m_pImageCUDA, m_InputShapePre[0] * m_InputShapePre[1] * m_InputShapePre[2] * m_InputShapePre[3] * sizeof(unsigned char));
	//cudaMalloc((void**)&m_pImageCUDATemp, m_InputShapePre[0] * m_InputShapePre[1] * m_InputShapePre[2] * m_InputShapePre[3] * sizeof(unsigned char));
	
	//m_pPreprocessing = new float[m_OutputShapePre[0] * m_OutputShapePre[1] * m_OutputShapePre[2] * m_OutputShapePre[3]];
	cudaMalloc((void**)&m_pPreprocessing, m_OutputShapePre[0] * m_OutputShapePre[1] * m_OutputShapePre[2] * m_OutputShapePre[3] * sizeof(float));

	cudaMalloc((void**)&m_pSegmentationFloat, m_OutputShapeSam[0] * m_OutputShapeSam[1] * m_OutputShapeSam[2] * m_OutputShapeSam[3] * sizeof(float));
	cudaMalloc((void**)&m_pSegmentation, width * height * sizeof(unsigned char));
	
	//m_maskInputValue = new float[256 * 256];
	cudaMalloc((void**)&m_maskInputValue, 256 * 256 * sizeof(float));

	m_hasMaskValue = 0;
	cudaMalloc((void**)&m_hasMaskValueCUDA, sizeof(float));
	cudaMemcpy(m_hasMaskValueCUDA, &m_hasMaskValue, sizeof(float), cudaMemcpyHostToDevice);

	m_orig_im_size_values = new float[2];
	m_orig_im_size_values[0] = (float)m_InputShapePre[2];
	m_orig_im_size_values[1] = (float)m_InputShapePre[3];

	cudaMalloc((void**)&m_orig_im_size_valuesCUDA, 2 * sizeof(float));
	cudaMemcpy(m_orig_im_size_valuesCUDA, m_orig_im_size_values, 2 * sizeof(float), cudaMemcpyHostToDevice);


	m_pInputTensorPre = new Ort::Value(Ort::Value::CreateTensor<unsigned char>(*m_pMemoryInfo, m_pImageCUDA, m_InputShapePre[0] * m_InputShapePre[1] * m_InputShapePre[2] * m_InputShapePre[3], m_InputShapePre.data(), m_InputShapePre.size()));
	m_pOutputTensorSam = new Ort::Value(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_pSegmentationFloat, m_OutputShapeSam[0] * m_OutputShapeSam[1] * m_OutputShapeSam[2] * m_OutputShapeSam[3], m_OutputShapeSam.data(), m_OutputShapeSam.size()));

	m_vecInputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_pPreprocessing, m_OutputShapePre[0] * m_OutputShapePre[1] * m_OutputShapePre[2] * m_OutputShapePre[3], m_OutputShapePre.data(), m_OutputShapePre.size()));
	m_vecInputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_orig_im_size_valuesCUDA, 2, m_vecOrigImSizeShape.data(), m_vecOrigImSizeShape.size()));
	m_vecInputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_orig_im_size_valuesCUDA, 2, m_vecOrigImSizeShape.data(), m_vecOrigImSizeShape.size()));

	m_vecInputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_maskInputValue, 256 * 256, m_vecMaskInputShape.data(), m_vecMaskInputShape.size()));
	m_vecInputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_hasMaskValueCUDA, 1, m_vecHasMaskInputShape.data(), m_vecHasMaskInputShape.size()));
	m_vecInputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_orig_im_size_valuesCUDA, 2, m_vecOrigImSizeShape.data(), m_vecOrigImSizeShape.size()));

	m_cudaThread.x = BLOCKDIM_X;
	m_cudaThread.y = BLOCKDIM_Y;
	m_cudaThread.z = 1;
	m_cudaGrid.x = iDivUp(m_InputShapePre[3], BLOCKDIM_X);
	m_cudaGrid.y = iDivUp(m_InputShapePre[2], BLOCKDIM_Y);
	m_cudaGrid.z = 1;

	return 0;
}

cv::Size SJSegmentAnythingGPU::GetInputSize()
{
	return cv::Size(m_InputShapePre[3], m_InputShapePre[2]);
}

void SJSegmentAnythingGPU::SamLoadImage(const cv::Mat& image)
{

	if (!m_pImageCUDATemp) {
		cudaMalloc((void**)&m_pImageCUDATemp, image.cols * image.rows * 3 * sizeof(unsigned char));
	}

	cudaMemcpy(m_pImageCUDATemp, image.data, image.cols * image.rows * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	SamLoadImage(m_pImageCUDATemp, image.cols, image.rows);
}

void SJSegmentAnythingGPU::GetMask(const std::vector<cv::Point>& points, const std::vector<cv::Point>& negativePoints, const cv::Rect& roi, cv::Mat& outputMaskSam, double& iouValue)
{
	GetMask(points, negativePoints, roi, outputMaskSam.data, outputMaskSam.cols, outputMaskSam.rows, iouValue);

}
