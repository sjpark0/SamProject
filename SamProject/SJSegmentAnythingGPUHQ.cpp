#include "SJSegmentAnythingGPUHQ.h"
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

SJSegmentAnythingGPUHQ::SJSegmentAnythingGPUHQ()
{
	m_pSessionPre = NULL;
	m_pSessionSam = NULL;
	m_pImage = NULL;
	m_pImageCUDA = NULL;
	m_pPreprocessing = NULL;
	m_pInterm = NULL;
	m_pSegmentation = NULL;
	m_maskInputValue = NULL;
	m_pMemoryInfo = NULL;
	m_pInputTensorPre = NULL;
	m_pOutputTensorPre = NULL;
	m_pOutputTensorSam = NULL;
}
SJSegmentAnythingGPUHQ::~SJSegmentAnythingGPUHQ()
{
	if (m_pSessionPre) {
		delete m_pSessionPre;
		m_pSessionPre = NULL;
	}
	if (m_pSessionSam) {
		delete m_pSessionSam;
		m_pSessionSam = NULL;
	}
	if (m_pImage) {
		delete[]m_pImage;
		m_pImage = NULL;
	}
	if (m_pImageCUDA) {
		cudaFree(m_pImageCUDA);
		m_pImageCUDA = NULL;
	}
	if (m_pPreprocessing) {
		cudaFree(m_pPreprocessing);
		m_pPreprocessing = NULL;
	}
	if (m_pInterm) {
		cudaFree(m_pInterm);
		m_pInterm = NULL;
	}
	if (m_pSegmentation) {
		delete[]m_pSegmentation;
		m_pSegmentation = NULL;
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
	if (m_pOutputTensorPre) {
		delete []m_pOutputTensorPre;
		m_pOutputTensorPre = NULL;
	}
	if (m_pOutputTensorSam) {
		delete m_pOutputTensorSam;
		m_pOutputTensorSam = NULL;
	}
}

int SJSegmentAnythingGPUHQ::InitializeSamModel(const char* preModelPath, const char* samModelPath)
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
	//OrtSessionOptionsAppendExecutionProvider_CUDA(m_pSessionOptions[0], 0);
	//OrtSessionOptionsAppendExecutionProvider_CUDA(m_pSessionOptions[1], 0);

	std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
	auto wpreModelPath = converter.from_bytes(preModelPath);
	auto wsamModelPath = converter.from_bytes(samModelPath);

	m_pSessionPre = new Ort::Session(m_env, wpreModelPath.c_str(), m_pSessionOptions[0]);
	m_pSessionSam = new Ort::Session(m_env, wsamModelPath.c_str(), m_pSessionOptions[1]);

	m_InputShapePre = m_pSessionPre->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	m_OutputShapePre = m_pSessionPre->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	m_IntermShapePre = m_pSessionSam->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();


	m_OutputShapeSam = m_InputShapePre;
	m_OutputShapeSam[1] = 1;

	m_pImage = new unsigned char[m_InputShapePre[0] * m_InputShapePre[1] * m_InputShapePre[2] * m_InputShapePre[3]];
	cudaMalloc((void**)&m_pImageCUDA, m_InputShapePre[0] * m_InputShapePre[1] * m_InputShapePre[2] * m_InputShapePre[3] * sizeof(unsigned char));

	cudaMalloc((void**)&m_pPreprocessing, m_OutputShapePre[0] * m_OutputShapePre[1] * m_OutputShapePre[2] * m_OutputShapePre[3] * sizeof(float));
	cudaMalloc((void**)&m_pInterm, m_IntermShapePre[0] * m_IntermShapePre[1] * m_IntermShapePre[2] * m_IntermShapePre[3] * m_IntermShapePre[4] * sizeof(float));

	cudaMalloc((void**)&m_pSegmentationCUDA, m_OutputShapeSam[0] * m_OutputShapeSam[1] * m_OutputShapeSam[2] * m_OutputShapeSam[3] * sizeof(float));
	m_pSegmentation = new float[m_OutputShapeSam[0] * m_OutputShapeSam[1] * m_OutputShapeSam[2] * m_OutputShapeSam[3]];

	//m_maskInputValue = new float[256 * 256];
	cudaMalloc((void**)&m_maskInputValue, 256 * 256 * sizeof(float));

	m_pInputTensorPre = new Ort::Value(Ort::Value::CreateTensor<unsigned char>(*m_pMemoryInfo, m_pImageCUDA, m_InputShapePre[0] * m_InputShapePre[1] * m_InputShapePre[2] * m_InputShapePre[3], m_InputShapePre.data(), m_InputShapePre.size()));
	m_pOutputTensorPre = new Ort::Value[2]{ 
		Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_pPreprocessing, m_OutputShapePre[0] * m_OutputShapePre[1] * m_OutputShapePre[2] * m_OutputShapePre[3], m_OutputShapePre.data(), m_OutputShapePre.size()),
		Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_pInterm, m_IntermShapePre[0] * m_IntermShapePre[1] * m_IntermShapePre[2] * m_IntermShapePre[3] * m_IntermShapePre[4], m_IntermShapePre.data(), m_IntermShapePre.size())
	};
	m_pOutputTensorSam = new Ort::Value(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_pSegmentationCUDA, m_OutputShapeSam[0] * m_OutputShapeSam[1] * m_OutputShapeSam[2] * m_OutputShapeSam[3], m_OutputShapeSam.data(), m_OutputShapeSam.size()));

	return 0;
}

cv::Size SJSegmentAnythingGPUHQ::GetInputSize()
{
	return cv::Size(m_InputShapePre[3], m_InputShapePre[2]);
}

void SJSegmentAnythingGPUHQ::SamLoadImage(const cv::Mat& image)
{

	int index;
	for (int i = 0; i < m_InputShapePre[2]; i++) {
		for (int j = 0; j < m_InputShapePre[3]; j++) {
			index = j + i * m_InputShapePre[3];
			m_pImage[index] = image.at<Vec3b>(i, j)[2];
			m_pImage[index + m_InputShapePre[2] * m_InputShapePre[3]] = image.at<Vec3b>(i, j)[1];
			m_pImage[index + 2 * m_InputShapePre[2] * m_InputShapePre[3]] = image.at<Vec3b>(i, j)[0];
		}
	}
	cudaMemcpy(m_pImageCUDA, m_pImage, m_InputShapePre[0] * m_InputShapePre[1] * m_InputShapePre[2] * m_InputShapePre[3] * sizeof(unsigned char), cudaMemcpyHostToDevice);

	Ort::RunOptions run_options;
	m_pSessionPre->Run(run_options, m_inputNamesPre, m_pInputTensorPre, 1, m_outputNamesPre, m_pOutputTensorPre, 2);
	/*FILE* fp;
	fopen_s(&fp, "test5.raw", "wb");
	fwrite(m_pOutputTensorPre->GetTensorMutableData<unsigned char>(), 1048576, sizeof(unsigned char), fp);
	fclose(fp);

	fopen_s(&fp, "test6.raw", "wb");
	fwrite(m_pPreprocessing, 1048576, sizeof(unsigned char), fp);
	fclose(fp);*/
}

void SJSegmentAnythingGPUHQ::GetMask(const std::vector<cv::Point>& points, const std::vector<cv::Point>& negativePoints, const cv::Rect& roi, cv::Mat& outputMaskSam, double& iouValue)
{
	float hasMaskValue = 0;
	float* hasMaskValueCUDA;
	float orig_im_size_values[] = { (float)m_InputShapePre[2], (float)m_InputShapePre[3] };
	float* orig_im_size_valuesCUDA;

	cudaMalloc((void**)&hasMaskValueCUDA, sizeof(float));
	cudaMemcpy(hasMaskValueCUDA, &hasMaskValue, sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&orig_im_size_valuesCUDA, 2 * sizeof(float));
	cudaMemcpy(orig_im_size_valuesCUDA, orig_im_size_values, 2 * sizeof(float), cudaMemcpyHostToDevice);

	//memset(m_maskInputValue, 0, 256 * 256 * sizeof(float));

	vector<float> inputPointValues, inputLabelValues;
	for (int i = 0; i < points.size(); i++) {
		inputPointValues.push_back((float)points[i].x);
		inputPointValues.push_back((float)points[i].y);
		inputLabelValues.push_back(1);
	}
	for (int i = 0; i < negativePoints.size(); i++) {
		inputPointValues.push_back((float)negativePoints[i].x);
		inputPointValues.push_back((float)negativePoints[i].y);
		inputLabelValues.push_back(0);
	}
	if (!roi.empty()) {
		inputPointValues.push_back((float)roi.x);
		inputPointValues.push_back((float)roi.y);
		inputLabelValues.push_back(2);
		inputPointValues.push_back((float)roi.br().x);
		inputPointValues.push_back((float)roi.br().y);
		inputLabelValues.push_back(3);
	}
	const int numPoints = inputLabelValues.size();

	float* inputPointValuesCUDA;
	float* labelPointValuesCUDA;
	cudaMalloc((void**)&inputPointValuesCUDA, numPoints * 2 * sizeof(float));
	cudaMemcpy(inputPointValuesCUDA, inputPointValues.data(), numPoints * 2 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&labelPointValuesCUDA, numPoints * sizeof(float));
	cudaMemcpy(labelPointValuesCUDA, inputLabelValues.data(), numPoints * sizeof(float), cudaMemcpyHostToDevice);


	float valIOU[] = { 0 };
	std::vector<int64_t> inputPointShape = { 1, numPoints, 2 };
	std::vector<int64_t> pointLabelShape = { 1, numPoints };
	std::vector<int64_t> maskInputShape = { 1, 1, 256, 256 };
	std::vector<int64_t> hasMaskInputShape = { 1 };
	std::vector<int64_t> origImSizeShape = { 2 };

	std::vector<Ort::Value> inputTensorsSam;
	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_pPreprocessing, m_OutputShapePre[0] * m_OutputShapePre[1] * m_OutputShapePre[2] * m_OutputShapePre[3], m_OutputShapePre.data(), m_OutputShapePre.size()));
	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_pInterm, m_IntermShapePre[0] * m_IntermShapePre[1] * m_IntermShapePre[2] * m_IntermShapePre[3] * m_IntermShapePre[4], m_IntermShapePre.data(), m_IntermShapePre.size()));

	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, inputPointValuesCUDA, 2 * numPoints, inputPointShape.data(), inputPointShape.size()));
	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, labelPointValuesCUDA, numPoints, pointLabelShape.data(), pointLabelShape.size()));

	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_maskInputValue, 256 * 256, maskInputShape.data(), maskInputShape.size()));
	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, hasMaskValueCUDA, 1, hasMaskInputShape.data(), hasMaskInputShape.size()));
	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, orig_im_size_valuesCUDA, 2, origImSizeShape.data(), origImSizeShape.size()));

	int outputNumber = 3, outputMaskIndex = 0, outputIOUIndex = 1;
	Ort::RunOptions runOptionsSam;


	//m_pSessionSam->Run(runOptionsSam, m_inputNamesSamHQ, inputTensorsSam.data(), inputTensorsSam.size(), m_outputNamesSam, m_pOutputTensorSam, 1);
	vector<Ort::Value> outputTensorSam = m_pSessionSam->Run(runOptionsSam, m_inputNamesSamHQ, inputTensorsSam.data(), inputTensorsSam.size(), m_outputNamesSam, 3);

	cudaMemcpy(m_pSegmentation, m_pSegmentationCUDA, m_OutputShapeSam[0] * m_OutputShapeSam[1] * m_OutputShapeSam[2] * m_OutputShapeSam[3] * sizeof(float), cudaMemcpyDeviceToHost);
	Mat outputMaskImage(m_OutputShapeSam[2], m_OutputShapeSam[3], CV_32FC1, m_pSegmentation);


	resize(outputMaskImage, outputMaskImage, outputMaskSam.size());
	for (int i = 0; i < outputMaskSam.rows; i++) {
		for (int j = 0; j < outputMaskSam.cols; j++) {
			outputMaskSam.at<uint8_t>(i, j) = outputMaskImage.at<float>(i, j) > 0 ? 255 : 0;
		}
	}

	cudaFree(inputPointValuesCUDA);
	cudaFree(labelPointValuesCUDA);
	cudaFree(hasMaskValueCUDA);
	cudaFree(orig_im_size_valuesCUDA);

}

