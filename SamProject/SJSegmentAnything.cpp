#include "SJSegmentAnything.h"
#include <codecvt>
#include <fstream>
#include <iostream>
#include <locale>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;

SJSegmentAnything::SJSegmentAnything()
{
	m_pSessionPre = NULL;
	m_pSessionSam = NULL;
	m_pImage = NULL;
	m_pPreprocessing = NULL;
	m_pSegmentation = NULL;
	m_maskInputValue = NULL;
	m_pMemoryInfo = NULL;
	m_pInputTensorPre = NULL;
	m_pOutputTensorSam = NULL;
	m_orig_im_size_values = NULL;
}
SJSegmentAnything::~SJSegmentAnything()
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
	if (m_pPreprocessing) {
		delete[]m_pPreprocessing;
		m_pPreprocessing = NULL;
	}
	if (m_pSegmentation) {
		delete[]m_pSegmentation;
		m_pSegmentation = NULL;
	}
	if (m_maskInputValue) {
		delete[]m_maskInputValue;
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
	if (m_orig_im_size_values) {
		delete[]m_orig_im_size_values;
		m_orig_im_size_values = NULL;
	}
}
int SJSegmentAnything::InitializeSamModel(const char* preModelPath, const char* samModelPath, int width, int height)
{
	m_pMemoryInfo = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
	OrtCUDAProviderOptions option;
	option.device_id = 0;
	OrtCUDAProviderOptions option1;
	option1.device_id = 0;
	m_pSessionOptions[0].SetIntraOpNumThreads(std::thread::hardware_concurrency());
	m_pSessionOptions[0].SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	//m_pSessionOptions[0].AppendExecutionProvider_CUDA(option);
	
	m_pSessionOptions[1].SetIntraOpNumThreads(std::thread::hardware_concurrency());
	m_pSessionOptions[1].SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	//m_pSessionOptions[1].AppendExecutionProvider_CUDA(option1);
	
	
	
	std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
	auto wpreModelPath = converter.from_bytes(preModelPath);
	auto wsamModelPath = converter.from_bytes(samModelPath);
	
	m_pSessionPre = new Ort::Session(m_env, wpreModelPath.c_str(), m_pSessionOptions[0]);
	m_pSessionSam = new Ort::Session(m_env, wsamModelPath.c_str(), m_pSessionOptions[1]);
	
	m_InputShapePre = m_pSessionPre->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	m_OutputShapePre = m_pSessionPre->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	m_OutputShapeSam = m_InputShapePre;
	m_OutputShapeSam[1] = 1;
	
	m_pImage = new unsigned char[m_InputShapePre[0] * m_InputShapePre[1] * m_InputShapePre[2] * m_InputShapePre[3]];
	m_pPreprocessing = new float[m_OutputShapePre[0] * m_OutputShapePre[1] * m_OutputShapePre[2] * m_OutputShapePre[3]];
	m_pSegmentationFloat = new float[m_OutputShapeSam[0] * m_OutputShapeSam[1] * m_OutputShapeSam[2] * m_OutputShapeSam[3]];
	m_pSegmentation = new unsigned char[width * height];
	m_maskInputValue = new float[256 * 256];
	memset(m_maskInputValue, 0, 256 * 256 * sizeof(float));
	m_hasMaskValue = 0;
	m_orig_im_size_values = new float[2];
	m_orig_im_size_values[0] = (float)m_InputShapePre[2];
	m_orig_im_size_values[1] = (float)m_InputShapePre[3];

	m_pInputTensorPre = new Ort::Value(Ort::Value::CreateTensor<unsigned char>(*m_pMemoryInfo, m_pImage, m_InputShapePre[0] * m_InputShapePre[1] * m_InputShapePre[2] * m_InputShapePre[3], m_InputShapePre.data(), m_InputShapePre.size()));
	m_pOutputTensorSam = new Ort::Value(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_pSegmentationFloat, m_OutputShapeSam[0] * m_OutputShapeSam[1] * m_OutputShapeSam[2] * m_OutputShapeSam[3], m_OutputShapeSam.data(), m_OutputShapeSam.size()));

	m_vecInputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_pPreprocessing, m_OutputShapePre[0] * m_OutputShapePre[1] * m_OutputShapePre[2] * m_OutputShapePre[3], m_OutputShapePre.data(), m_OutputShapePre.size()));
	m_vecInputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_orig_im_size_values, 2, m_vecOrigImSizeShape.data(), m_vecOrigImSizeShape.size()));
	m_vecInputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_orig_im_size_values, 2, m_vecOrigImSizeShape.data(), m_vecOrigImSizeShape.size()));

	m_vecInputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_maskInputValue, 256 * 256, m_vecMaskInputShape.data(), m_vecMaskInputShape.size()));
	m_vecInputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, &m_hasMaskValue, 1, m_vecHasMaskInputShape.data(), m_vecHasMaskInputShape.size()));
	m_vecInputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_orig_im_size_values, 2, m_vecOrigImSizeShape.data(), m_vecOrigImSizeShape.size()));

	return 0;
}

cv::Size SJSegmentAnything::GetInputSize()
{
	return cv::Size(m_InputShapePre[3], m_InputShapePre[2]);
}

void SJSegmentAnything::SamLoadImage(const cv::Mat& image)
{
	//Mat dst(m_InputShapePre[2], m_InputShapePre[3], CV_8UC3);
	Mat dst;
	resize(image, dst, Size(m_InputShapePre[3], m_InputShapePre[2]));
	//imwrite("test.png", dst);
	int index;
	for (int i = 0; i < m_InputShapePre[2]; i++) {
		for (int j = 0; j < m_InputShapePre[3]; j++) {
			index = j + i * m_InputShapePre[3];
			m_pImage[index] = dst.at<Vec3b>(i, j)[2];
			m_pImage[index + m_InputShapePre[2] * m_InputShapePre[3]] = dst.at<Vec3b>(i, j)[1];
			m_pImage[index + 2 * m_InputShapePre[2] * m_InputShapePre[3]] = dst.at<Vec3b>(i, j)[0];
		}
	}
	Ort::RunOptions run_options;
	m_pSessionPre->Run(run_options, m_inputNamesPre, m_pInputTensorPre, 1, m_outputNamesPre, &m_vecInputTensorsSam[0], 1);

	/*FILE* fp;
	fopen_s(&fp, "test5.raw", "wb");
	fwrite(m_vecInputTensorsSam[0].GetTensorMutableData<unsigned char>(), 1048576, sizeof(unsigned char), fp);
	fclose(fp);
	
	fopen_s(&fp, "test6.raw", "wb");
	fwrite(m_pPreprocessing, 1048576, sizeof(unsigned char), fp);
	fclose(fp);*/
}

void SJSegmentAnything::GetMask(const std::vector<cv::Point>& points, const std::vector<cv::Point>& negativePoints, const cv::Rect& roi, cv::Mat& outputMaskSam, double& iouValue)
{
	GetMask(points, negativePoints, roi, outputMaskSam.data, outputMaskSam.cols, outputMaskSam.rows, iouValue);

}

void SJSegmentAnything::GetMask(const std::vector<cv::Point>& points, const std::vector<cv::Point>& negativePoints, const cv::Rect& roi, unsigned char* mask, int width, int height, double& iouValu)
{	
	float factor = (float)width / (float)m_InputShapePre[3];
	vector<float> inputPointValues, inputLabelValues;
	for (int i = 0; i < points.size(); i++) {
		inputPointValues.push_back((float)points[i].x / factor);
		inputPointValues.push_back((float)points[i].y / factor);
		inputLabelValues.push_back(1);
	}
	for (int i = 0; i < negativePoints.size(); i++) {
		inputPointValues.push_back((float)negativePoints[i].x / factor);
		inputPointValues.push_back((float)negativePoints[i].y / factor);
		inputLabelValues.push_back(0);
	}
	cout << inputPointValues[0] << "," << inputPointValues[1] << endl;
	if (!roi.empty()) {
		inputPointValues.push_back((float)roi.x / factor);
		inputPointValues.push_back((float)roi.y / factor);
		inputLabelValues.push_back(2);
		inputPointValues.push_back((float)roi.br().x / factor);
		inputPointValues.push_back((float)roi.br().y / factor);
		inputLabelValues.push_back(3);
	}
	const int numPoints = inputLabelValues.size();
	float valIOU[] = { 0 };
	
	std::vector<int64_t> inputPointShape = { 1, numPoints, 2 };
	std::vector<int64_t> pointLabelShape = { 1, numPoints };

	m_vecInputTensorsSam[1] = Ort::Value::CreateTensor<float>(*m_pMemoryInfo, inputPointValues.data(), 2 * numPoints, inputPointShape.data(), inputPointShape.size());
	m_vecInputTensorsSam[2] = Ort::Value::CreateTensor<float>(*m_pMemoryInfo, inputLabelValues.data(), numPoints, pointLabelShape.data(), pointLabelShape.size());
	
	int outputNumber = 3, outputMaskIndex = 0, outputIOUIndex = 1;
	Ort::RunOptions runOptionsSam;

	
	m_pSessionSam->Run(runOptionsSam, m_inputNamesSam, m_vecInputTensorsSam.data(), m_vecInputTensorsSam.size(), m_outputNamesSam, m_pOutputTensorSam, 1);
	
	Mat outputMaskImage(m_OutputShapeSam[2], m_OutputShapeSam[3], CV_32FC1, m_pSegmentationFloat);


	resize(outputMaskImage, outputMaskImage, Size(width, height));
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			mask[j + i * width] = outputMaskImage.at<float>(i, j) > 0 ? 255 : 0;
		}
	}
}

