#include "SJSegmentAnythingTRT.h"
#include <codecvt>
#include <fstream>
#include <iostream>
#include <locale>
#include <opencv2/opencv.hpp>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "logger.h"
#include "argsParser.h"
#include "NvOnnxParser.h"
#include "common.h"

#include <cstdlib>
#include <fstream>
using namespace cv;
using namespace std;

void Logger::log(Severity severity, const char* msg) noexcept {
	switch (severity) {
	case Severity::kVERBOSE:
		//printf("%s\n", msg);
		break;
	case Severity::kINFO:
		//printf("%s\n", msg);
		break;
	case Severity::kWARNING:
		//printf("%s\n", msg);
		break;
	case Severity::kERROR:
		printf("%s\n", msg);
		break;
	case Severity::kINTERNAL_ERROR:
		printf("%s\n", msg);
		break;
	default:
		printf("Unexpected severity level\n");
		
	}
}

SJSegmentAnythingTRT::SJSegmentAnythingTRT()
{
	m_pImage = NULL;
	m_pImageCUDA = NULL;
	m_pPreprocessing = NULL;
	m_pSegmentation = NULL;
	m_maskInputValue = NULL;
	
}
SJSegmentAnythingTRT::~SJSegmentAnythingTRT()
{
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
	if (m_pSegmentation) {
		delete[]m_pSegmentation;
		m_pSegmentation = NULL;
	}
	if (m_maskInputValue) {
		cudaFree(m_maskInputValue);
		m_maskInputValue = NULL;
	}
	
}

int SJSegmentAnythingTRT::InitializeSamModel(const char* preModelPath, const char* samModelPath)
{
	m_ppBuilder = new nvinfer1::IBuilder * [2];
	m_ppNetwork = new nvinfer1::INetworkDefinition * [2];
	m_ppConfig = new nvinfer1::IBuilderConfig * [2];
	m_ppParser = new nvonnxparser::IParser * [2];
	m_ppPlan = new nvinfer1::IHostMemory * [2];
	m_ppRuntime = new nvinfer1::IRuntime * [2];
	m_ppEngine = new nvinfer1::ICudaEngine * [2];


	for (int i = 0; i < 2; i++) {
		m_ppBuilder[i] = nvinfer1::createInferBuilder(m_logger);
		m_ppNetwork[i] = m_ppBuilder[i]->createNetworkV2(0);
		m_ppConfig[i] = m_ppBuilder[i]->createBuilderConfig();
		m_ppParser[i] = nvonnxparser::createParser(*m_ppNetwork[i], m_logger);
	}
	bool bPre = m_ppParser[0]->parseFromFile(preModelPath, 1);
	bool bSam = m_ppParser[1]->parseFromFile(samModelPath, 0);

	printf("%d, %d\n", bPre, bSam);
	for (int i = 0; i < 2; i++){
		m_ppPlan[i] = m_ppBuilder[i]->buildSerializedNetwork(*m_ppNetwork[i], *m_ppConfig[i]);
		m_ppRuntime[i] = nvinfer1::createInferRuntime(m_logger);
		m_ppEngine[i] = m_ppRuntime[i]->deserializeCudaEngine(m_ppPlan[i]->data(), m_ppPlan[i]->size());
	}
	

	/*nvinfer1::IHostMemory* m_plan = m_builder->buildSerializedNetwork(*m_network, *m_config);
	if (m_plan == NULL) {
		return -1;
	}
	printf("%d\n", m_plan);

	nvinfer1::IRuntime* m_runtime = nvinfer1::createInferRuntime(m_logger);
	if (m_runtime == NULL) {
		return -1;
	}
	printf("%d\n", m_runtime);*/
	//nvinfer1::ICudaEngine* m_engine = m_runtime->deserializeCudaEngine(m_plan->data(), m_plan->size());

	m_preInputDims = m_ppNetwork[0]->getInput(0)->getDimensions();
	m_preOutputDims = m_ppNetwork[0]->getOutput(0)->getDimensions();
	m_samInputDims = m_ppNetwork[1]->getInput(0)->getDimensions();
	m_samOutputDims = m_ppNetwork[1]->getOutput(0)->getDimensions();

	cout << m_preInputDims << endl;
	cout << m_preOutputDims << endl;
	cout << m_samInputDims << endl;
	cout << m_samOutputDims << endl;

	//m_pImage = new unsigned char[m_inputDims.d[0] * m_inputDims.d[1] * m_inputDims.d[2] * m_inputDims.d[3]];
	//cudaMalloc((void**)&m_pImageCUDA, m_inputDims.d[0] * m_inputDims.d[1] * m_inputDims.d[2] * m_inputDims.d[3] * sizeof(unsigned char));

	//m_pPreprocessing = new float[m_OutputShapePre[0] * m_OutputShapePre[1] * m_OutputShapePre[2] * m_OutputShapePre[3]];
	//cudaMalloc((void**)&m_pPreprocessing, m_outputDims.d[0] * m_outputDims.d[1] * m_outputDims.d[2] * m_outputDims.d[3] * sizeof(float));

	
	for (int i = 0; i < 2; i++) {
		//m_ppContext[i] = m_ppEngine[i]->createExecutionContext();
		
		for (int j = 0; j < m_ppEngine[i]->getNbIOTensors(); j++) {
			const char* name = m_ppEngine[i]->getIOTensorName(j);
			cout << name << endl;
		}
	//	m_context->setTensorAddress("input", m_pImageCUDA);
	//	m_context->setTensorAddress("output", m_pPreprocessing);
	}
	return 0;
}

cv::Size SJSegmentAnythingTRT::GetInputSize()
{
	return cv::Size(m_preInputDims.d[3], m_preInputDims.d[2]);
}

void SJSegmentAnythingTRT::SamLoadImage(const cv::Mat& image)
{

	int index;
	for (int i = 0; i < m_preInputDims.d[2]; i++) {
		for (int j = 0; j < m_preInputDims.d[3]; j++) {
			index = j + i * m_preInputDims.d[3];
			m_pImage[index] = image.at<Vec3b>(i, j)[2];
			m_pImage[index + m_preInputDims.d[2] * m_preInputDims.d[3]] = image.at<Vec3b>(i, j)[1];
			m_pImage[index + 2 * m_preInputDims.d[2] * m_preInputDims.d[3]] = image.at<Vec3b>(i, j)[0];
		}
	}
	cudaMemcpy(m_pImageCUDA, m_pImage, m_preInputDims.d[0] * m_preInputDims.d[1] * m_preInputDims.d[2] * m_preInputDims.d[3] * sizeof(unsigned char), cudaMemcpyHostToDevice);

	//Ort::RunOptions run_options;
	//m_pSessionPre->Run(run_options, m_inputNamesPre, m_pInputTensorPre, 1, m_outputNamesPre, m_pOutputTensorPre, 1);
	/*FILE* fp;
	fopen_s(&fp, "test5.raw", "wb");
	fwrite(m_pOutputTensorPre->GetTensorMutableData<unsigned char>(), 1048576, sizeof(unsigned char), fp);
	fclose(fp);

	fopen_s(&fp, "test6.raw", "wb");
	fwrite(m_pPreprocessing, 1048576, sizeof(unsigned char), fp);
	fclose(fp);*/
}

void SJSegmentAnythingTRT::GetMask(const std::vector<cv::Point>& points, const std::vector<cv::Point>& negativePoints, const cv::Rect& roi, cv::Mat& outputMaskSam, double& iouValue)
{
	/*float hasMaskValue = 0;
	float* hasMaskValueCUDA;
	float orig_im_size_values[] = { (float)m_preInputDims.d[2], (float)m_preInputDims.d[3] };
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
	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, inputPointValuesCUDA, 2 * numPoints, inputPointShape.data(), inputPointShape.size()));
	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, labelPointValuesCUDA, numPoints, pointLabelShape.data(), pointLabelShape.size()));

	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_maskInputValue, 256 * 256, maskInputShape.data(), maskInputShape.size()));
	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, hasMaskValueCUDA, 1, hasMaskInputShape.data(), hasMaskInputShape.size()));
	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, orig_im_size_valuesCUDA, 2, origImSizeShape.data(), origImSizeShape.size()));

	int outputNumber = 3, outputMaskIndex = 0, outputIOUIndex = 1;
	Ort::RunOptions runOptionsSam;


	m_pSessionSam->Run(runOptionsSam, m_inputNamesSam, inputTensorsSam.data(), inputTensorsSam.size(), m_outputNamesSam, m_pOutputTensorSam, 1);
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
	cudaFree(orig_im_size_valuesCUDA);*/

}

