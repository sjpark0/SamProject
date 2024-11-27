#include "SJSegmentAnythingCPU2.h"

#include <opencv2/opencv.hpp>
#include <codecvt>
#include <vector>

using namespace cv;
using namespace std;

SJSegmentAnythingCPU2::SJSegmentAnythingCPU2()
{
}
SJSegmentAnythingCPU2::~SJSegmentAnythingCPU2()
{
	if (m_pImage) {
		delete[]m_pImage;
		m_pImage = NULL;
	}
	if (m_pEmbedding) {
		delete[]m_pEmbedding;
		m_pEmbedding = NULL;
	}
	if (m_pFeature1) {
		delete[]m_pFeature1;
		m_pFeature1 = NULL;
	}
	if (m_pFeature2) {
		delete[]m_pFeature2;
		m_pFeature2 = NULL;
	}
	if (m_pSegmentation) {
		delete[]m_pSegmentation;
		m_pSegmentation = NULL;
	}
	if (m_pSegmentationFloat) {
		delete[]m_pSegmentationFloat;
		m_pSegmentationFloat = NULL;
	}
	if (m_maskInputValue) {
		delete[]m_maskInputValue;
		m_maskInputValue = NULL;
	}
	if (m_hasMaskValue) {
		delete[]m_hasMaskValue;
		m_hasMaskValue = NULL;
	}
	if (m_orig_im_size_values) {
		delete[]m_orig_im_size_values;
		m_orig_im_size_values = NULL;
	}
}

int SJSegmentAnythingCPU2::InitializeSamModel(const char* encModelPath, const char* decModelPath, int numCam, int width, int height)
{
	m_pMemoryInfo = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

	OrtCUDAProviderOptions option;
	option.device_id = 0;
	OrtCUDAProviderOptions option1;
	option1.device_id = 0;
	m_pSessionOptions[0].SetInterOpNumThreads(std::thread::hardware_concurrency());
	m_pSessionOptions[0].SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	m_pSessionOptions[1].SetInterOpNumThreads(std::thread::hardware_concurrency());
	m_pSessionOptions[1].SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);


	std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
	auto wpreModelPath = converter.from_bytes(encModelPath);
	auto wsamModelPath = converter.from_bytes(decModelPath);
	
	m_pSessionEnc = new Ort::Session(m_env, wpreModelPath.c_str(), m_pSessionOptions[0]);
	m_pSessionDec = new Ort::Session(m_env, wsamModelPath.c_str(), m_pSessionOptions[1]);
	
	m_InputShapeEnc = m_pSessionEnc->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();	
	Ort::AllocatorWithDefaultOptions allocator;
	
	/*for (int i = 0; i < m_pSessionEnc->GetInputCount(); i++) {
		printf("%s\n", m_pSessionEnc->GetInputNameAllocated(i, allocator).get());
	}
	for (int i = 0; i < m_pSessionEnc->GetOutputCount(); i++) {
		printf("%s\n", m_pSessionEnc->GetOutputNameAllocated(i, allocator).get());
	}

	for (int i = 0; i < m_pSessionDec->GetInputCount(); i++) {
		printf("%s\n", m_pSessionDec->GetInputNameAllocated(i, allocator).get());
	}
	for (int i = 0; i < m_pSessionDec->GetOutputCount(); i++) {
		printf("%s\n", m_pSessionDec->GetOutputNameAllocated(i, allocator).get());
	}*/

	m_OutputShapeDec = m_InputShapeEnc;
	m_OutputShapeDec[1] = 1;
	//m_OutputShapeDec = m_pSessionDec->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	
	m_pImage = new float[m_InputShapeEnc[0] * m_InputShapeEnc[1] * m_InputShapeEnc[2] * m_InputShapeEnc[3]];
	
	
	std::vector<int64_t> outputShapeEnc0 = m_pSessionDec->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
	std::vector<int64_t> outputShapeEnc1 = m_pSessionDec->GetInputTypeInfo(2).GetTensorTypeAndShapeInfo().GetShape();
	std::vector<int64_t> outputShapeEnc2 = m_pSessionDec->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

	//printf("%d, %d, %d, %d\n", outputShapeEnc0[0], outputShapeEnc0[1], outputShapeEnc0[2], outputShapeEnc0[3]);
	//printf("%d, %d, %d, %d\n", outputShapeEnc1[0], outputShapeEnc1[1], outputShapeEnc1[2], outputShapeEnc1[3]);
	//printf("%d, %d, %d, %d\n", outputShapeEnc2[0], outputShapeEnc2[1], outputShapeEnc2[2], outputShapeEnc2[3]);

	m_pFeature1 = new float[numCam * outputShapeEnc0[0] * outputShapeEnc0[1] * outputShapeEnc0[2] * outputShapeEnc0[3]];
	m_pFeature2 = new float[numCam * outputShapeEnc1[0] * outputShapeEnc1[1] * outputShapeEnc1[2] * outputShapeEnc1[3]];
	m_pEmbedding = new float[numCam * outputShapeEnc2[0] * outputShapeEnc2[1] * outputShapeEnc2[2] * outputShapeEnc2[3]];
	//printf("%d, %d, %d, %d\n", m_InputShapeEnc[0], m_InputShapeEnc[1], m_InputShapeEnc[2], m_InputShapeEnc[3]);
	//printf("%d, %d, %d, %d\n", m_OutputShapeDec[0], m_OutputShapeDec[1], m_OutputShapeDec[2], m_OutputShapeDec[3]);

	m_pSegmentationFloat = new float[m_OutputShapeDec[0] * m_OutputShapeDec[1] * m_OutputShapeDec[2] * m_OutputShapeDec[3]];
	
	m_maskInputValue = new float[256 * 256];
	memset(m_maskInputValue, 0, 256 * 256 * sizeof(float));
	m_hasMaskValue = new float[1];
	m_hasMaskValue[0] = 0.0;

	m_orig_im_size_values = new int[2];
	m_orig_im_size_values[0] = m_InputShapeEnc[2];
	m_orig_im_size_values[1] = m_InputShapeEnc[3];
	
	m_pInputTensorEnc = new Ort::Value(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_pImage, m_InputShapeEnc[0] * m_InputShapeEnc[1] * m_InputShapeEnc[2] * m_InputShapeEnc[3], m_InputShapeEnc.data(), m_InputShapeEnc.size()));
	m_pOutputTensorDec = new Ort::Value(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_pSegmentationFloat, m_OutputShapeDec[0] * m_OutputShapeDec[1] * m_OutputShapeDec[2] * m_OutputShapeDec[3], m_OutputShapeDec.data(), m_OutputShapeDec.size()));
	
	m_vecInputTensorsDec = new std::vector<Ort::Value>[numCam];
	m_vecOutputTensorsEnc = new std::vector<Ort::Value>[numCam];
	
	for (int i = 0; i < numCam; i++) {
		m_vecInputTensorsDec[i].push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, &m_pEmbedding[i * outputShapeEnc2[0] * outputShapeEnc2[1] * outputShapeEnc2[2] * outputShapeEnc2[3]], outputShapeEnc2[0] * outputShapeEnc2[1] * outputShapeEnc2[2] * outputShapeEnc2[3], outputShapeEnc2.data(), outputShapeEnc2.size()));
		m_vecInputTensorsDec[i].push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, &m_pFeature1[i * outputShapeEnc0[0] * outputShapeEnc0[1] * outputShapeEnc0[2] * outputShapeEnc0[3]], outputShapeEnc0[0] * outputShapeEnc0[1] * outputShapeEnc0[2] * outputShapeEnc0[3], outputShapeEnc0.data(), outputShapeEnc0.size()));
		m_vecInputTensorsDec[i].push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, &m_pFeature2[i * outputShapeEnc1[0] * outputShapeEnc1[1] * outputShapeEnc1[2] * outputShapeEnc1[3]], outputShapeEnc1[0] * outputShapeEnc1[1] * outputShapeEnc1[2] * outputShapeEnc1[3], outputShapeEnc1.data(), outputShapeEnc1.size()));

		m_vecInputTensorsDec[i].push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_hasMaskValue, 1, m_vecHasMaskInputShape.data(), m_vecHasMaskInputShape.size()));
		m_vecInputTensorsDec[i].push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_hasMaskValue, 1, m_vecHasMaskInputShape.data(), m_vecHasMaskInputShape.size()));

		m_vecInputTensorsDec[i].push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_maskInputValue, 256 * 256, m_vecMaskInputShape.data(), m_vecMaskInputShape.size()));
		m_vecInputTensorsDec[i].push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_hasMaskValue, 1, m_vecHasMaskInputShape.data(), m_vecHasMaskInputShape.size()));
		m_vecInputTensorsDec[i].push_back(Ort::Value::CreateTensor<int>(*m_pMemoryInfo, m_orig_im_size_values, 2, m_vecOrigImSizeShape.data(), m_vecOrigImSizeShape.size()));

		m_vecOutputTensorsEnc[i].push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, &m_pFeature1[i * outputShapeEnc0[0] * outputShapeEnc0[1] * outputShapeEnc0[2] * outputShapeEnc0[3]], outputShapeEnc0[0] * outputShapeEnc0[1] * outputShapeEnc0[2] * outputShapeEnc0[3], outputShapeEnc0.data(), outputShapeEnc0.size()));
		m_vecOutputTensorsEnc[i].push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, &m_pFeature2[i * outputShapeEnc1[0] * outputShapeEnc1[1] * outputShapeEnc1[2] * outputShapeEnc1[3]], outputShapeEnc1[0] * outputShapeEnc1[1] * outputShapeEnc1[2] * outputShapeEnc1[3], outputShapeEnc1.data(), outputShapeEnc1.size()));
		m_vecOutputTensorsEnc[i].push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, &m_pEmbedding[i * outputShapeEnc2[0] * outputShapeEnc2[1] * outputShapeEnc2[2] * outputShapeEnc2[3]], outputShapeEnc2[0] * outputShapeEnc2[1] * outputShapeEnc2[2] * outputShapeEnc2[3], outputShapeEnc2.data(), outputShapeEnc2.size()));
	}
	
}

void SJSegmentAnythingCPU2::SamLoadImage(const unsigned char* imageBuffer, int width, int height, int camID)
{
	printf("SamLoadImage Start\n");
	Mat src(height, width, CV_8UC3, (unsigned char*)imageBuffer);
	Mat dst(m_InputShapeEnc[2], m_InputShapeEnc[3], CV_8UC3);
	resize(src, dst, Size(m_InputShapeEnc[3], m_InputShapeEnc[2]));
	int index;
	for (int i = 0; i < m_InputShapeEnc[2]; i++) {
		for (int j = 0; j < m_InputShapeEnc[3]; j++) {
			index = j + i * m_InputShapeEnc[3];
			m_pImage[index] = (dst.at<Vec3b>(i, j)[2] / 255.0 - 0.485) / 0.229;
			m_pImage[index + m_InputShapeEnc[2] * m_InputShapeEnc[3]] = (dst.at<Vec3b>(i, j)[1] / 255.0 - 0.456) / 0.224;
			m_pImage[index + 2 * m_InputShapeEnc[2] * m_InputShapeEnc[3]] = (dst.at<Vec3b>(i, j)[0] / 255.0 - 0.406) / 0.225;
		}
	}
	
	//cout << m_OutputShapePre[0] << "," << m_OutputShapePre[1] << "," << m_OutputShapePre[2] << "," << m_OutputShapePre[3] << endl;
	Ort::RunOptions run_options;
	//m_pSessionPre->Run(run_options, m_inputNamesPre, m_pInputTensorPre, 1, m_outputNamesPre, &m_vecInputTensorsSam[camID][0], 1);
	m_pSessionEnc->Run(run_options, m_inputNamesEnc, m_pInputTensorEnc, 1, m_outputNamesEnc, m_vecOutputTensorsEnc[camID].data(), m_vecOutputTensorsEnc[camID].size());
	/*FILE* fp;
	fopen_s(&fp, "test.raw", "wb");
	fwrite(m_pPreprocessing, 1048576, sizeof(unsigned char), fp);
	fclose(fp);*/
	printf("SamLoadImage End\n");

}

void SJSegmentAnythingCPU2::GetMask(const std::vector<cv::Point>& points, const std::vector<cv::Point>& negativePoints, const cv::Rect& roi, unsigned char* mask, int width, int height, int camID)
{
	printf("GetMask Start\n");
	float factorX = (float)width / (float)m_InputShapeEnc[3];
	float factorY = (float)height / (float)m_InputShapeEnc[2];
	printf("%f, %f\n", factorX, factorY);
	printf("%f, %f\n", (float)points[0].x / factorX, (float)points[0].y / factorY);

	vector<float> inputPointValues, inputLabelValues;
	for (int i = 0; i < points.size(); i++) {
		inputPointValues.push_back((float)points[i].x / factorX);
		inputPointValues.push_back((float)points[i].y / factorY);
		inputLabelValues.push_back(1);
	}
	for (int i = 0; i < negativePoints.size(); i++) {
		inputPointValues.push_back((float)negativePoints[i].x / factorX);
		inputPointValues.push_back((float)negativePoints[i].y / factorY);
		inputLabelValues.push_back(0);
	}
	if (!roi.empty()) {
		inputPointValues.push_back((float)roi.x / factorX);
		inputPointValues.push_back((float)roi.y / factorY);
		inputLabelValues.push_back(2);
		inputPointValues.push_back((float)roi.br().x / factorX);
		inputPointValues.push_back((float)roi.br().y / factorY);
		inputLabelValues.push_back(3);
	}
	const int numPoints = inputLabelValues.size();
	
	float valIOU[] = { 0 };
	std::vector<int64_t> inputPointShape = { 1, numPoints, 2 };
	std::vector<int64_t> pointLabelShape = { 1, numPoints };
	m_vecInputTensorsDec[camID][3] = Ort::Value::CreateTensor<float>(*m_pMemoryInfo, inputPointValues.data(), 2 * numPoints, inputPointShape.data(), inputPointShape.size());
	m_vecInputTensorsDec[camID][4] = Ort::Value::CreateTensor<float>(*m_pMemoryInfo, inputLabelValues.data(), numPoints, pointLabelShape.data(), pointLabelShape.size());
	Ort::RunOptions runOptionsSam;
	m_pSessionDec->Run(runOptionsSam, m_inputNamesDec, m_vecInputTensorsDec[camID].data(), m_vecInputTensorsDec[camID].size(), m_outputNamesDec, m_pOutputTensorDec, 1);
	Mat outputMaskImage(m_OutputShapeDec[2], m_OutputShapeDec[3], CV_32FC1, m_pSegmentationFloat);
	resize(outputMaskImage, outputMaskImage, Size(width, height));

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			mask[j + i * width] = outputMaskImage.at<float>(i, j) > 0 ? 255 : 0;
		}
	}
	printf("GetMask End\n");

}
