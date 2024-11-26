#include "SJSegmentAnythingCPU.h"

#include <opencv2/opencv.hpp>
#include <codecvt>
#include <vector>

using namespace cv;
using namespace std;

SJSegmentAnythingCPU::SJSegmentAnythingCPU()
{
}
SJSegmentAnythingCPU::~SJSegmentAnythingCPU()
{
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

int SJSegmentAnythingCPU::InitializeSamModel(const char* preModelPath, const char* samModelPath, int numCam, int width, int height)
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
	auto wpreModelPath = converter.from_bytes(preModelPath);
	auto wsamModelPath = converter.from_bytes(samModelPath);

	m_pSessionPre = new Ort::Session(m_env, wpreModelPath.c_str(), m_pSessionOptions[0]);
	m_pSessionSam = new Ort::Session(m_env, wsamModelPath.c_str(), m_pSessionOptions[1]);

	m_InputShapePre = m_pSessionPre->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	m_OutputShapePre = m_pSessionPre->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	m_OutputShapeSam = m_InputShapePre;
	m_OutputShapeSam[1] = 1;

	m_pImage = new unsigned char[m_InputShapePre[0] * m_InputShapePre[1] * m_InputShapePre[2] * m_InputShapePre[3]];
	m_pPreprocessing = new float[numCam * m_OutputShapePre[0] * m_OutputShapePre[1] * m_OutputShapePre[2] * m_OutputShapePre[3]];
	m_pSegmentationFloat = new float[m_OutputShapeSam[0] * m_OutputShapeSam[1] * m_OutputShapeSam[2] * m_OutputShapeSam[3]];

	m_maskInputValue = new float[256 * 256];
	memset(m_maskInputValue, 0, 256 * 256 * sizeof(float));
	m_hasMaskValue = new float[1];
	m_hasMaskValue[0] = 0.0;

	m_orig_im_size_values = new float[2];
	m_orig_im_size_values[0] = (float)m_InputShapePre[2];
	m_orig_im_size_values[1] = (float)m_InputShapePre[3];


	m_pInputTensorPre = new Ort::Value(Ort::Value::CreateTensor<unsigned char>(*m_pMemoryInfo, m_pImage, m_InputShapePre[0] * m_InputShapePre[1] * m_InputShapePre[2] * m_InputShapePre[3], m_InputShapePre.data(), m_InputShapePre.size()));
	m_pOutputTensorSam = new Ort::Value(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_pSegmentationFloat, m_OutputShapeSam[0] * m_OutputShapeSam[1] * m_OutputShapeSam[2] * m_OutputShapeSam[3], m_OutputShapeSam.data(), m_OutputShapeSam.size()));

	m_vecInputTensorsSam = new std::vector<Ort::Value>[numCam];
	for (int i = 0; i < numCam; i++) {
		m_vecInputTensorsSam[i].push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, &m_pPreprocessing[i * m_OutputShapePre[0] * m_OutputShapePre[1] * m_OutputShapePre[2] * m_OutputShapePre[3]], m_OutputShapePre[0] * m_OutputShapePre[1] * m_OutputShapePre[2] * m_OutputShapePre[3], m_OutputShapePre.data(), m_OutputShapePre.size()));
		m_vecInputTensorsSam[i].push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_orig_im_size_values, 2, m_vecOrigImSizeShape.data(), m_vecOrigImSizeShape.size()));
		m_vecInputTensorsSam[i].push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_orig_im_size_values, 2, m_vecOrigImSizeShape.data(), m_vecOrigImSizeShape.size()));

		m_vecInputTensorsSam[i].push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_maskInputValue, 256 * 256, m_vecMaskInputShape.data(), m_vecMaskInputShape.size()));
		m_vecInputTensorsSam[i].push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_hasMaskValue, 1, m_vecHasMaskInputShape.data(), m_vecHasMaskInputShape.size()));
		m_vecInputTensorsSam[i].push_back(Ort::Value::CreateTensor<float>(*m_pMemoryInfo, m_orig_im_size_values, 2, m_vecOrigImSizeShape.data(), m_vecOrigImSizeShape.size()));
	}
}

void SJSegmentAnythingCPU::SamLoadImage(const unsigned char* imageBuffer, int width, int height, int camID)
{
	Mat src(height, width, CV_8UC3, (unsigned char*)imageBuffer);
	Mat dst(m_InputShapePre[2], m_InputShapePre[3], CV_8UC3);
	resize(src, dst, Size(m_InputShapePre[3], m_InputShapePre[2]));

	int index;
	for (int i = 0; i < m_InputShapePre[2]; i++) {
		for (int j = 0; j < m_InputShapePre[3]; j++) {
			index = j + i * m_InputShapePre[3];
			m_pImage[index] = dst.at<Vec3b>(i, j)[2];
			m_pImage[index + m_InputShapePre[2] * m_InputShapePre[3]] = dst.at<Vec3b>(i, j)[1];
			m_pImage[index + 2 * m_InputShapePre[2] * m_InputShapePre[3]] = dst.at<Vec3b>(i, j)[0];
		}
	}

	//cout << m_OutputShapePre[0] << "," << m_OutputShapePre[1] << "," << m_OutputShapePre[2] << "," << m_OutputShapePre[3] << endl;
	Ort::RunOptions run_options;
	m_pSessionPre->Run(run_options, m_inputNamesPre, m_pInputTensorPre, 1, m_outputNamesPre, &m_vecInputTensorsSam[camID][0], 1);
	/*FILE* fp;
	fopen_s(&fp, "test.raw", "wb");
	fwrite(m_pPreprocessing, 1048576, sizeof(unsigned char), fp);
	fclose(fp);*/
	
}

void SJSegmentAnythingCPU::GetMask(const std::vector<cv::Point>& points, const std::vector<cv::Point>& negativePoints, const cv::Rect& roi, unsigned char* mask, int width, int height, int camID)
{
	float factor = (float)width / (float)m_InputShapePre[3];
	printf("%f\n", factor);
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

	m_vecInputTensorsSam[camID][1] = Ort::Value::CreateTensor<float>(*m_pMemoryInfo, inputPointValues.data(), 2 * numPoints, inputPointShape.data(), inputPointShape.size());
	m_vecInputTensorsSam[camID][2] = Ort::Value::CreateTensor<float>(*m_pMemoryInfo, inputLabelValues.data(), numPoints, pointLabelShape.data(), pointLabelShape.size());

	int outputNumber = 3, outputMaskIndex = 0, outputIOUIndex = 1;
	Ort::RunOptions runOptionsSam;


	m_pSessionSam->Run(runOptionsSam, m_inputNamesSam, m_vecInputTensorsSam[camID].data(), m_vecInputTensorsSam[camID].size(), m_outputNamesSam, m_pOutputTensorSam, 1);
	Mat outputMaskImage(m_OutputShapeSam[2], m_OutputShapeSam[3], CV_32FC1, m_pSegmentationFloat);
	resize(outputMaskImage, outputMaskImage, Size(width, height));

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			mask[j + i * width] = outputMaskImage.at<float>(i, j) > 0 ? 255 : 0;
		}
	}
}
