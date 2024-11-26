#include "SJSegmentAnythingGPU.h"
#include <opencv2/opencv.hpp>
#include <codecvt>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace cv;
using namespace std;

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

__global__ void convertSamBuffer(const unsigned char* src, unsigned char* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight)
{
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;
	float resizeFactor = (float)srcWidth / (float)dstWidth;
	float sx = (float)ix * resizeFactor;
	float sy = (float)iy * resizeFactor;
	int sx1 = (int)sx;
	int sy1 = (int)sy;
	float w1, w2, w3, w4;
	float sum[3];
	int index1, index2;
	if (ix < dstWidth && iy < dstHeight) {
		index1 = ix + iy * dstWidth;
		if (sx1 >= 0 && sx1 + 1 < srcWidth && sy1 >= 0 && sy1 + 1 < srcHeight) {
			index2 = sx1 + sy1 * srcWidth;
			w1 = ((float)sx1 + 1 - sx) * ((float)sy1 + 1 - sy);
			w2 = (sx - (float)sx1) * ((float)sy1 + 1 - sy);
			w3 = ((float)sx1 + 1 - sx) * (sy - (float)sy1);
			w4 = (sx - (float)sx1) * (sy - (float)sy1);

			sum[0] = w1 * src[index2 * 3] + w2 * src[(index2 + 1) * 3] + w3 * src[(index2 + srcWidth) * 3] + w4 * src[(index2 + 1 + srcWidth) * 3];
			sum[1] = w1 * src[index2 * 3 + 1] + w2 * src[(index2 + 1) * 3 + 1] + w3 * src[(index2 + srcWidth) * 3 + 1] + w4 * src[(index2 + 1 + srcWidth) * 3 + 1];
			sum[2] = w1 * src[index2 * 3 + 2] + w2 * src[(index2 + 1) * 3 + 2] + w3 * src[(index2 + srcWidth) * 3 + 2] + w4 * src[(index2 + 1 + srcWidth) * 3 + 2];

			dst[index1] = (unsigned char)MIN(MAX(sum[2], 0), 255);
			dst[index1 + dstWidth * dstHeight] = (unsigned char)MIN(MAX(sum[1], 0), 255);
			dst[index1 + 2 * dstWidth * dstHeight] = (unsigned char)MIN(MAX(sum[0], 0), 255);

		}
	}
}

__global__ void convertMaskBuffer(const float* src, unsigned char* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight)
{
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;
	float resizeFactor = (float)srcWidth / (float)dstWidth;
	float sx = (float)ix * resizeFactor;
	float sy = (float)iy * resizeFactor;
	int sx1 = (int)sx;
	int sy1 = (int)sy;
	float w1, w2, w3, w4;
	float sum;
	int index1, index2;
	if (ix < dstWidth && iy < dstHeight) {
		index1 = ix + iy * dstWidth;
		if (sx1 >= 0 && sx1 + 1 < srcWidth && sy1 >= 0 && sy1 + 1 < srcHeight) {
			index2 = sx1 + sy1 * srcWidth;
			w1 = ((float)sx1 + 1 - sx) * ((float)sy1 + 1 - sy);
			w2 = (sx - (float)sx1) * ((float)sy1 + 1 - sy);
			w3 = ((float)sx1 + 1 - sx) * (sy - (float)sy1);
			w4 = (sx - (float)sx1) * (sy - (float)sy1);

			sum = w1 * src[index2] + w2 * src[(index2 + 1)] + w3 * src[(index2 + srcWidth)] + w4 * src[(index2 + 1 + srcWidth)];
			dst[index1] = sum > 0 ? 255 : 0;
		}
	}
}


using namespace cv;
using namespace std;
size_t SJSegmentAnythingGPU::iDivUp(size_t a, size_t b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

SJSegmentAnythingGPU::SJSegmentAnythingGPU()
{
	m_pImageCUDA = NULL;
}
SJSegmentAnythingGPU::~SJSegmentAnythingGPU()
{
	if (m_pImage) {
		cudaFree(m_pImage);
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
	if (m_hasMaskValue) {
		cudaFree(m_hasMaskValue);
		m_hasMaskValue = NULL;
	}
	if (m_orig_im_size_values) {
		delete[]m_orig_im_size_values;
		m_orig_im_size_values = NULL;
	}
}

int SJSegmentAnythingGPU::InitializeSamModel(const char* preModelPath, const char* samModelPath, int numCam, int width, int height)
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

	cudaMalloc((void**)&m_pImage, m_InputShapePre[0] * m_InputShapePre[1] * m_InputShapePre[2] * m_InputShapePre[3] * sizeof(unsigned char));
	cudaMalloc((void**)&m_pImageCUDA, width * height * 3 * sizeof(unsigned char));

	cudaMalloc((void**)&m_pPreprocessing, numCam * m_OutputShapePre[0] * m_OutputShapePre[1] * m_OutputShapePre[2] * m_OutputShapePre[3] * sizeof(float));

	cudaMalloc((void**)&m_pSegmentationFloat, m_OutputShapeSam[0] * m_OutputShapeSam[1] * m_OutputShapeSam[2] * m_OutputShapeSam[3] * sizeof(float));
	cudaMalloc((void**)&m_pSegmentation, width * height * sizeof(unsigned char));

	//m_maskInputValue = new float[256 * 256];
	cudaMalloc((void**)&m_maskInputValue, 256 * 256 * sizeof(float));
	cudaMemset(m_maskInputValue, 0, 256 * 256 * sizeof(float));

	float hasMaskValue = 0;
	cudaMalloc((void**)&m_hasMaskValue, sizeof(float));
	cudaMemcpy(m_hasMaskValue, &hasMaskValue, sizeof(float), cudaMemcpyHostToDevice);

	float orig_im_size_value[] = { (float)m_InputShapePre[2] , (float)m_InputShapePre[3] };

	cudaMalloc((void**)&m_orig_im_size_values, 2 * sizeof(float));
	cudaMemcpy(m_orig_im_size_values, orig_im_size_value, 2 * sizeof(float), cudaMemcpyHostToDevice);


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
	m_cudaThread.x = BLOCKDIM_X;
	m_cudaThread.y = BLOCKDIM_Y;
	m_cudaThread.z = 1;
	m_cudaGrid.x = iDivUp(m_InputShapePre[3], BLOCKDIM_X);
	m_cudaGrid.y = iDivUp(m_InputShapePre[2], BLOCKDIM_Y);
	m_cudaGrid.z = 1;
}

void SJSegmentAnythingGPU::SamLoadImage(const unsigned char* imageBuffer, int width, int height, int camID)
{
	cudaMemcpy(m_pImageCUDA, &imageBuffer[camID * width * height * 3], width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	convertSamBuffer << <m_cudaGrid, m_cudaThread >> > (m_pImageCUDA, m_pImage, width, height, m_InputShapePre[3], m_InputShapePre[2]);
	Ort::RunOptions run_options;
	m_pSessionPre->Run(run_options, m_inputNamesPre, m_pInputTensorPre, 1, m_outputNamesPre, &m_vecInputTensorsSam[camID][0], 1);
}

void SJSegmentAnythingGPU::GetMask(const std::vector<cv::Point>& points, const std::vector<cv::Point>& negativePoints, const cv::Rect& roi, unsigned char* mask, int width, int height, int camID)
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
	if (!roi.empty()) {
		inputPointValues.push_back((float)roi.x / factor);
		inputPointValues.push_back((float)roi.y / factor);
		inputLabelValues.push_back(2);
		inputPointValues.push_back((float)roi.br().x / factor);
		inputPointValues.push_back((float)roi.br().y / factor);
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

	m_vecInputTensorsSam[camID][1] = Ort::Value::CreateTensor<float>(*m_pMemoryInfo, inputPointValuesCUDA, 2 * numPoints, inputPointShape.data(), inputPointShape.size());
	m_vecInputTensorsSam[camID][2] = Ort::Value::CreateTensor<float>(*m_pMemoryInfo, labelPointValuesCUDA, numPoints, pointLabelShape.data(), pointLabelShape.size());

	int outputNumber = 3, outputMaskIndex = 0, outputIOUIndex = 1;
	Ort::RunOptions runOptionsSam;


	m_pSessionSam->Run(runOptionsSam, m_inputNamesSam, m_vecInputTensorsSam[camID].data(), m_vecInputTensorsSam[camID].size(), m_outputNamesSam, m_pOutputTensorSam, 1);

	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(width, BLOCKDIM_X), iDivUp(height, BLOCKDIM_Y));

	convertMaskBuffer << <grid, threads >> > (m_pSegmentationFloat, m_pSegmentation, m_OutputShapeSam[3], m_OutputShapeSam[2], width, height);
	cudaMemcpy(mask, m_pSegmentation, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);


	cudaFree(inputPointValuesCUDA);
	cudaFree(labelPointValuesCUDA);
}
