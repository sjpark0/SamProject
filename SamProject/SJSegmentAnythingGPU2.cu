#include "SJSegmentAnythingGPU2.h"
#include <opencv2/opencv.hpp>
#include <codecvt>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace cv;
using namespace std;

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

__global__ void convertSamBuffer2(const unsigned char* src, float* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight)
{
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;
	float resizeFactorX = (float)srcWidth / (float)dstWidth;
	float resizeFactorY = (float)srcHeight / (float)dstHeight;

	float sx = (float)ix * resizeFactorX;
	float sy = (float)iy * resizeFactorY;
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

			dst[index1] = (MIN(MAX(sum[2], 0), 255) / 255.0 - 0.485) / 0.229;
			dst[index1 + dstWidth * dstHeight] = (MIN(MAX(sum[1], 0), 255) / 255.0 - 0.456) / 0.224;
			dst[index1 + 2 * dstWidth * dstHeight] = (MIN(MAX(sum[0], 0), 255) / 255.0 - 0.406) / 0.225;

		}
	}
}

__global__ void convertMaskBuffer2(const float* src, unsigned char* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight)
{
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;
	float resizeFactorX = (float)srcWidth / (float)dstWidth;
	float resizeFactorY = (float)srcHeight / (float)dstHeight;
	float sx = (float)ix * resizeFactorX;
	float sy = (float)iy * resizeFactorY;
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
size_t SJSegmentAnythingGPU2::iDivUp(size_t a, size_t b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

SJSegmentAnythingGPU2::SJSegmentAnythingGPU2()
{
	m_pImageCUDA = NULL;
}
SJSegmentAnythingGPU2::~SJSegmentAnythingGPU2()
{
	if (m_pImage) {
		cudaFree(m_pImage);
		m_pImage = NULL;
	}
	if (m_pImageCUDA) {
		cudaFree(m_pImageCUDA);
		m_pImageCUDA = NULL;
	}
	if (m_pEmbedding) {
		cudaFree(m_pEmbedding);
		m_pEmbedding = NULL;
	}
	if (m_pFeature1) {
		cudaFree(m_pFeature1);
		m_pFeature1 = NULL;
	}
	if (m_pFeature2) {
		cudaFree(m_pFeature2);
		m_pFeature2 = NULL;
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
		cudaFree(m_orig_im_size_values);
		m_orig_im_size_values = NULL;
	}
}

int SJSegmentAnythingGPU2::InitializeSamModel(const char* preModelPath, const char* samModelPath, int numCam, int width, int height)
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

	m_pSessionEnc = new Ort::Session(m_env, wpreModelPath.c_str(), m_pSessionOptions[0]);
	m_pSessionDec = new Ort::Session(m_env, wsamModelPath.c_str(), m_pSessionOptions[1]);

	m_InputShapeEnc = m_pSessionEnc->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	
	m_OutputShapeDec = m_InputShapeEnc;
	m_OutputShapeDec[1] = 1;

	cudaMalloc((void**)&m_pImage, m_InputShapeEnc[0] * m_InputShapeEnc[1] * m_InputShapeEnc[2] * m_InputShapeEnc[3] * sizeof(float));
	cudaMalloc((void**)&m_pImageCUDA, width * height * 3 * sizeof(unsigned char));

	std::vector<int64_t> outputShapeEnc0 = m_pSessionDec->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
	std::vector<int64_t> outputShapeEnc1 = m_pSessionDec->GetInputTypeInfo(2).GetTensorTypeAndShapeInfo().GetShape();
	std::vector<int64_t> outputShapeEnc2 = m_pSessionDec->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

	cudaMalloc((void**)&m_pFeature1, numCam * outputShapeEnc0[0] * outputShapeEnc0[1] * outputShapeEnc0[2] * outputShapeEnc0[3] * sizeof(float));
	cudaMalloc((void**)&m_pFeature2, numCam * outputShapeEnc1[0] * outputShapeEnc1[1] * outputShapeEnc1[2] * outputShapeEnc1[3] * sizeof(float));
	cudaMalloc((void**)&m_pEmbedding, numCam * outputShapeEnc2[0] * outputShapeEnc2[1] * outputShapeEnc2[2] * outputShapeEnc2[3] * sizeof(float));

	
	cudaMalloc((void**)&m_pSegmentationFloat, m_OutputShapeDec[0] * m_OutputShapeDec[1] * m_OutputShapeDec[2] * m_OutputShapeDec[3] * sizeof(float));
	cudaMalloc((void**)&m_pSegmentation, width * height * sizeof(unsigned char));

	//m_maskInputValue = new float[256 * 256];
	cudaMalloc((void**)&m_maskInputValue, 256 * 256 * sizeof(float));
	cudaMemset(m_maskInputValue, 0, 256 * 256 * sizeof(float));

	float hasMaskValue = 0;
	cudaMalloc((void**)&m_hasMaskValue, sizeof(float));
	cudaMemcpy(m_hasMaskValue, &hasMaskValue, sizeof(float), cudaMemcpyHostToDevice);

	int orig_im_size_value[] = { m_InputShapeEnc[2] , m_InputShapeEnc[3] };

	cudaMalloc((void**)&m_orig_im_size_values, 2 * sizeof(int));
	cudaMemcpy(m_orig_im_size_values, orig_im_size_value, 2 * sizeof(int), cudaMemcpyHostToDevice);


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

	m_cudaThread.x = BLOCKDIM_X;
	m_cudaThread.y = BLOCKDIM_Y;
	m_cudaThread.z = 1;
	m_cudaGrid.x = iDivUp(m_InputShapeEnc[3], BLOCKDIM_X);
	m_cudaGrid.y = iDivUp(m_InputShapeEnc[2], BLOCKDIM_Y);
	m_cudaGrid.z = 1;
}

void SJSegmentAnythingGPU2::SamLoadImage(const unsigned char* imageBuffer, int width, int height, int camID)
{
	cudaMemcpy(m_pImageCUDA, &imageBuffer[camID * width * height * 3], width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	convertSamBuffer2 << <m_cudaGrid, m_cudaThread >> > (m_pImageCUDA, m_pImage, width, height, m_InputShapeEnc[3], m_InputShapeEnc[2]);
	Ort::RunOptions run_options;
	m_pSessionEnc->Run(run_options, m_inputNamesEnc, m_pInputTensorEnc, 1, m_outputNamesEnc, m_vecOutputTensorsEnc[camID].data(), m_vecOutputTensorsEnc[camID].size());
}

void SJSegmentAnythingGPU2::GetMask(const std::vector<cv::Point>& points, const std::vector<cv::Point>& negativePoints, const cv::Rect& roi, unsigned char* mask, int width, int height, int camID)
{
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

	float* inputPointValuesCUDA;
	float* labelPointValuesCUDA;
	cudaMalloc((void**)&inputPointValuesCUDA, numPoints * 2 * sizeof(float));
	cudaMemcpy(inputPointValuesCUDA, inputPointValues.data(), numPoints * 2 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&labelPointValuesCUDA, numPoints * sizeof(float));
	cudaMemcpy(labelPointValuesCUDA, inputLabelValues.data(), numPoints * sizeof(float), cudaMemcpyHostToDevice);


	float valIOU[] = { 0 };
	std::vector<int64_t> inputPointShape = { 1, numPoints, 2 };
	std::vector<int64_t> pointLabelShape = { 1, numPoints };

	m_vecInputTensorsDec[camID][3] = Ort::Value::CreateTensor<float>(*m_pMemoryInfo, inputPointValuesCUDA, 2 * numPoints, inputPointShape.data(), inputPointShape.size());
	m_vecInputTensorsDec[camID][4] = Ort::Value::CreateTensor<float>(*m_pMemoryInfo, labelPointValuesCUDA, numPoints, pointLabelShape.data(), pointLabelShape.size());

	Ort::RunOptions runOptionsSam;
	m_pSessionDec->Run(runOptionsSam, m_inputNamesDec, m_vecInputTensorsDec[camID].data(), m_vecInputTensorsDec[camID].size(), m_outputNamesDec, m_pOutputTensorDec, 1);

	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(width, BLOCKDIM_X), iDivUp(height, BLOCKDIM_Y));

	convertMaskBuffer2 << <grid, threads >> > (m_pSegmentationFloat, m_pSegmentation, m_OutputShapeDec[3], m_OutputShapeDec[2], width, height);
	cudaMemcpy(mask, m_pSegmentation, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);


	cudaFree(inputPointValuesCUDA);
	cudaFree(labelPointValuesCUDA);
}
