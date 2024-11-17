#include "SJSegmentAnythingGPU.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace cv;
using namespace std;
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
		if (sx1 >= 0 && sx1+1 < srcWidth && sy1 >= 0 && sy1+1 < srcHeight) {
			index2 = sx1 + sy1 * srcWidth;
			w1 = ((float)sx1 + 1 - sx) * ((float)sy1 + 1 - sy);
			w2 = (sx - (float)sx1) * ((float)sy1 + 1 - sy);
			w3 = ((float)sx1 + 1 - sx) * (sy - (float)sy1);
			w4 = (sx - (float)sx1) * (sy - (float)sy1);

			sum[0] = w1 * src[index2 * 3 + 2] + w2 * src[(index2 + 1) * 3 + 2] + w3 * src[(index2 + srcWidth) * 3 + 2] + w4 * src[(index2 + 1 + srcWidth) * 3 + 2];
			sum[1] = w1 * src[index2 * 3 + 1] + w2 * src[(index2 + 1) * 3 + 1] + w3 * src[(index2 + srcWidth) * 3 + 1] + w4 * src[(index2 + 1 + srcWidth) * 3 + 1];
			sum[2] = w1 * src[index2 * 3] + w2 * src[(index2 + 1) * 3] + w3 * src[(index2 + srcWidth) * 3] + w4 * src[(index2 + 1 + srcWidth) * 3];

			dst[index1] = (unsigned char)MIN(MAX(sum[0], 0), 255);
			dst[index1 + dstWidth * dstHeight] = (unsigned char)MIN(MAX(sum[1], 0), 255);
			dst[index1 + 2 * dstWidth * dstHeight] = (unsigned char)MIN(MAX(sum[2], 0), 255);

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
void SJSegmentAnythingGPU::SamLoadImage(unsigned char* imageBufferCUDA, int width, int height)
{
	convertSamBuffer << <m_cudaGrid, m_cudaThread >> > (imageBufferCUDA, m_pImageCUDA, width, height, m_InputShapePre[3], m_InputShapePre[2]);
	Ort::RunOptions run_options;
	m_pSessionPre->Run(run_options, m_inputNamesPre, m_pInputTensorPre, 1, m_outputNamesPre, &m_vecInputTensorsSam[0], 1);

}

void SJSegmentAnythingGPU::GetMask(const std::vector<cv::Point>& points, const std::vector<cv::Point>& negativePoints, const cv::Rect& roi, unsigned char* mask, int width, int height, double& iouValue)
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

	m_vecInputTensorsSam[1] = Ort::Value::CreateTensor<float>(*m_pMemoryInfo, inputPointValuesCUDA, 2 * numPoints, inputPointShape.data(), inputPointShape.size());
	m_vecInputTensorsSam[2] = Ort::Value::CreateTensor<float>(*m_pMemoryInfo, labelPointValuesCUDA, numPoints, pointLabelShape.data(), pointLabelShape.size());

	int outputNumber = 3, outputMaskIndex = 0, outputIOUIndex = 1;
	Ort::RunOptions runOptionsSam;


	m_pSessionSam->Run(runOptionsSam, m_inputNamesSam, m_vecInputTensorsSam.data(), m_vecInputTensorsSam.size(), m_outputNamesSam, m_pOutputTensorSam, 1);

	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(width, BLOCKDIM_X), iDivUp(height, BLOCKDIM_Y));

	convertMaskBuffer << <grid, threads >> > (m_pSegmentationFloat, m_pSegmentation, m_OutputShapeSam[3], m_OutputShapeSam[2], width, height);
	cudaMemcpy(mask, m_pSegmentation, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);


	cudaFree(inputPointValuesCUDA);
	cudaFree(labelPointValuesCUDA);
}
