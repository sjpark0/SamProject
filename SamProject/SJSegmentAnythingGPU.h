#pragma once
#include "SJSegmentAnything.h"
#include <vector>
#include <list>

#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"

class SJSegmentAnythingGPU : public SJSegmentAnything
{
private:
	dim3 m_cudaThread;
	dim3 m_cudaGrid;
	unsigned char* m_pImageCUDA;
	size_t iDivUp(size_t a, size_t b);
public:
	SJSegmentAnythingGPU();
	~SJSegmentAnythingGPU();

	virtual int InitializeSamModel(const char* preModelPath, const char* samModelPath, int numCam, int width, int height) override;
	virtual void SamLoadImage(const unsigned char* imageBuffer, int width, int hieght, int camID) override;
	virtual void GetMask(const std::vector<cv::Point>& points, const std::vector<cv::Point>& negativePoints, const cv::Rect& roi, unsigned char* mask, int width, int height, int camID) override;
};

