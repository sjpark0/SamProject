#pragma once
#include "SJSegmentAnything2.h"
#include <vector>
#include <list>

#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"

class SJSegmentAnythingGPU2 : public SJSegmentAnything2
{
private:
	dim3 m_cudaThread;
	dim3 m_cudaGrid;
	unsigned char* m_pImageCUDA;
	size_t iDivUp(size_t a, size_t b);
public:
	SJSegmentAnythingGPU2();
	~SJSegmentAnythingGPU2();

	virtual int InitializeSamModel(const char* encModelPath, const char* decModelPath, int numCam, int width, int height) override;
	virtual void SamLoadImage(const unsigned char* imageBuffer, int width, int hieght, int camID) override;
	virtual void GetMask(const std::vector<cv::Point>& points, const std::vector<cv::Point>& negativePoints, const cv::Rect& roi, unsigned char* mask, int width, int height, int camID) override;

};

