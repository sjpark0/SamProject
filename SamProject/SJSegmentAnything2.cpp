#include "SJSegmentAnything2.h"

SJSegmentAnything2::SJSegmentAnything2()
{
	m_pSessionEnc = NULL;
	m_pSessionDec = NULL;
	m_pImage = NULL;
	m_pEmbedding = NULL;
	m_pFeature1 = NULL;
	m_pFeature2 = NULL;
	m_pSegmentation = NULL;
	m_pSegmentationFloat = NULL;
	m_maskInputValue = NULL;
	m_pMemoryInfo = NULL;
	m_pInputTensorEnc = NULL;
	m_pOutputTensorDec = NULL;
	m_hasMaskValue = NULL;
	m_orig_im_size_values = NULL;
	m_vecInputTensorsDec = NULL;
	m_vecOutputTensorsEnc = NULL;
}
SJSegmentAnything2::~SJSegmentAnything2()
{

	if (m_pSessionEnc) {
		delete m_pSessionEnc;
		m_pSessionEnc = NULL;
	}
	if (m_pSessionDec) {
		delete m_pSessionDec;
		m_pSessionDec = NULL;
	}
	if (m_pMemoryInfo) {
		delete m_pMemoryInfo;
		m_pMemoryInfo = NULL;
	}
	if (m_pInputTensorEnc) {
		delete m_pInputTensorEnc;
		m_pInputTensorEnc = NULL;
	}
	if (m_pOutputTensorDec) {
		delete m_pOutputTensorDec;
		m_pOutputTensorDec = NULL;
	}
	if (m_vecInputTensorsDec) {
		delete[]m_vecInputTensorsDec;
		m_vecInputTensorsDec = NULL;
	}
	if (m_vecOutputTensorsEnc) {
		delete[]m_vecOutputTensorsEnc;
		m_vecOutputTensorsEnc = NULL;
	}

}

cv::Size SJSegmentAnything2::GetInputSize()
{
	return cv::Size(m_InputShapeEnc[3], m_InputShapeEnc[2]);
}

void SJSegmentAnything2::GetMask(const std::vector<cv::Point>& points, const std::vector<cv::Point>& negativePoints, const cv::Rect& roi, cv::Mat& outputMaskSam, double& iouValue)
{
	GetMask(points, negativePoints, roi, outputMaskSam.data, outputMaskSam.cols, outputMaskSam.rows, iouValue);

}

void SJSegmentAnything2::SamLoadImage(const cv::Mat& image, int camID)
{
	SamLoadImage(image.data, image.cols, image.rows, camID);
}
