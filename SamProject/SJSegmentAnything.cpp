#include "SJSegmentAnything.h"

SJSegmentAnything::SJSegmentAnything()
{
	m_pSessionPre = NULL;
	m_pSessionSam = NULL;
	m_pImage = NULL;
	m_pPreprocessing = NULL;
	m_pSegmentation = NULL;
	m_pSegmentationFloat = NULL;
	m_maskInputValue = NULL;
	m_pMemoryInfo = NULL;
	m_pInputTensorPre = NULL;
	m_pOutputTensorSam = NULL;
	m_hasMaskValue = NULL;
	m_orig_im_size_values = NULL;
	m_vecInputTensorsSam = NULL;
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
	if (m_vecInputTensorsSam) {
		delete[]m_vecInputTensorsSam;
		m_vecInputTensorsSam = NULL;
	}

}

cv::Size SJSegmentAnything::GetInputSize()
{
	return cv::Size(m_InputShapePre[3], m_InputShapePre[2]);
}

void SJSegmentAnything::GetMask(const std::vector<cv::Point>& points, const std::vector<cv::Point>& negativePoints, const cv::Rect& roi, cv::Mat& outputMaskSam, double& iouValue)
{
	GetMask(points, negativePoints, roi, outputMaskSam.data, outputMaskSam.cols, outputMaskSam.rows, iouValue);

}

void SJSegmentAnything::SamLoadImage(const cv::Mat& image, int camID)
{
	SamLoadImage(image.data, image.cols, image.rows, camID);
}
