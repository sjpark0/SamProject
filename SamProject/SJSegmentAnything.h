#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

class SJSegmentAnything
{
protected:
	Ort::Env m_env;
	Ort::SessionOptions m_pSessionOptions[2];
	Ort::Session* m_pSessionPre;
	Ort::Session* m_pSessionSam;

	std::vector<int64_t> m_InputShapePre;
	std::vector<int64_t> m_OutputShapePre;
	std::vector<int64_t> m_OutputShapeSam;

	Ort::MemoryInfo* m_pMemoryInfo;

	const char* m_inputNamesPre[1] = { "input" };
	const char* m_outputNamesPre[1] = { "output" };

	const char* m_inputNamesSam[6]{ "image_embeddings", "point_coords",   "point_labels", "mask_input", "has_mask_input", "orig_im_size" };
	const char* m_outputNamesSam[3]{ "masks", "iou_predictions", "low_res_masks" };

	Ort::Value* m_pInputTensorPre;
	Ort::Value* m_pOutputTensorSam;

	std::vector<Ort::Value>* m_vecInputTensorsSam;
	std::vector<int64_t> m_vecMaskInputShape = { 1, 1, 256, 256 };

	std::vector<int64_t> m_vecHasMaskInputShape = { 1 };

	std::vector<int64_t> m_vecOrigImSizeShape = { 2 };

	unsigned char* m_pImage;
	float* m_pPreprocessing;
	unsigned char* m_pSegmentation;
	float* m_pSegmentationFloat;
	float* m_maskInputValue;
	float* m_hasMaskValue;
	float* m_orig_im_size_values;
	
public:
	SJSegmentAnything();	
	virtual ~SJSegmentAnything();

	cv::Size GetInputSize();

	virtual int InitializeSamModel(const char* preModelPath, const char* samModelPath, int numCam, int width, int height) = 0;
	virtual void SamLoadImage(const unsigned char* imageBuffer, int width, int hieght, int camID) = 0;
	void SamLoadImage(const cv::Mat& image, int camID);

	virtual void GetMask(const std::vector<cv::Point>& points, const std::vector<cv::Point>& negativePoints, const cv::Rect& roi, unsigned char* mask, int width, int height, int camID) = 0;
	void GetMask(const std::vector<cv::Point>& points, const std::vector<cv::Point>& negativePoints, const cv::Rect& roi, cv::Mat& outputMaskSam, double& iouValue);

};

