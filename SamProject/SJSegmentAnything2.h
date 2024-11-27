#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

class SJSegmentAnything2
{
protected:
	Ort::Env m_env;
	Ort::SessionOptions m_pSessionOptions[2];
	Ort::Session* m_pSessionEnc;
	Ort::Session* m_pSessionDec;

	std::vector<int64_t> m_InputShapeEnc;
	//std::vector<int64_t> m_OutputShapeEnc;
	std::vector<int64_t> m_OutputShapeDec;

	Ort::MemoryInfo* m_pMemoryInfo;

	const char* m_inputNamesEnc[1] = { "image" };
	const char* m_outputNamesEnc[3] = { "high_res_feats_0", "high_res_feats_1", "image_embed" };

	const char* m_inputNamesDec[8]{ "image_embed", "high_res_feats_0", "high_res_feats_1", "point_coords",   "point_labels", "mask_input", "has_mask_input", "orig_im_size" };
	const char* m_outputNamesDec[2]{ "masks", "iou_predictions" };

	Ort::Value* m_pInputTensorEnc;
	Ort::Value* m_pOutputTensorDec;

	std::vector<Ort::Value>* m_vecInputTensorsDec;
	std::vector<Ort::Value>* m_vecOutputTensorsEnc;

	std::vector<int64_t> m_vecMaskInputShape = { 1, 1, 256, 256 };

	std::vector<int64_t> m_vecHasMaskInputShape = { 1 };

	std::vector<int64_t> m_vecOrigImSizeShape = { 2 };

	
	float* m_pImage;
	float* m_pEmbedding;
	float* m_pFeature1;
	float* m_pFeature2;

	unsigned char* m_pSegmentation;
	float* m_pSegmentationFloat;
	float* m_maskInputValue;
	float* m_hasMaskValue;
	int* m_orig_im_size_values;
	
public:
	SJSegmentAnything2();
	virtual ~SJSegmentAnything2();

	cv::Size GetInputSize();

	virtual int InitializeSamModel(const char* encModelPath, const char* decModelPath, int numCam, int width, int height) = 0;
	virtual void SamLoadImage(const unsigned char* imageBuffer, int width, int hieght, int camID) = 0;
	void SamLoadImage(const cv::Mat& image, int camID);

	virtual void GetMask(const std::vector<cv::Point>& points, const std::vector<cv::Point>& negativePoints, const cv::Rect& roi, unsigned char* mask, int width, int height, int camID) = 0;
	void GetMask(const std::vector<cv::Point>& points, const std::vector<cv::Point>& negativePoints, const cv::Rect& roi, cv::Mat& outputMaskSam, double& iouValue);

};

