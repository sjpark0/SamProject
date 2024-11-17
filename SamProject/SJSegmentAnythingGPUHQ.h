#pragma once
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <list>

#include <opencv2/opencv.hpp>
class SJSegmentAnythingGPUHQ
{
private:
	Ort::Env m_env;
	Ort::SessionOptions m_pSessionOptions[2];
	Ort::Session* m_pSessionPre;
	Ort::Session* m_pSessionSam;

	std::vector<int64_t> m_InputShapePre;
	std::vector<int64_t> m_OutputShapePre;
	std::vector<int64_t> m_OutputShapeSam;
	std::vector<int64_t> m_IntermShapePre;

	Ort::MemoryInfo* m_pMemoryInfo;
	
	unsigned char* m_pImage;
	unsigned char* m_pImageCUDA;
	float* m_pPreprocessing;
	float* m_pInterm;
	float* m_pSegmentation;
	float* m_pSegmentationCUDA;

	float* m_maskInputValue;
	const char* m_inputNamesPre[1] = { "input" };
	const char* m_outputNamesPre[2] = { "output", "interm_embeddings" };

	//const char* m_inputNamesSam[6]{ "image_embeddings", "point_coords",   "point_labels", "mask_input", "has_mask_input", "orig_im_size" };
	const char* m_inputNamesSamHQ[7]{ "image_embeddings", "interm_embeddings", "point_coords", "point_labels", "mask_input",       "has_mask_input",    "orig_im_size" };
	//const char* inputNamesEdgeSam[3]{ "image_embeddings", "point_coords", "point_labels" };
	const char* m_outputNamesSam[3]{ "masks", "iou_predictions", "low_res_masks" };
	//const char* outputNamesEdgeSam[2]{ "scores", "masks" };

	Ort::Value* m_pInputTensorPre;
	Ort::Value* m_pOutputTensorPre;
	Ort::Value* m_pOutputTensorSam;

public:
	SJSegmentAnythingGPUHQ();
	~SJSegmentAnythingGPUHQ();

	int InitializeSamModel(const char* preModelPath, const char* samModelPath);
	cv::Size GetInputSize();
	void SamLoadImage(const cv::Mat& image);
	void GetMask(const std::vector<cv::Point>& points, const std::vector<cv::Point>& negativePoints, const cv::Rect& roi, cv::Mat& outputMaskSam, double& iouValue);
};

