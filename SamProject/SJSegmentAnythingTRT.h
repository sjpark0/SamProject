#pragma once
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "onnxruntime_cxx_api.h"
#include <vector>
#include <list>

#include <opencv2/opencv.hpp>


#include "logger.h"


class Logger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override;
};

class SJSegmentAnythingTRT
{
private:
	Logger m_logger;

	nvinfer1::IBuilder** m_ppBuilder;
	nvinfer1::INetworkDefinition** m_ppNetwork;
	nvinfer1::IBuilderConfig** m_ppConfig;
	nvonnxparser::IParser** m_ppParser;
	nvinfer1::IHostMemory** m_ppPlan;
	nvinfer1::IRuntime** m_ppRuntime;
	nvinfer1::ICudaEngine** m_ppEngine;
	nvinfer1::IExecutionContext** m_ppContext;
	nvinfer1::Dims m_preInputDims;
	nvinfer1::Dims m_preOutputDims;
	nvinfer1::Dims m_samInputDims;
	nvinfer1::Dims m_samOutputDims;

	unsigned char* m_pImage;
	unsigned char* m_pImageCUDA;
	float* m_pPreprocessing;
	float* m_pSegmentation;
	float* m_pSegmentationCUDA;

	float* m_maskInputValue;
	const char* m_inputNamesPre[1] = { "input" };
	const char* m_outputNamesPre[2] = { "output", "interm_embeddings" };

	const char* m_inputNamesSam[6]{ "image_embeddings", "point_coords",   "point_labels", "mask_input", "has_mask_input", "orig_im_size" };
	//const char* inputNamesSamHQ[7]{ "image_embeddings", "interm_embeddings", "point_coords", "point_labels", "mask_input",       "has_mask_input",    "orig_im_size" };
	//const char* inputNamesEdgeSam[3]{ "image_embeddings", "point_coords", "point_labels" };
	const char* m_outputNamesSam[3]{ "masks", "iou_predictions", "low_res_masks" };
	//const char* outputNamesEdgeSam[2]{ "scores", "masks" };

	Ort::Value* m_pInputTensorPre;
	Ort::Value* m_pOutputTensorPre;
	Ort::Value* m_pOutputTensorSam;

public:
	SJSegmentAnythingTRT();
	~SJSegmentAnythingTRT();

	int InitializeSamModel(const char* preModelPath, const char* samModelPath);
	cv::Size GetInputSize();
	void SamLoadImage(const cv::Mat& image);
	void GetMask(const std::vector<cv::Point>& points, const std::vector<cv::Point>& negativePoints, const cv::Rect& roi, cv::Mat& outputMaskSam, double& iouValue);
};

