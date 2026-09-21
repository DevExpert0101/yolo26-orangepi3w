#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <cstdint>
#include <string>
#include <vector>

#define YOLO26_NMS_THRESH 0.45f
#define YOLO26_CONF_THRESH 0.25f
#define YOLO26_CLASS_NUM 80

struct LetterboxInfo {
    float ratio = 1.f;
    float pad_x = 0.f;
    float pad_y = 0.f;
    int   imgsz = 640;
};

struct Detection {
    cv::Rect2f box;
    float      score = 0.f;
    int        class_id = -1;
};

class OnnxEngine {
public:
    bool load(const std::string &onnx_path, int imgsz, int threads);
    bool infer(const float *nchw, size_t elements);
    size_t output_count() const { return host_out_.size(); }
    const float *output_f32(int index = 0) const;
    size_t output_elements(int index = 0) const;
    const std::vector<int64_t> &output_shape(int index = 0) const;
    const std::vector<int64_t> &input_shape() const { return in_shape_; }
    size_t input_elements() const;
    bool nhwc() const { return nhwc_; }
    bool uint8_input() const { return uint8_input_; }
    int imgsz() const { return imgsz_; }

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "yolo26"};
    Ort::SessionOptions opts_;
    Ort::Session session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string in_name_;
    std::vector<std::string> out_names_;
    std::vector<int64_t> in_shape_;
    std::vector<std::vector<int64_t>> out_shapes_;
    std::vector<std::vector<float>> host_out_;
    int imgsz_ = 640;
    bool nhwc_ = false;
    bool uint8_input_ = false;
    bool ready_ = false;
};

LetterboxInfo letterbox(const cv::Mat &src, cv::Mat &dst, int imgsz);
void pack_nchw_f32(const cv::Mat &rgb, std::vector<float> &dst);
void pack_nhwc_f32(const cv::Mat &rgb, std::vector<float> &dst);
void decode_yolo26_onnx(const OnnxEngine &engine, const LetterboxInfo &lb, const cv::Size &orig,
                        float conf, float nms, int nc, std::vector<Detection> &dets);
void draw_detections(cv::Mat &image, const std::vector<Detection> &dets);
const char *coco_name(int id);
