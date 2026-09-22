#pragma once

#include "vip_lite.h"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#define YOLO26_NMS_THRESH 0.45f
#define YOLO26_CONF_THRESH 0.25f
#define YOLO26_CLASS_NUM 80
#define YOLO26_MAX_IO 8

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

struct TensorInfo {
    std::string name;
    int         format = 0;
    int         quant_format = 0;
    float       scale = 1.f;
    int         zero_point = 0;
    int         fl = 0;
    int         dims = 0;
    uint32_t    sizes[VIP_MAX_DIM_NUM] = {};
    size_t      elements = 0;
    size_t      bytes = 0;
};

class VipEngine {
public:
    ~VipEngine();
    bool load(const std::string &nbg_path);
    void pack_rgb(const cv::Mat &rgb, std::vector<uint8_t> &dst) const;
    bool infer(const uint8_t *nchw_or_packed, size_t bytes);
    const std::vector<TensorInfo> &inputs() const { return inputs_; }
    const std::vector<TensorInfo> &outputs() const { return outputs_; }
    const float *output_f32(int index) const;
    size_t output_elements(int index) const;
    void close();

private:
    bool query_io();
    bool create_buffers();
    size_t format_bytes(int format) const;

    vip_network network_ = nullptr;
    vip_buffer  in_buf_[YOLO26_MAX_IO] = {};
    vip_buffer  out_buf_[YOLO26_MAX_IO] = {};
    std::vector<TensorInfo> inputs_;
    std::vector<TensorInfo> outputs_;
    std::vector<std::vector<float>> host_out_;
    bool ready_ = false;
};

LetterboxInfo letterbox(const cv::Mat &src, cv::Mat &dst, int imgsz);
void pack_nchw_uint8(const cv::Mat &rgb, std::vector<uint8_t> &dst);
void decode_yolo26_6(const VipEngine &engine, const LetterboxInfo &lb, const cv::Size &orig,
                     float conf, float nms, int nc, std::vector<Detection> &dets,
                     const char *layout = "chw");
void draw_detections(cv::Mat &image, const std::vector<Detection> &dets);
const char *coco_name(int id);
