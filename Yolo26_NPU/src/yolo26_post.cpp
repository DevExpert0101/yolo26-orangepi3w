#include "yolo26_npu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

static const char *kCoco[YOLO26_CLASS_NUM] = {
    "person",        "bicycle",       "car",           "motorcycle",    "airplane",
    "bus",           "train",         "truck",         "boat",          "traffic light",
    "fire hydrant",  "stop sign",     "parking meter", "bench",         "bird",
    "cat",           "dog",           "horse",         "sheep",         "cow",
    "elephant",      "bear",          "zebra",         "giraffe",       "backpack",
    "umbrella",      "handbag",       "tie",           "suitcase",      "frisbee",
    "skis",          "snowboard",     "sports ball",   "kite",          "baseball bat",
    "baseball glove","skateboard",    "surfboard",     "tennis racket", "bottle",
    "wine glass",    "cup",           "fork",          "knife",         "spoon",
    "bowl",          "banana",        "apple",         "sandwich",      "orange",
    "broccoli",      "carrot",        "hot dog",       "pizza",         "donut",
    "cake",          "chair",         "couch",         "potted plant",  "bed",
    "dining table",  "toilet",        "tv",            "laptop",        "mouse",
    "remote",        "keyboard",      "cell phone",    "microwave",     "oven",
    "toaster",       "sink",          "refrigerator",  "book",          "clock",
    "vase",          "scissors",      "teddy bear",    "hair drier",    "toothbrush"};

const char *coco_name(int id) {
    if (id < 0 || id >= YOLO26_CLASS_NUM) {
        return "?";
    }
    return kCoco[id];
}

LetterboxInfo letterbox(const cv::Mat &src, cv::Mat &dst, int imgsz) {
    LetterboxInfo info;
    info.imgsz = imgsz;
    info.ratio = std::min(static_cast<float>(imgsz) / src.rows, static_cast<float>(imgsz) / src.cols);
    const int new_w = static_cast<int>(std::round(src.cols * info.ratio));
    const int new_h = static_cast<int>(std::round(src.rows * info.ratio));
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));
    const int dw = imgsz - new_w;
    const int dh = imgsz - new_h;
    info.pad_x = dw / 2.f;
    info.pad_y = dh / 2.f;
    cv::copyMakeBorder(resized, dst,
                       static_cast<int>(std::round(info.pad_y - 0.1f)),
                       static_cast<int>(std::round(info.pad_y + 0.1f)),
                       static_cast<int>(std::round(info.pad_x - 0.1f)),
                       static_cast<int>(std::round(info.pad_x + 0.1f)),
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    if (dst.cols != imgsz || dst.rows != imgsz) {
        cv::resize(dst, dst, cv::Size(imgsz, imgsz));
    }
    return info;
}

void pack_nchw_uint8(const cv::Mat &rgb, std::vector<uint8_t> &dst) {
    const int h = rgb.rows;
    const int w = rgb.cols;
    dst.resize(static_cast<size_t>(3 * h * w));
    std::vector<cv::Mat> ch(3);
    cv::split(rgb, ch);
    std::memcpy(dst.data(), ch[0].data, static_cast<size_t>(h * w));
    std::memcpy(dst.data() + h * w, ch[1].data, static_cast<size_t>(h * w));
    std::memcpy(dst.data() + 2 * h * w, ch[2].data, static_cast<size_t>(h * w));
}

static float iou(const cv::Rect2f &a, const cv::Rect2f &b) {
    const float inter = (a & b).area();
    const float uni = a.area() + b.area() - inter;
    return uni <= 0.f ? 0.f : inter / uni;
}

static void nms(std::vector<Detection> &dets, float thr) {
    std::sort(dets.begin(), dets.end(), [](const Detection &a, const Detection &b) { return a.score > b.score; });
    std::vector<Detection> kept;
    kept.reserve(dets.size());
    std::vector<char> dead(dets.size(), 0);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (dead[i]) {
            continue;
        }
        kept.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!dead[j] && dets[i].class_id == dets[j].class_id && iou(dets[i].box, dets[j].box) > thr) {
                dead[j] = 1;
            }
        }
    }
    dets.swap(kept);
}

static bool match_head(size_t elems, int hw, int nc, bool &is_box) {
    if (elems == static_cast<size_t>(4 * hw)) {
        is_box = true;
        return true;
    }
    if (elems == static_cast<size_t>(nc * hw)) {
        is_box = false;
        return true;
    }
    return false;
}

void decode_yolo26_6(const VipEngine &engine, const LetterboxInfo &lb, const cv::Size &orig,
                     float conf, float nms_thr, int nc, std::vector<Detection> &dets) {
    dets.clear();
    const int strides[3] = {8, 16, 32};
    const auto &outs = engine.outputs();
    if (outs.size() < 6) {
        std::fprintf(stderr, "expected 6 YOLO26 heads, got %zu\n", outs.size());
        return;
    }

    for (int s = 0; s < 3; ++s) {
        const int stride = strides[s];
        const int h = lb.imgsz / stride;
        const int w = h;
        const int hw = h * w;
        const float *box = nullptr;
        const float *cls = nullptr;
        for (size_t i = 0; i < outs.size(); ++i) {
            bool is_box = false;
            if (!match_head(engine.output_elements(static_cast<int>(i)), hw, nc, is_box)) {
                continue;
            }
            if (is_box && !box) {
                box = engine.output_f32(static_cast<int>(i));
            } else if (!is_box && !cls) {
                cls = engine.output_f32(static_cast<int>(i));
            }
        }
        if (!box || !cls) {
            std::fprintf(stderr, "missing head for stride %d\n", stride);
            continue;
        }

        // A733 VIPLite writes CHW even when query reports HWC.
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int cell = y * w + x;
                int best_c = 0;
                float best = -1e9f;
                for (int c = 0; c < nc; ++c) {
                    const float logit = cls[c * hw + cell];
                    if (logit > best) {
                        best = logit;
                        best_c = c;
                    }
                }
                const float score = 1.f / (1.f + std::exp(-best));
                if (score < conf) {
                    continue;
                }
                const float l = box[0 * hw + cell];
                const float t = box[1 * hw + cell];
                const float r = box[2 * hw + cell];
                const float b = box[3 * hw + cell];
                const float ax = (x + 0.5f);
                const float ay = (y + 0.5f);
                float x1 = (ax - l) * stride;
                float y1 = (ay - t) * stride;
                float x2 = (ax + r) * stride;
                float y2 = (ay + b) * stride;
                x1 = (x1 - lb.pad_x) / lb.ratio;
                y1 = (y1 - lb.pad_y) / lb.ratio;
                x2 = (x2 - lb.pad_x) / lb.ratio;
                y2 = (y2 - lb.pad_y) / lb.ratio;
                x1 = std::clamp(x1, 0.f, static_cast<float>(orig.width - 1));
                y1 = std::clamp(y1, 0.f, static_cast<float>(orig.height - 1));
                x2 = std::clamp(x2, 0.f, static_cast<float>(orig.width - 1));
                y2 = std::clamp(y2, 0.f, static_cast<float>(orig.height - 1));
                if (x2 <= x1 || y2 <= y1) {
                    continue;
                }
                dets.push_back({cv::Rect2f(x1, y1, x2 - x1, y2 - y1), score, best_c});
            }
        }
    }
    nms(dets, nms_thr);
}

void draw_detections(cv::Mat &image, const std::vector<Detection> &dets) {
    for (const auto &d : dets) {
        const cv::Rect box(d.box);
        cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);
        char label[64];
        std::snprintf(label, sizeof(label), "%s %.2f", coco_name(d.class_id), d.score);
        const int y = std::max(0, box.y - 4);
        cv::putText(image, label, cv::Point(box.x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
    }
}
