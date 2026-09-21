#include "yolo26_onnx.hpp"

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

void pack_nchw_f32(const cv::Mat &rgb, std::vector<float> &dst) {
    const int h = rgb.rows;
    const int w = rgb.cols;
    dst.resize(static_cast<size_t>(3 * h * w));
    std::vector<cv::Mat> ch(3);
    cv::split(rgb, ch);
    const float scale = 1.f / 255.f;
    for (int c = 0; c < 3; ++c) {
        float *out = dst.data() + static_cast<size_t>(c * h * w);
        const uint8_t *in = ch[c].data;
        const int n = h * w;
        for (int i = 0; i < n; ++i) {
            out[i] = static_cast<float>(in[i]) * scale;
        }
    }
}

void pack_nhwc_f32(const cv::Mat &rgb, std::vector<float> &dst) {
    const int h = rgb.rows;
    const int w = rgb.cols;
    dst.resize(static_cast<size_t>(3 * h * w));
    const float scale = 1.f / 255.f;
    const uint8_t *in = rgb.data;
    for (int i = 0, n = h * w * 3; i < n; ++i) {
        dst[static_cast<size_t>(i)] = static_cast<float>(in[i]) * scale;
    }
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

static void scale_box(float &x1, float &y1, float &x2, float &y2, const LetterboxInfo &lb, const cv::Size &orig) {
    x1 = (x1 - lb.pad_x) / lb.ratio;
    y1 = (y1 - lb.pad_y) / lb.ratio;
    x2 = (x2 - lb.pad_x) / lb.ratio;
    y2 = (y2 - lb.pad_y) / lb.ratio;
    x1 = std::clamp(x1, 0.f, static_cast<float>(orig.width - 1));
    y1 = std::clamp(y1, 0.f, static_cast<float>(orig.height - 1));
    x2 = std::clamp(x2, 0.f, static_cast<float>(orig.width - 1));
    y2 = std::clamp(y2, 0.f, static_cast<float>(orig.height - 1));
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

static void decode_six_heads(const OnnxEngine &engine, const LetterboxInfo &lb, const cv::Size &orig,
                             float conf, float nms_thr, int nc, std::vector<Detection> &dets) {
    dets.clear();
    const int strides[3] = {8, 16, 32};
    for (int s = 0; s < 3; ++s) {
        const int stride = strides[s];
        const int h = lb.imgsz / stride;
        const int w = h;
        const int hw = h * w;
        const float *box = nullptr;
        const float *cls = nullptr;
        for (size_t i = 0; i < engine.output_count(); ++i) {
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
            std::fprintf(stderr, "missing YOLO26 head for stride %d (imgsz=%d)\n", stride, lb.imgsz);
            continue;
        }
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
                float x1 = (static_cast<float>(x) + 0.5f - l) * static_cast<float>(stride);
                float y1 = (static_cast<float>(y) + 0.5f - t) * static_cast<float>(stride);
                float x2 = (static_cast<float>(x) + 0.5f + r) * static_cast<float>(stride);
                float y2 = (static_cast<float>(y) + 0.5f + b) * static_cast<float>(stride);
                scale_box(x1, y1, x2, y2, lb, orig);
                if (x2 <= x1 || y2 <= y1) {
                    continue;
                }
                dets.push_back({cv::Rect2f(x1, y1, x2 - x1, y2 - y1), score, best_c});
            }
        }
    }
    nms(dets, nms_thr);
}

static std::vector<int64_t> squeeze2d(const std::vector<int64_t> &shape) {
    std::vector<int64_t> dims;
    for (auto d : shape) {
        if (d != 1) {
            dims.push_back(d > 0 ? d : 1);
        }
    }
    if (dims.size() == 1) {
        dims.insert(dims.begin(), 1);
    }
    return dims;
}

static void decode_single(const float *data, const std::vector<int64_t> &shape,
                          const LetterboxInfo &lb, const cv::Size &orig, float conf, float nms_thr,
                          int nc, std::vector<Detection> &dets) {
    dets.clear();
    if (!data || shape.empty()) {
        return;
    }
    const auto dims = squeeze2d(shape);
    if (dims.size() != 2) {
        std::fprintf(stderr, "unexpected ONNX output rank after squeeze (%zu)\n", dims.size());
        return;
    }
    int rows = static_cast<int>(dims[0]);
    int cols = static_cast<int>(dims[1]);
    const int dim1 = cols;
    const bool e2e = (cols == 6 || rows == 6);
    const bool transposed = (e2e && rows == 6) ||
                            (!e2e && rows < cols && (rows == 4 + nc || rows == 5 + nc || rows < cols));
    if (transposed) {
        std::swap(rows, cols);
    }
    auto at = [&](int r, int c) -> float {
        return transposed ? data[c * dim1 + r] : data[r * dim1 + c];
    };

    if (e2e) {
        for (int i = 0; i < rows; ++i) {
            const float score = at(i, 4);
            if (score < conf) {
                continue;
            }
            float x1 = at(i, 0);
            float y1 = at(i, 1);
            float x2 = at(i, 2);
            float y2 = at(i, 3);
            scale_box(x1, y1, x2, y2, lb, orig);
            if (x2 <= x1 || y2 <= y1) {
                continue;
            }
            dets.push_back({cv::Rect2f(x1, y1, x2 - x1, y2 - y1), score, static_cast<int>(at(i, 5))});
        }
        return;
    }

    const int ncls = cols - 4;
    if (ncls <= 0) {
        std::fprintf(stderr, "ONNX output is not e2e (N,6) and not (N,4+nc): %dx%d\n", rows, cols);
        return;
    }
    for (int i = 0; i < rows; ++i) {
        int best_c = 0;
        float best = at(i, 4);
        for (int c = 1; c < ncls; ++c) {
            const float s = at(i, 4 + c);
            if (s > best) {
                best = s;
                best_c = c;
            }
        }
        if (best < conf) {
            continue;
        }
        const float cx = at(i, 0);
        const float cy = at(i, 1);
        const float w = at(i, 2);
        const float h = at(i, 3);
        float x1 = cx - w * 0.5f;
        float y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f;
        float y2 = cy + h * 0.5f;
        scale_box(x1, y1, x2, y2, lb, orig);
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }
        dets.push_back({cv::Rect2f(x1, y1, x2 - x1, y2 - y1), best, best_c});
    }
    nms(dets, nms_thr);
}

void decode_yolo26_onnx(const OnnxEngine &engine, const LetterboxInfo &lb, const cv::Size &orig,
                        float conf, float nms_thr, int nc, std::vector<Detection> &dets) {
    if (engine.output_count() >= 6) {
        decode_six_heads(engine, lb, orig, conf, nms_thr, nc, dets);
        return;
    }
    decode_single(engine.output_f32(0), engine.output_shape(0), lb, orig, conf, nms_thr, nc, dets);
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
