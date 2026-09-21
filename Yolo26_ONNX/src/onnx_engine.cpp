#include "yolo26_onnx.hpp"

#include <algorithm>
#include <cstdio>
#include <numeric>

size_t OnnxEngine::input_elements() const {
    if (in_shape_.empty()) {
        return 0;
    }
    size_t n = 1;
    for (auto d : in_shape_) {
        n *= static_cast<size_t>(d > 0 ? d : 1);
    }
    return n;
}

bool OnnxEngine::load(const std::string &onnx_path, int imgsz, int threads) {
    imgsz_ = imgsz;
    ready_ = false;
    try {
        opts_ = Ort::SessionOptions();
        opts_.SetIntraOpNumThreads(std::max(1, threads));
        opts_.SetInterOpNumThreads(1);
        opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = Ort::Session(env_, onnx_path.c_str(), opts_);

        if (session_.GetInputCount() < 1 || session_.GetOutputCount() < 1) {
            std::fprintf(stderr, "ONNX has no inputs/outputs\n");
            return false;
        }

        const auto in_name = session_.GetInputNameAllocated(0, allocator_);
        const auto out_name = session_.GetOutputNameAllocated(0, allocator_);
        in_name_ = in_name.get();
        out_name_ = out_name.get();

        const auto info = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        in_shape_ = info.GetShape();
        uint8_input_ = info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;

        if (in_shape_.size() != 4) {
            std::fprintf(stderr, "expected 4-D input, got rank %zu\n", in_shape_.size());
            return false;
        }
        // NCHW {1,3,H,W} or NHWC {1,H,W,3}
        nhwc_ = in_shape_[3] == 3 || in_shape_[1] != 3;
        for (auto &d : in_shape_) {
            if (d <= 0) {
                d = imgsz_;
            }
        }
        if (nhwc_) {
            in_shape_[1] = imgsz_;
            in_shape_[2] = imgsz_;
            in_shape_[3] = 3;
        } else {
            in_shape_[1] = 3;
            in_shape_[2] = imgsz_;
            in_shape_[3] = imgsz_;
        }

        std::printf("onnx input=%s %s [", in_name_.c_str(), nhwc_ ? "NHWC" : "NCHW");
        for (size_t i = 0; i < in_shape_.size(); ++i) {
            std::printf("%s%lld", i ? "," : "", static_cast<long long>(in_shape_[i]));
        }
        std::printf("] type=%s  output=%s\n", uint8_input_ ? "uint8" : "float32", out_name_.c_str());
        ready_ = true;
        return true;
    } catch (const Ort::Exception &ex) {
        std::fprintf(stderr, "ONNX load failed: %s\n", ex.what());
        return false;
    }
}

bool OnnxEngine::infer(const float *nchw, size_t elements) {
    if (!ready_) {
        return false;
    }
    if (elements != input_elements()) {
        std::fprintf(stderr, "input size mismatch: got %zu expected %zu\n", elements, input_elements());
        return false;
    }
    try {
        const Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<uint8_t> u8;
        Ort::Value tensor{nullptr};
        if (uint8_input_) {
            u8.resize(elements);
            for (size_t i = 0; i < elements; ++i) {
                const float v = nchw[i] * 255.f;
                u8[i] = static_cast<uint8_t>(std::clamp(v, 0.f, 255.f));
            }
            tensor = Ort::Value::CreateTensor<uint8_t>(mem, u8.data(), u8.size(), in_shape_.data(), in_shape_.size());
        } else {
            tensor = Ort::Value::CreateTensor<float>(
                mem, const_cast<float *>(nchw), elements, in_shape_.data(), in_shape_.size());
        }

        const char *in_names[] = {in_name_.c_str()};
        const char *out_names[] = {out_name_.c_str()};
        auto outs = session_.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
        if (outs.empty() || !outs[0].IsTensor()) {
            std::fprintf(stderr, "ONNX produced no tensor output\n");
            return false;
        }
        const auto out_info = outs[0].GetTensorTypeAndShapeInfo();
        out_shape_ = out_info.GetShape();
        const size_t n = static_cast<size_t>(
            std::accumulate(out_shape_.begin(), out_shape_.end(), int64_t{1},
                            [](int64_t a, int64_t b) { return a * (b > 0 ? b : 1); }));
        const float *src = outs[0].GetTensorData<float>();
        host_out_.assign(src, src + n);
        return true;
    } catch (const Ort::Exception &ex) {
        std::fprintf(stderr, "ONNX infer failed: %s\n", ex.what());
        return false;
    }
}
