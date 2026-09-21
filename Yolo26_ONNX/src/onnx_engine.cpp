#include "yolo26_types.hpp"
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

const float *OnnxEngine::output_f32(int index) const {
    if (index < 0 || static_cast<size_t>(index) >= host_out_.size() || host_out_[index].empty()) {
        return nullptr;
    }
    return host_out_[index].data();
}

size_t OnnxEngine::output_elements(int index) const {
    if (index < 0 || static_cast<size_t>(index) >= host_out_.size()) {
        return 0;
    }
    return host_out_[index].size();
}

const std::vector<int64_t> &OnnxEngine::output_shape(int index) const {
    static const std::vector<int64_t> empty;
    if (index < 0 || static_cast<size_t>(index) >= out_shapes_.size()) {
        return empty;
    }
    return out_shapes_[index];
}

bool OnnxEngine::load(const std::string &onnx_path, int imgsz, int threads) {
    imgsz_ = imgsz;
    ready_ = false;
    out_names_.clear();
    host_out_.clear();
    out_shapes_.clear();
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
        in_name_ = in_name.get();
        for (size_t i = 0; i < session_.GetOutputCount(); ++i) {
            const auto name = session_.GetOutputNameAllocated(i, allocator_);
            out_names_.emplace_back(name.get());
        }

        const auto info = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        in_shape_ = info.GetShape();
        uint8_input_ = info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;

        if (in_shape_.size() != 4) {
            std::fprintf(stderr, "expected 4-D input, got rank %zu\n", in_shape_.size());
            return false;
        }
        // Static YOLO: {1,3,H,W}. Dynamic / NHWC: last dim is 3.
        nhwc_ = in_shape_[3] == 3 && in_shape_[1] != 3;
        if (!nhwc_) {
            if (in_shape_[2] > 0) {
                imgsz_ = static_cast<int>(in_shape_[2]);
            }
            in_shape_ = {1, 3, imgsz_, imgsz_};
        } else {
            if (in_shape_[1] > 0) {
                imgsz_ = static_cast<int>(in_shape_[1]);
            }
            in_shape_ = {1, imgsz_, imgsz_, 3};
        }

        std::printf("onnx input=%s %s [", in_name_.c_str(), nhwc_ ? "NHWC" : "NCHW");
        for (size_t i = 0; i < in_shape_.size(); ++i) {
            std::printf("%s%lld", i ? "," : "", static_cast<long long>(in_shape_[i]));
        }
        std::printf("] type=%s\n", uint8_input_ ? "uint8" : "float32");
        for (size_t i = 0; i < out_names_.size(); ++i) {
            const auto oinfo = session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
            const auto osh = oinfo.GetShape();
            std::printf("onnx output[%zu]=%s [", i, out_names_[i].c_str());
            for (size_t d = 0; d < osh.size(); ++d) {
                std::printf("%s%lld", d ? "," : "", static_cast<long long>(osh[d]));
            }
            std::printf("]\n");
        }
        if (out_names_.size() >= 6) {
            std::printf("decode=yolo26-6head (box_p3/p4/p5 + cls_p3/p4/p5)\n");
        } else {
            std::printf("decode=single-tensor (e2e 300x6 or 4+nc x N)\n");
        }
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
            tensor = Ort::Value::CreateTensor<uint8_t>(mem, u8.data(), u8.size(), in_shape_.data(),
                                                       in_shape_.size());
        } else {
            tensor = Ort::Value::CreateTensor<float>(mem, const_cast<float *>(nchw), elements,
                                                     in_shape_.data(), in_shape_.size());
        }

        const char *in_names[] = {in_name_.c_str()};
        std::vector<const char *> out_names;
        out_names.reserve(out_names_.size());
        for (const auto &n : out_names_) {
            out_names.push_back(n.c_str());
        }
        auto outs = session_.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names.data(),
                                 out_names.size());
        if (outs.size() != out_names_.size()) {
            std::fprintf(stderr, "ONNX returned %zu outputs, expected %zu\n", outs.size(),
                         out_names_.size());
            return false;
        }
        host_out_.resize(outs.size());
        out_shapes_.resize(outs.size());
        for (size_t i = 0; i < outs.size(); ++i) {
            if (!outs[i].IsTensor()) {
                std::fprintf(stderr, "ONNX output[%zu] is not a tensor\n", i);
                return false;
            }
            const auto info = outs[i].GetTensorTypeAndShapeInfo();
            out_shapes_[i] = info.GetShape();
            const size_t n = static_cast<size_t>(std::accumulate(
                out_shapes_[i].begin(), out_shapes_[i].end(), int64_t{1},
                [](int64_t a, int64_t b) { return a * (b > 0 ? b : 1); }));
            const float *src = outs[i].GetTensorData<float>();
            host_out_[i].assign(src, src + n);
        }
        return true;
    } catch (const Ort::Exception &ex) {
        std::fprintf(stderr, "ONNX infer failed: %s\n", ex.what());
        return false;
    }
}
