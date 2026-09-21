#include "yolo26_npu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

static bool ok(vip_status_e status, const char *what) {
    if (status != VIP_SUCCESS) {
        std::fprintf(stderr, "%s failed, vip_status=%d\n", what, status);
        return false;
    }
    return true;
}

VipEngine::~VipEngine() { close(); }

void VipEngine::close() {
    if (!ready_ && !network_) {
        return;
    }
    for (auto &buf : in_buf_) {
        if (buf) {
            vip_destroy_buffer(buf);
            buf = nullptr;
        }
    }
    for (auto &buf : out_buf_) {
        if (buf) {
            vip_destroy_buffer(buf);
            buf = nullptr;
        }
    }
    if (network_) {
        vip_finish_network(network_);
        vip_destroy_network(network_);
        network_ = nullptr;
    }
    vip_destroy();
    ready_ = false;
}

size_t VipEngine::format_bytes(int format) const {
    switch (format) {
        case VIP_BUFFER_FORMAT_UINT8:
        case VIP_BUFFER_FORMAT_INT8:
            return 1;
        case VIP_BUFFER_FORMAT_FP16:
        case VIP_BUFFER_FORMAT_INT16:
            return 2;
        default:
            return 4;
    }
}

static const char *format_name(int format) {
    switch (format) {
        case VIP_BUFFER_FORMAT_FP32:
            return "fp32";
        case VIP_BUFFER_FORMAT_FP16:
            return "fp16";
        case VIP_BUFFER_FORMAT_UINT8:
            return "uint8";
        case VIP_BUFFER_FORMAT_INT8:
            return "int8";
        case VIP_BUFFER_FORMAT_INT16:
            return "int16";
        default:
            return "unknown";
    }
}

static void query_quant(vip_network net, bool is_input, uint32_t index, TensorInfo &info) {
    auto query = is_input ? vip_query_input : vip_query_output;
    vip_enum qf = VIP_BUFFER_QUANTIZE_NONE;
    if (query(net, index, VIP_BUFFER_PROP_QUANT_FORMAT, &qf) != VIP_SUCCESS) {
        return;
    }
    info.quant_format = qf;
    info.scale = 1.f;
    info.zero_point = 0;
    info.fl = 0;
    if (qf == VIP_BUFFER_QUANTIZE_TF_ASYMM) {
        struct {
            vip_float_t scale;
            vip_int32_t zp;
        } aff = {};
        if (query(net, index, VIP_BUFFER_PROP_QUANT_DATA, &aff) == VIP_SUCCESS && aff.scale > 0.f) {
            info.scale = aff.scale;
            info.zero_point = aff.zp;
        }
    } else if (qf == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        vip_uint8_t fl = 0;
        if (query(net, index, VIP_BUFFER_PROP_QUANT_DATA, &fl) == VIP_SUCCESS) {
            info.fl = fl;
        }
    }
}

static float dequant_q(double q, const TensorInfo &info) {
    if (info.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        return static_cast<float>(q * std::ldexp(1.0, -info.fl));
    }
    if (info.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM || info.scale != 1.f || info.zero_point != 0) {
        return static_cast<float>((q - static_cast<double>(info.zero_point)) * static_cast<double>(info.scale));
    }
    return static_cast<float>(q);
}

static uint16_t f32_to_f16(float f) {
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000);
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) {
            return sign;
        }
        man = (man | 0x800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | (man >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00);
    }
    return static_cast<uint16_t>(sign | (exp << 10) | (man >> 13));
}

static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        f = (sign << 31) | (man ? (man << 13) : 0);
    } else if (exp == 31) {
        f = (sign << 31) | 0x7f800000 | (man << 13);
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

bool VipEngine::load(const std::string &nbg_path) {
    close();
#ifdef VIP_INIT_HAS_SIZE
    if (!ok(vip_init(16), "vip_init")) {
#else
    if (!ok(vip_init(), "vip_init")) {
#endif
        return false;
    }
    if (!ok(vip_create_network(nbg_path.c_str(), 0, VIP_CREATE_NETWORK_FROM_FILE, &network_),
            "vip_create_network")) {
        return false;
    }
    if (!ok(vip_prepare_network(network_), "vip_prepare_network")) {
        return false;
    }
    if (!query_io() || !create_buffers()) {
        return false;
    }
    ready_ = true;
    return true;
}

bool VipEngine::query_io() {
    vip_uint32_t in_n = 0;
    vip_uint32_t out_n = 0;
    if (!ok(vip_query_network(network_, VIP_NETWORK_PROP_INPUT_COUNT, &in_n), "query input count") ||
        !ok(vip_query_network(network_, VIP_NETWORK_PROP_OUTPUT_COUNT, &out_n), "query output count")) {
        return false;
    }
    if (in_n == 0 || out_n == 0 || in_n > YOLO26_MAX_IO || out_n > YOLO26_MAX_IO) {
        std::fprintf(stderr, "unexpected io counts in=%u out=%u\n", in_n, out_n);
        return false;
    }

    auto fill = [&](bool is_input, uint32_t count, std::vector<TensorInfo> &dst) {
        dst.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            TensorInfo info;
            char name[256] = {};
            vip_uint32_t ndims = 0;
            vip_enum fmt = 0;
            auto query = is_input ? vip_query_input : vip_query_output;
            query(network_, i, VIP_BUFFER_PROP_NAME, name);
            query(network_, i, VIP_BUFFER_PROP_NUM_OF_DIMENSION, &ndims);
            query(network_, i, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, info.sizes);
            query(network_, i, VIP_BUFFER_PROP_DATA_FORMAT, &fmt);
            info.name = name;
            info.dims = static_cast<int>(ndims);
            info.format = fmt;
            query_quant(network_, is_input, i, info);
            info.elements = 1;
            for (int d = 0; d < info.dims; ++d) {
                info.elements *= info.sizes[d] ? info.sizes[d] : 1;
            }
            info.bytes = info.elements * format_bytes(info.format);
            std::printf("%s %u name=%s dim=", is_input ? "input" : "output", i, info.name.c_str());
            for (int d = 0; d < info.dims; ++d) {
                std::printf("%u ", info.sizes[d]);
            }
            std::printf("format=%s elems=%zu", format_name(info.format), info.elements);
            if (info.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) {
                std::printf(" quant=asymm scale=%.8g zp=%d", info.scale, info.zero_point);
            } else if (info.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
                std::printf(" quant=dfp fl=%d", info.fl);
            }
            std::printf("\n");
            dst[i] = info;
        }
    };
    fill(true, in_n, inputs_);
    fill(false, out_n, outputs_);
    host_out_.assign(outputs_.size(), {});
    return true;
}

bool VipEngine::create_buffers() {
    bool out_need_driver_dequant = false;
    for (const auto &o : outputs_) {
        if ((o.format == VIP_BUFFER_FORMAT_INT8 || o.format == VIP_BUFFER_FORMAT_UINT8) &&
            o.quant_format == VIP_BUFFER_QUANTIZE_NONE && o.zero_point == 0 &&
            std::fabs(o.scale - 1.f) < 1e-6f) {
            out_need_driver_dequant = true;
        }
    }

    auto make = [&](const TensorInfo &info, vip_buffer *handle, bool as_fp32) {
        vip_buffer_create_params_t p;
        std::memset(&p, 0, sizeof(p));
        p.num_of_dims = static_cast<vip_uint32_t>(info.dims);
        for (int i = 0; i < info.dims; ++i) {
            p.sizes[i] = info.sizes[i];
        }
        p.data_format = as_fp32 ? VIP_BUFFER_FORMAT_FP32 : info.format;
        p.quant_format = as_fp32 ? VIP_BUFFER_QUANTIZE_NONE : info.quant_format;
        if (!as_fp32 && info.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
            p.quant_data.dfp.fixed_point_pos = static_cast<vip_uint8_t>(info.fl);
        } else {
            p.quant_data.affine.scale = as_fp32 ? 1.f : info.scale;
            p.quant_data.affine.zeroPoint = as_fp32 ? 0 : info.zero_point;
        }
        p.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
        return ok(vip_create_buffer(&p, sizeof(p), handle), "vip_create_buffer");
    };

    for (size_t i = 0; i < inputs_.size(); ++i) {
        if (!make(inputs_[i], &in_buf_[i], false) ||
            !ok(vip_set_input(network_, static_cast<vip_uint32_t>(i), in_buf_[i]), "vip_set_input")) {
            return false;
        }
    }

    auto create_outs = [&](bool as_fp32) -> bool {
        for (size_t i = 0; i < outputs_.size(); ++i) {
            if (out_buf_[i]) {
                vip_destroy_buffer(out_buf_[i]);
                out_buf_[i] = nullptr;
            }
            if (!make(outputs_[i], &out_buf_[i], as_fp32) ||
                !ok(vip_set_output(network_, static_cast<vip_uint32_t>(i), out_buf_[i]), "vip_set_output")) {
                return false;
            }
            host_out_[i].resize(outputs_[i].elements);
        }
        return true;
    };

    if (out_need_driver_dequant) {
        std::printf("output quant not reported; requesting FP32 buffers from VIPLite\n");
        if (create_outs(true)) {
            for (auto &o : outputs_) {
                o.format = VIP_BUFFER_FORMAT_FP32;
                o.quant_format = VIP_BUFFER_QUANTIZE_NONE;
                o.scale = 1.f;
                o.zero_point = 0;
                o.bytes = o.elements * 4;
            }
            return true;
        }
        std::fprintf(stderr, "FP32 output buffers failed, falling back to native format\n");
    }
    return create_outs(false);
}

void VipEngine::pack_rgb(const cv::Mat &rgb, std::vector<uint8_t> &dst) const {
    if (inputs_.empty()) {
        dst.clear();
        return;
    }
    const TensorInfo &in = inputs_[0];
    const int h = rgb.rows;
    const int w = rgb.cols;
    const size_t hw = static_cast<size_t>(h) * static_cast<size_t>(w);
    const bool nhwc = in.dims >= 3 && in.sizes[in.dims - 1] == 3 && in.sizes[0] != 3;

    float scale = in.scale > 0.f ? in.scale : (1.f / 255.f);
    int zp = in.zero_point;
    if (in.format == VIP_BUFFER_FORMAT_INT8 && in.quant_format == VIP_BUFFER_QUANTIZE_NONE &&
        zp == 0 && std::fabs(scale - 1.f) < 1e-6f) {
        // A733 PCQ NBGs store RGB as int8 with zp=-128, scale=1/255 (see nbg_meta.json).
        scale = 1.f / 255.f;
        zp = -128;
    }

    auto quant_u8 = [&](uint8_t px) -> uint8_t { return px; };
    auto quant_i8 = [&](uint8_t px) -> int8_t {
        const int q = static_cast<int>(std::lround(static_cast<float>(px) / 255.f / scale + zp));
        return static_cast<int8_t>(std::clamp(q, -128, 127));
    };
    auto quant_f32 = [&](uint8_t px) -> float { return static_cast<float>(px) / 255.f; };

    dst.resize(in.bytes);
    std::memset(dst.data(), 0, dst.size());
    std::vector<cv::Mat> ch(3);
    cv::split(rgb, ch);

    if (in.format == VIP_BUFFER_FORMAT_FP32 || in.format == VIP_BUFFER_FORMAT_FP16) {
        if (in.format == VIP_BUFFER_FORMAT_FP16) {
            auto *out = reinterpret_cast<uint16_t *>(dst.data());
            if (nhwc) {
                const uint8_t *src = rgb.data;
                for (size_t i = 0; i < hw * 3; ++i) {
                    out[i] = f32_to_f16(quant_f32(src[i]));
                }
            } else {
                for (int c = 0; c < 3; ++c) {
                    const uint8_t *src = ch[c].data;
                    uint16_t *plane = out + static_cast<size_t>(c) * hw;
                    for (size_t i = 0; i < hw; ++i) {
                        plane[i] = f32_to_f16(quant_f32(src[i]));
                    }
                }
            }
            return;
        }
        auto *out = reinterpret_cast<float *>(dst.data());
        if (nhwc) {
            const uint8_t *src = rgb.data;
            for (size_t i = 0; i < hw * 3; ++i) {
                out[i] = quant_f32(src[i]);
            }
        } else {
            for (int c = 0; c < 3; ++c) {
                const uint8_t *src = ch[c].data;
                float *plane = out + static_cast<size_t>(c) * hw;
                for (size_t i = 0; i < hw; ++i) {
                    plane[i] = quant_f32(src[i]);
                }
            }
        }
        return;
    }

    if (in.format == VIP_BUFFER_FORMAT_INT8) {
        auto *out = reinterpret_cast<int8_t *>(dst.data());
        if (nhwc) {
            const uint8_t *src = rgb.data;
            for (size_t i = 0; i < hw * 3; ++i) {
                out[i] = quant_i8(src[i]);
            }
        } else {
            for (int c = 0; c < 3; ++c) {
                const uint8_t *src = ch[c].data;
                int8_t *plane = out + static_cast<size_t>(c) * hw;
                for (size_t i = 0; i < hw; ++i) {
                    plane[i] = quant_i8(src[i]);
                }
            }
        }
        return;
    }

    // UINT8 (and anything else 1-byte): raw 0-255 RGB.
    if (nhwc) {
        std::memcpy(dst.data(), rgb.data, std::min(dst.size(), hw * 3));
    } else {
        pack_nchw_uint8(rgb, dst);
        if (dst.size() < in.bytes) {
            dst.resize(in.bytes, 0);
        }
    }
}

bool VipEngine::infer(const uint8_t *packed, size_t bytes) {
    if (!ready_) {
        return false;
    }
    void *ptr = vip_map_buffer(in_buf_[0]);
    if (!ptr) {
        std::fprintf(stderr, "vip_map_buffer input is null\n");
        return false;
    }
    const size_t n = std::min(bytes, inputs_[0].bytes);
    std::memcpy(ptr, packed, n);
    vip_unmap_buffer(in_buf_[0]);
    if (!ok(vip_flush_buffer(in_buf_[0], VIP_BUFFER_OPER_TYPE_SYNC_FOR_WRITE), "flush input")) {
        return false;
    }
    if (!ok(vip_run_network(network_), "vip_run_network")) {
        return false;
    }
    for (size_t i = 0; i < outputs_.size(); ++i) {
        if (!ok(vip_flush_buffer(out_buf_[i], VIP_BUFFER_OPER_TYPE_SYNC_FOR_READ), "flush output")) {
            return false;
        }
        void *optr = vip_map_buffer(out_buf_[i]);
        if (!optr) {
            std::fprintf(stderr, "vip_map_buffer output %zu is null\n", i);
            return false;
        }
        const TensorInfo &info = outputs_[i];
        const size_t count = info.elements;
        float *dst = host_out_[i].data();
        if (info.format == VIP_BUFFER_FORMAT_FP32) {
            std::memcpy(dst, optr, count * sizeof(float));
        } else if (info.format == VIP_BUFFER_FORMAT_UINT8) {
            const auto *src = static_cast<const uint8_t *>(optr);
            for (size_t k = 0; k < count; ++k) {
                dst[k] = dequant_q(src[k], info);
            }
        } else if (info.format == VIP_BUFFER_FORMAT_INT8) {
            const auto *src = static_cast<const int8_t *>(optr);
            for (size_t k = 0; k < count; ++k) {
                dst[k] = dequant_q(src[k], info);
            }
        } else if (info.format == VIP_BUFFER_FORMAT_INT16) {
            const auto *src = static_cast<const int16_t *>(optr);
            for (size_t k = 0; k < count; ++k) {
                dst[k] = dequant_q(src[k], info);
            }
        } else {
            const auto *src = static_cast<const uint16_t *>(optr);
            for (size_t k = 0; k < count; ++k) {
                dst[k] = fp16_to_f32(src[k]);
            }
        }
        vip_unmap_buffer(out_buf_[i]);
    }
    return true;
}

const float *VipEngine::output_f32(int index) const {
    return host_out_.at(static_cast<size_t>(index)).data();
}

size_t VipEngine::output_elements(int index) const {
    return host_out_.at(static_cast<size_t>(index)).size();
}
