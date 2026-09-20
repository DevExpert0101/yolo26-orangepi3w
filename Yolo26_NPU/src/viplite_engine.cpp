#include "yolo26_npu.hpp"

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
            info.elements = 1;
            for (int d = 0; d < info.dims; ++d) {
                info.elements *= info.sizes[d] ? info.sizes[d] : 1;
            }
            info.bytes = info.elements * format_bytes(info.format);
            std::printf("%s %u name=%s dim=", is_input ? "input" : "output", i, info.name.c_str());
            for (int d = 0; d < info.dims; ++d) {
                std::printf("%u ", info.sizes[d]);
            }
            std::printf("format=%d elems=%zu\n", info.format, info.elements);
            dst[i] = info;
        }
    };
    fill(true, in_n, inputs_);
    fill(false, out_n, outputs_);
    host_out_.assign(outputs_.size(), {});
    return true;
}

bool VipEngine::create_buffers() {
    auto make = [&](const TensorInfo &info, vip_buffer *handle) {
        vip_buffer_create_params_t p;
        std::memset(&p, 0, sizeof(p));
        p.num_of_dims = static_cast<vip_uint32_t>(info.dims);
        for (int i = 0; i < info.dims; ++i) {
            p.sizes[i] = info.sizes[i];
        }
        p.data_format = info.format;
        p.quant_format = VIP_BUFFER_QUANTIZE_NONE;
        p.quant_data.affine.scale = 1.f;
        p.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
        return ok(vip_create_buffer(&p, sizeof(p), handle), "vip_create_buffer");
    };

    for (size_t i = 0; i < inputs_.size(); ++i) {
        if (!make(inputs_[i], &in_buf_[i]) || !ok(vip_set_input(network_, static_cast<vip_uint32_t>(i), in_buf_[i]), "vip_set_input")) {
            return false;
        }
    }
    for (size_t i = 0; i < outputs_.size(); ++i) {
        if (!make(outputs_[i], &out_buf_[i]) ||
            !ok(vip_set_output(network_, static_cast<vip_uint32_t>(i), out_buf_[i]), "vip_set_output")) {
            return false;
        }
        host_out_[i].resize(outputs_[i].elements);
    }
    return true;
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
        const int fmt = outputs_[i].format;
        const size_t count = outputs_[i].elements;
        float *dst = host_out_[i].data();
        if (fmt == VIP_BUFFER_FORMAT_FP32) {
            std::memcpy(dst, optr, count * sizeof(float));
        } else if (fmt == VIP_BUFFER_FORMAT_UINT8) {
            const auto *src = static_cast<const uint8_t *>(optr);
            for (size_t k = 0; k < count; ++k) {
                dst[k] = static_cast<float>(src[k]);
            }
        } else if (fmt == VIP_BUFFER_FORMAT_INT8) {
            const auto *src = static_cast<const int8_t *>(optr);
            for (size_t k = 0; k < count; ++k) {
                dst[k] = static_cast<float>(src[k]);
            }
        } else if (fmt == VIP_BUFFER_FORMAT_INT16) {
            const auto *src = static_cast<const int16_t *>(optr);
            for (size_t k = 0; k < count; ++k) {
                dst[k] = static_cast<float>(src[k]);
            }
        } else {
            const auto *src = static_cast<const uint16_t *>(optr);
            for (size_t k = 0; k < count; ++k) {
                uint16_t h = src[k];
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
                std::memcpy(dst + k, &f, sizeof(float));
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
