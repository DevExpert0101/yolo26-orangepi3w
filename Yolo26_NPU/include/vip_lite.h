#ifndef VIP_LITE_H
#define VIP_LITE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t  vip_status_e;
typedef int32_t  vip_enum;
typedef uint32_t vip_uint32_t;
typedef int32_t  vip_int32_t;
typedef uint8_t  vip_uint8_t;
typedef float    vip_float_t;
typedef size_t   vip_size_t;
typedef void    *vip_network;
typedef void    *vip_buffer;

#define VIP_SUCCESS 0
#define VIP_ERROR_INVALID_ARGUMENTS -1
#define VIP_ERROR_IO -2
#define VIP_ERROR_OUT_OF_MEMORY -4
#define VIP_MAX_DIM_NUM 6

typedef enum _vip_create_network_type_e {
    VIP_CREATE_NETWORK_FROM_MEMORY = 0,
    VIP_CREATE_NETWORK_FROM_FILE = 1
} vip_create_network_type_e;

typedef enum _vip_buffer_format_e {
    VIP_BUFFER_FORMAT_FP32 = 0,
    VIP_BUFFER_FORMAT_FP16 = 1,
    VIP_BUFFER_FORMAT_UINT8 = 2,
    VIP_BUFFER_FORMAT_INT8 = 3,
    VIP_BUFFER_FORMAT_INT16 = 4
} vip_buffer_format_e;

typedef enum _vip_buffer_quantize_format_e {
    VIP_BUFFER_QUANTIZE_NONE = 0,
    VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT = 1,
    VIP_BUFFER_QUANTIZE_TF_ASYMM = 2
} vip_buffer_quantize_format_e;

typedef enum _vip_buffer_memory_type_e {
    VIP_BUFFER_MEMORY_TYPE_DEFAULT = 0
} vip_buffer_memory_type_e;

typedef enum _vip_network_property_e {
    VIP_NETWORK_PROP_LAYER_COUNT = 0,
    VIP_NETWORK_PROP_INPUT_COUNT = 1,
    VIP_NETWORK_PROP_OUTPUT_COUNT = 2,
    VIP_NETWORK_PROP_NETWORK_NAME = 3
} vip_network_property_e;

typedef enum _vip_buffer_property_e {
    VIP_BUFFER_PROP_DATA_FORMAT = 0,
    VIP_BUFFER_PROP_NUM_OF_DIMENSION = 1,
    VIP_BUFFER_PROP_SIZES_OF_DIMENSION = 2,
    VIP_BUFFER_PROP_QUANT_FORMAT = 3,
    VIP_BUFFER_PROP_QUANT_DATA = 4,
    VIP_BUFFER_PROP_NAME = 5
} vip_buffer_property_e;

typedef enum _vip_buffer_operation_e {
    VIP_BUFFER_OPER_TYPE_SYNC_FOR_READ = 1,
    VIP_BUFFER_OPER_TYPE_SYNC_FOR_WRITE = 2
} vip_buffer_operation_e;

typedef struct _vip_buffer_create_params {
    vip_uint32_t num_of_dims;
    vip_uint32_t sizes[VIP_MAX_DIM_NUM];
    vip_enum     data_format;
    vip_enum     quant_format;
    union {
        struct {
            vip_uint8_t fixed_point_pos;
        } dfp;
        struct {
            vip_float_t scale;
            vip_int32_t zeroPoint;
        } affine;
    } quant_data;
    vip_uint32_t memory_type;
} vip_buffer_create_params_t;

#ifdef VIP_INIT_HAS_SIZE
vip_status_e vip_init(vip_uint32_t malloc_mb);
#else
vip_status_e vip_init(void);
#endif
vip_status_e vip_destroy(void);
vip_status_e vip_create_network(const void *data, vip_size_t size, vip_enum type, vip_network *network);
vip_status_e vip_destroy_network(vip_network network);
vip_status_e vip_query_network(vip_network network, vip_enum property, void *value);
vip_status_e vip_prepare_network(vip_network network);
vip_status_e vip_finish_network(vip_network network);
vip_status_e vip_query_input(vip_network network, vip_uint32_t index, vip_enum property, void *value);
vip_status_e vip_query_output(vip_network network, vip_uint32_t index, vip_enum property, void *value);
vip_status_e vip_create_buffer(vip_buffer_create_params_t *create_param, vip_size_t size_of_param, vip_buffer *buffer);
vip_status_e vip_destroy_buffer(vip_buffer buffer);
vip_status_e vip_set_input(vip_network network, vip_uint32_t index, vip_buffer input);
vip_status_e vip_set_output(vip_network network, vip_uint32_t index, vip_buffer output);
vip_status_e vip_run_network(vip_network network);
vip_status_e vip_flush_buffer(vip_buffer buffer, vip_enum type);
void        *vip_map_buffer(vip_buffer buffer);
vip_status_e vip_unmap_buffer(vip_buffer buffer);

#ifdef __cplusplus
}
#endif
#endif
