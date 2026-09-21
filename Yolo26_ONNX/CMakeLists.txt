cmake_minimum_required(VERSION 3.10)
project(Yolo26_ONNX LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH
    "Folder that contains ONNX Runtime include/ and lib/")
if(NOT ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime)
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS
        ${ONNXRUNTIME_ROOT}/include
        /usr/include
        /usr/local/include
        /usr/include/onnxruntime
        /usr/local/include/onnxruntime
)
find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    HINTS
        ${ONNXRUNTIME_ROOT}/lib
        /usr/lib
        /usr/lib/aarch64-linux-gnu
        /usr/local/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR
        "ONNX Runtime not found. On the Orange Pi run: ../setup_onnx.sh\n"
        "Or set -DONNXRUNTIME_ROOT=/path/to/onnxruntime (include/ + lib/)")
endif()

get_filename_component(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_LIB} DIRECTORY)

add_executable(yolo26_onnx
    src/main.cpp
    src/onnx_engine.cpp
    src/yolo26_post.cpp
)

target_include_directories(yolo26_onnx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolo26_onnx PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
    pthread
    dl
)

set_target_properties(yolo26_onnx PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(yolo26_onnx PRIVATE -march=armv8-a)
endif()

install(TARGETS yolo26_onnx RUNTIME DESTINATION bin)
