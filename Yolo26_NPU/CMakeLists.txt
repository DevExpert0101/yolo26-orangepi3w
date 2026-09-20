cmake_minimum_required(VERSION 3.10)
project(Yolo26_NPU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

find_package(OpenCV REQUIRED)
option(VIP_INIT_HAS_SIZE "vip_init takes a malloc-MB argument (some Allwinner VIPLite builds)" OFF)

set(VIPLITE_ROOT "$ENV{VIPLITE_ROOT}" CACHE PATH "Folder that contains VIPLite headers and libs")
set(VIPLITE_INCLUDE_HINTS
    ${VIPLITE_ROOT}/include
    /usr/include
    /usr/local/include
    /opt/npu/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set(VIPLITE_LIB_HINTS
    ${VIPLITE_ROOT}/lib
    /usr/lib
    /usr/lib/aarch64-linux-gnu
    /usr/local/lib
    /opt/yolov5
    /opt/npu/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

find_path(VIP_LITE_INCLUDE_DIR
    NAMES vip_lite.h
    HINTS ${VIPLITE_INCLUDE_HINTS}
)
find_library(VIP_NBGLINKER
    NAMES NBGlinker libNBGlinker
    HINTS ${VIPLITE_LIB_HINTS}
)
find_library(VIP_HAL
    NAMES VIPhal libVIPhal
    HINTS ${VIPLITE_LIB_HINTS}
)

if(NOT VIP_LITE_INCLUDE_DIR)
    message(STATUS "Board vip_lite.h not found, using bundled header")
    set(VIP_LITE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

if(NOT VIP_NBGLINKER)
    message(WARNING "libNBGlinker.so not found. Set VIPLITE_ROOT or copy the .so files into ${CMAKE_CURRENT_SOURCE_DIR}/lib")
endif()

add_executable(yolo26_npu
    src/main.cpp
    src/viplite_engine.cpp
    src/yolo26_post.cpp
)

target_include_directories(yolo26_npu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${VIP_LITE_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolo26_npu PRIVATE
    ${OpenCV_LIBS}
    ${VIP_NBGLINKER}
    ${VIP_HAL}
    dl
    pthread
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(yolo26_npu PRIVATE -march=armv8-a)
endif()
if(VIP_INIT_HAS_SIZE)
    target_compile_definitions(yolo26_npu PRIVATE VIP_INIT_HAS_SIZE)
endif()

install(TARGETS yolo26_npu RUNTIME DESTINATION bin)
