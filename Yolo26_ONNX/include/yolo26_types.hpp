#pragma once

#include <opencv2/opencv.hpp>

#ifndef YOLO26_NMS_THRESH
#define YOLO26_NMS_THRESH 0.45f
#endif
#ifndef YOLO26_CONF_THRESH
#define YOLO26_CONF_THRESH 0.25f
#endif
#ifndef YOLO26_CLASS_NUM
#define YOLO26_CLASS_NUM 80
#endif

struct YoloLetterbox {
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

using LetterboxInfo = YoloLetterbox;
