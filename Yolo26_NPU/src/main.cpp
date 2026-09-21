#include "yolo26_npu.hpp"

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>

static void usage(const char *exe) {
    std::fprintf(stderr,
                 "Usage: %s <model.nb> <image|video|camera> [options]\n"
                 "  --imgsz N     input size (default 640)\n"
                 "  --conf  F     confidence (default 0.25)\n"
                 "  --nms   F     IoU NMS (default 0.45)\n"
                 "  --nc    N     classes (default 80)\n"
                 "  --loop  N     repeat image N times for FPS\n"
                 "  --no-show     do not open a window\n"
                 "  --save  PATH  write annotated image/video\n",
                 exe);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }

    const std::string model = argv[1];
    const std::string source = argv[2];
    int imgsz = 640;
    int nc = YOLO26_CLASS_NUM;
    int loops = 1;
    float conf = YOLO26_CONF_THRESH;
    float nms = YOLO26_NMS_THRESH;
    bool show = true;
    std::string save_path = "result.jpg";

    for (int i = 3; i < argc; ++i) {
        auto eat = [&](const char *key, auto &dst) {
            if (std::strcmp(argv[i], key) == 0 && i + 1 < argc) {
                dst = static_cast<std::decay_t<decltype(dst)>>(std::atof(argv[++i]));
                return true;
            }
            return false;
        };
        if (eat("--imgsz", imgsz) || eat("--conf", conf) || eat("--nms", nms) || eat("--nc", nc) ||
            eat("--loop", loops)) {
            continue;
        }
        if (std::strcmp(argv[i], "--no-show") == 0) {
            show = false;
        } else if (std::strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            save_path = argv[++i];
        }
    }

    std::printf("model=%s source=%s imgsz=%d conf=%.2f nms=%.2f\n", model.c_str(), source.c_str(), imgsz, conf, nms);

    VipEngine engine;
    if (!engine.load(model)) {
        std::fprintf(stderr, "failed to load NBG. Check /dev/vipcore and libNBGlinker.so\n");
        return 2;
    }

    const bool camera = source.size() == 1 && std::isdigit(static_cast<unsigned char>(source[0]));
    const bool video = source.size() > 4 && (source.rfind(".mp4") != std::string::npos ||
                                             source.rfind(".avi") != std::string::npos ||
                                             source.rfind(".mkv") != std::string::npos ||
                                             source.rfind(".mov") != std::string::npos);

    cv::VideoCapture cap;
    cv::VideoWriter writer;
    cv::Mat frame;
    if (camera || video) {
        if (camera) {
            cap.open(std::stoi(source));
        } else {
            cap.open(source);
        }
        if (!cap.isOpened()) {
            std::fprintf(stderr, "cannot open source %s\n", source.c_str());
            return 3;
        }
        if (!save_path.empty() && save_path.find('.') != std::string::npos && save_path.find(".jpg") == std::string::npos) {
            const double fps = cap.get(cv::CAP_PROP_FPS);
            const int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            const int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            writer.open(save_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps > 1 ? fps : 25, cv::Size(w, h));
        }
    } else {
        frame = cv::imread(source);
        if (frame.empty()) {
            std::fprintf(stderr, "cannot read image %s\n", source.c_str());
            return 3;
        }
    }

    std::vector<uint8_t> packed;
    std::vector<Detection> dets;
    double acc_ms = 0;
    int count = 0;
    const int repeats = (camera || video) ? 1 : std::max(1, loops);

    while (true) {
        if (camera || video) {
            if (!cap.read(frame) || frame.empty()) {
                break;
            }
        }
        for (int r = 0; r < repeats; ++r) {
            cv::Mat rgb, letter;
            cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
            const LetterboxInfo lb = letterbox(rgb, letter, imgsz);
            pack_nchw_uint8(letter, packed);

            const auto t0 = std::chrono::steady_clock::now();
            if (!engine.infer(packed.data(), packed.size())) {
                std::fprintf(stderr, "NPU infer failed\n");
                return 4;
            }
            decode_yolo26_6(engine, lb, frame.size(), conf, nms, nc, dets);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            acc_ms += ms;
            ++count;

            draw_detections(frame, dets);
            char hud[64];
            std::snprintf(hud, sizeof(hud), "%.1f ms  %.1f FPS  %zu det", ms, ms > 0 ? 1000.0 / ms : 0, dets.size());
            cv::putText(frame, hud, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
            std::printf("%s\n", hud);

            if (writer.isOpened()) {
                writer.write(frame);
            }
            if (show) {
                cv::imshow("YOLO26 NPU", frame);
                const int key = cv::waitKey(camera || video ? 1 : 0);
                if (key == 27 || key == 'q') {
                    return 0;
                }
            }
        }
        if (!camera && !video) {
            if (!save_path.empty()) {
                cv::imwrite(save_path, frame);
                std::printf("saved %s\n", save_path.c_str());
            }
            break;
        }
    }

    if (count) {
        std::printf("avg %.1f ms (%.1f FPS) over %d frames\n", acc_ms / count, 1000.0 * count / acc_ms, count);
    }
    return 0;
}
