#include "yolo26_npu.hpp"

#include <algorithm>
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
                 "  --imgsz   N     input size (default 640)\n"
                 "  --conf    F     confidence (default 0.25)\n"
                 "  --nms     F     IoU NMS (default 0.45)\n"
                 "  --nc      N     classes (default 80)\n"
                 "  --loop    N     timed image repeats after warmup (default 1)\n"
                 "  --warmup  N     discarded runs before timing (default 3 if --loop>1)\n"
                 "  --no-show       do not open a window\n"
                 "  --save    PATH  write annotated image/video\n",
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
    int warmup = -1;
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
            eat("--loop", loops) || eat("--warmup", warmup)) {
            continue;
        }
        if (std::strcmp(argv[i], "--no-show") == 0) {
            show = false;
        } else if (std::strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            save_path = argv[++i];
        }
    }

    std::printf("model=%s source=%s imgsz=%d conf=%.2f nms=%.2f\n", model.c_str(), source.c_str(), imgsz,
                conf, nms);

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
        if (!save_path.empty() && save_path.find('.') != std::string::npos &&
            save_path.find(".jpg") == std::string::npos) {
            const double fps = cap.get(cv::CAP_PROP_FPS);
            const int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            const int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            writer.open(save_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps > 1 ? fps : 25,
                        cv::Size(w, h));
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
    double acc_pre = 0, acc_inf = 0, acc_post = 0, acc_e2e = 0;
    double min_inf = 1e100, max_inf = 0, min_e2e = 1e100, max_e2e = 0;
    int count = 0;
    const int repeats = (camera || video) ? 1 : std::max(1, loops);
    if (warmup < 0) {
        warmup = (repeats > 1 && !camera && !video) ? 3 : 0;
    }

    using clock = std::chrono::steady_clock;
    auto ms_since = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    while (true) {
        if (camera || video) {
            if (!cap.read(frame) || frame.empty()) {
                break;
            }
        }
        const int total_runs = repeats + ((!camera && !video) ? warmup : 0);
        for (int r = 0; r < total_runs; ++r) {
            const bool timed = camera || video || r >= warmup;
            cv::Mat rgb, letter;
            const auto t0 = clock::now();
            cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
            const LetterboxInfo lb = letterbox(rgb, letter, imgsz);
            engine.pack_rgb(letter, packed);
            const auto t1 = clock::now();
            if (!engine.infer(packed.data(), packed.size())) {
                std::fprintf(stderr, "NPU infer failed\n");
                return 4;
            }
            const auto t2 = clock::now();
            decode_yolo26_6(engine, lb, frame.size(), conf, nms, nc, dets);
            const auto t3 = clock::now();

            const double pre = ms_since(t0, t1);
            const double inf = ms_since(t1, t2);
            const double post = ms_since(t2, t3);
            const double e2e = ms_since(t0, t3);

            if (timed) {
                acc_pre += pre;
                acc_inf += inf;
                acc_post += post;
                acc_e2e += e2e;
                min_inf = std::min(min_inf, inf);
                max_inf = std::max(max_inf, inf);
                min_e2e = std::min(min_e2e, e2e);
                max_e2e = std::max(max_e2e, e2e);
                ++count;
            }

            if (camera || video || r + 1 == total_runs) {
                cv::Mat vis = frame.clone();
                draw_detections(vis, dets);
                char hud[96];
                std::snprintf(hud, sizeof(hud), "NPU %.1f ms  e2e %.1f ms  %zu det", inf, e2e,
                              dets.size());
                cv::putText(vis, hud, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                            cv::Scalar(0, 255, 255), 2);
                std::printf("%s%s  pre %.1f  post %.1f\n", timed ? "" : "warmup ", hud, pre, post);
                if (writer.isOpened()) {
                    writer.write(vis);
                }
                if (show) {
                    cv::imshow("YOLO26 NPU", vis);
                    const int key = cv::waitKey(camera || video ? 1 : 0);
                    if (key == 27 || key == 'q') {
                        return 0;
                    }
                }
                if (!camera && !video && !save_path.empty()) {
                    cv::imwrite(save_path, vis);
                    std::printf("saved %s\n", save_path.c_str());
                }
            } else if ((r + 1) % 10 == 0 || r < warmup) {
                std::printf("%s#%d  NPU %.1f ms  e2e %.1f ms  %zu det\n", timed ? "" : "warmup ", r + 1,
                            inf, e2e, dets.size());
            }
        }
        if (!camera && !video) {
            break;
        }
    }

    if (count) {
        std::printf("\n--- NPU timing (%d timed runs, %d warmup discarded) ---\n", count,
                    (!camera && !video) ? warmup : 0);
        std::printf("NPU infer   avg %.1f  min %.1f  max %.1f ms   (%.2f FPS)\n", acc_inf / count,
                    min_inf, max_inf, 1000.0 * count / acc_inf);
        std::printf("preprocess  avg %.1f ms\n", acc_pre / count);
        std::printf("postprocess avg %.1f ms\n", acc_post / count);
        std::printf("end-to-end  avg %.1f  min %.1f  max %.1f ms   (%.2f FPS)\n", acc_e2e / count,
                    min_e2e, max_e2e, 1000.0 * count / acc_e2e);
    }
    return 0;
}
