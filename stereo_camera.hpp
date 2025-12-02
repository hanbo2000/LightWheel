#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <sys/stat.h>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <functional>

// 由 save_to_disk.cpp 提供
extern void saveImageTask(std::function<void()> task);
extern bool SHOW_IMAGE;

// 由 save_to_disk.cpp 提供（与深度/彩色相机共用）
void createDirectory(const std::string &path);

/* ----------------------------
 *  Stereo Camera 配置
 * ---------------------------- */
struct StereoCameraConfig {
    bool enable = false;                     // 总开关：默认关闭，想用改成 true
    std::string name = "stereo_camera";
    std::string device = "/dev/video8";      // 记得根据 v4l2-ctl 调整
    std::string output_dir = "stereo_camera";
    int width = 3200;
    int height = 1200;
    int fps = 30;
    std::string fourcc = "MJPG";
    std::string file_ext = ".jpg";
};

/* ----------------------------
 *  Stereo Camera 统计
 * ---------------------------- */
struct StereoCameraStats {
    int frames_captured = 0;
    int frames_failed = 0;
    std::string first_frame_ts;
    std::string last_frame_ts;
};

/* ----------------------------
 *  Stereo Camera 类
 * ---------------------------- */
class StereoCamera {
public:
    StereoCameraConfig cfg;
    StereoCameraStats stats;

    StereoCamera(const StereoCameraConfig &config) : cfg(config) {}

    /* C++11 兼容：毫秒级时间戳 */
    static long long currentTimestampMs() {
        using namespace std::chrono;
        return duration_cast<milliseconds>(
                system_clock::now().time_since_epoch()).count();
    }

    /* 主采集函数（内部自己处理异常，不让它跑出线程） */
    void run(double duration_sec) {
        try {
            if (!cfg.enable) {
                std::cout << "[StereoCamera] disabled, skip run().\n";
                return;
            }

            createDirectory(cfg.output_dir);

            cv::VideoCapture cap(cfg.device, cv::CAP_V4L2);
            if (!cap.isOpened()) {
                std::cerr << "[StereoCamera] ERROR: Cannot open device: "
                          << cfg.device << std::endl;
                return;
            }

            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc(
                    cfg.fourcc[0], cfg.fourcc[1], cfg.fourcc[2], cfg.fourcc[3]));
            cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
            cap.set(cv::CAP_PROP_FPS, cfg.fps);

            cv::Mat frame;
            int frameIndex = 0;
            auto start = std::chrono::steady_clock::now();

            while (true) {
                auto elapsed = std::chrono::steady_clock::now() - start;
                if (elapsed >= std::chrono::duration<double>(duration_sec))
                    break;

                if (!cap.read(frame)) {
                    stats.frames_failed++;
                    std::cerr << "[StereoCamera] read frame failed\n";
                    continue;
                }

                // 可选显示
                if (SHOW_IMAGE) {
                    cv::imshow(cfg.name, frame);
                    if (cv::waitKey(1) == 27)
                        break;
                }

                long long ts = currentTimestampMs();

                // 文件名：Stereo_3200x1200_000001_123456ms.jpg
                char fname[256];
                sprintf(fname,
                        "Stereo_%dx%d_%06d_%lldms%s",
                        cfg.width, cfg.height,
                        frameIndex, ts,
                        cfg.file_ext.c_str()
                );

                std::string save_path = cfg.output_dir + "/" + fname;

                // 异步保存（内部捕获异常）
                cv::Mat copyFrame = frame.clone();
                saveImageTask([save_path, copyFrame]() {
                    try {
                        if (!cv::imwrite(save_path, copyFrame)) {
                            std::cerr << "[StereoCamera] imwrite failed: "
                                      << save_path << std::endl;
                        }
                    }
                    catch (const cv::Exception &e) {
                        std::cerr << "[StereoCamera] OpenCV exception: "
                                  << e.what() << std::endl;
                    }
                });

                stats.frames_captured++;
                if (stats.first_frame_ts.empty())
                    stats.first_frame_ts = std::to_string(ts);
                stats.last_frame_ts = std::to_string(ts);

                frameIndex++;
            }

            std::cout << "[StereoCamera] Saved " << stats.frames_captured
                      << " frames to folder: " << cfg.output_dir << std::endl;
        }
        catch (const std::exception &e) {
            std::cerr << "[StereoCamera::run] std::exception: "
                      << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "[StereoCamera::run] unknown exception\n";
        }
    }
};

