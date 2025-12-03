#include <libobsensor/ObSensor.hpp>
#include "utils.hpp"
#include "utils_opencv.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <sys/stat.h>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

#include "stereo_camera.hpp"

bool SHOW_IMAGE   = false;   
bool ENABLE_STEREO = false;  

void createDirectory(const std::string& path);
void saveDepthFrame(const std::shared_ptr<ob::DepthFrame> depthFrame, uint32_t frameIndex);
void saveColorFrame(const std::shared_ptr<ob::ColorFrame> colorFrame, uint32_t frameIndex);
void saveImageTask(std::function<void()> task);

std::queue<std::function<void()>> taskQueue;
std::mutex queueMutex;
std::condition_variable conditionVar;
bool stopThread = false;

void saveImageTask(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        taskQueue.push(task);
    }
    conditionVar.notify_one();
}

void taskProcessor() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            conditionVar.wait(lock, [] { return !taskQueue.empty() || stopThread; });

            if (stopThread && taskQueue.empty()) {
                return;
            }

            task = taskQueue.front();
            taskQueue.pop();
        }

        try {
            task();
        }
        catch (const std::exception& e) {
            std::cerr << "[taskProcessor] Exception: " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "[taskProcessor] Unknown exception in task" << std::endl;
        }
    }
}

// ----------------- 主函数 -----------------
int main() {
    try {
        // 如果需要命令行控制显示，可以改成从 argv 解析
        // SHOW_IMAGE = true;  // 测试时可打开

        // 启动任务处理线程
        std::thread worker(taskProcessor);

        // Stereo camera 配置（可选）
        StereoCameraConfig stereoCfg;
        stereoCfg.enable     = ENABLE_STEREO;
        stereoCfg.device     = "/dev/video8";    // 根据实际设备修改
        stereoCfg.width      = 3200;
        stereoCfg.height     = 1200;
        stereoCfg.fps        = 30;
        stereoCfg.output_dir = "stereo_camera";

        StereoCamera stereo(stereoCfg);

        std::thread stereoThread;
        if (stereoCfg.enable) {
            stereoThread = std::thread([&]() {
                stereo.run(5.0);  // 例如采 5 秒
            });
        }

        // 创建 Orbbec Pipeline
        std::shared_ptr<ob::Pipeline> pipeline = std::make_shared<ob::Pipeline>();
        std::shared_ptr<ob::Config>   config   = std::make_shared<ob::Config>();

        // 配置彩色 + 深度流（根据设备实际支持调整）
        config->enableVideoStream(OB_STREAM_COLOR, 1920, 1080, 30, OB_FORMAT_MJPG);
        config->enableVideoStream(OB_STREAM_DEPTH, 1280, 800,  30, OB_FORMAT_Y16);
        config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);

        uint32_t frameIndex = 0;
        auto formatConverter = std::make_shared<ob::FormatConvertFilter>();

        // 启动 pipeline（加异常保护）
        try {
            pipeline->start(config);
        }
        catch (const ob::Error &e) {
            std::cerr << "[main] Failed to start pipeline: "
                      << e.getMessage() << std::endl;

            // 清理线程
            {
                std::lock_guard<std::mutex> lock(queueMutex);
                stopThread = true;
            }
            conditionVar.notify_one();
            worker.join();
            if (stereoCfg.enable && stereoThread.joinable())
                stereoThread.join();
            return -1;
        }

        // 丢弃前几帧
        for (int i = 0; i < 15; ++i) {
            auto lost = pipeline->waitForFrameset(100);
            (void)lost;
        }

        auto startTime = std::chrono::steady_clock::now();
        auto duration  = std::chrono::seconds(5);  // 捕获时长 5 秒

        while (true) {
            auto elapsedTime = std::chrono::steady_clock::now() - startTime;
            if (elapsedTime >= duration) {
                std::cout << "5 seconds have passed, stopping frame capture." << std::endl;
                break;
            }

            std::shared_ptr<ob::FrameSet> frameSet;
            try {
                frameSet = pipeline->waitForFrameset(100);
            }
            catch (const ob::Error &e) {
                std::cerr << "[main] waitForFrameset error: "
                          << e.getMessage() << std::endl;
                continue;
            }


            if (!frameSet) {
                std::cout << "No frames received in 100ms..." << std::endl;
                continue;
            }


            // std::cout << "FrameSet types: " << frameSet->frameCount() << std::endl;



            // auto depth = frameSet->getFrame(OB_FRAME_DEPTH);
            // if (!depth) {
            //     std::cout << "[WARN] No DEPTH frame\n";
            //     continue;
            // }
            // auto depthFrame = depth->as<ob::DepthFrame>();


            auto depth = frameSet->getFrame(OB_FRAME_DEPTH);
            if(!depth){
                std::cout << "[WARN] No DEPTH frame\n";
            } else {
                auto depthFrame = depth->as<ob::DepthFrame>();

                saveImageTask([depthFrame, frameIndex]() {
                    // 异步保存深度（无需格式转换）
                    saveDepthFrame(depthFrame, frameIndex);
                });
            }

            auto color = frameSet->getFrame(OB_FRAME_COLOR);
            if (!color) {
                std::cout << "[WARN] No COLOR frame\n";
                continue;
            }
            auto colorFrame = color->as<ob::ColorFrame>();

            // 处理彩色帧（MJPG → RGB → BGR）
            try {
                // if (colorFrame->format() == OB_FORMAT_MJPG) {
                //     formatConverter->setFormatConvertType(FORMAT_MJPG_TO_RGB);
                //     colorFrame = formatConverter->process(colorFrame)->as<ob::ColorFrame>();
                // }

                // formatConverter->setFormatConvertType(FORMAT_RGB_TO_BGR);
                // colorFrame = formatConverter->process(colorFrame)->as<ob::ColorFrame>();
            }
            catch (const ob::Error &e) {
                std::cerr << "[main] FormatConvert error: "
                          << e.getMessage() << std::endl;
                continue;
            }

            // 可选显示
            // if (SHOW_IMAGE) {
            //     cv::Mat colorMat(colorFrame->height(), colorFrame->width(), CV_8UC3, colorFrame->data());
            //     cv::Mat depthMat(depthFrame->height(), depthFrame->width(), CV_16UC1, depthFrame->data());
            //     cv::imshow("Color Frame", colorMat);
            //     cv::imshow("Depth Frame", depthMat);

            //     if (cv::waitKey(1) == 27)
            //         break;
            // }

            // 异步保存
            // saveImageTask([depthFrame, frameIndex]() {
            //     saveDepthFrame(depthFrame, frameIndex);
            // });
            // saveImageTask([colorFrame, frameIndex]() {
            //     saveColorFrame(colorFrame, frameIndex);
            // });
            saveImageTask([colorFrame, frameIndex, formatConverter]() {
                auto frame = colorFrame;
                if (frame->format() == OB_FORMAT_MJPG) {
                    formatConverter->setFormatConvertType(FORMAT_MJPG_TO_RGB);
                    frame = formatConverter->process(frame)->as<ob::ColorFrame>();
                }
                formatConverter->setFormatConvertType(FORMAT_RGB_TO_BGR);
                frame = formatConverter->process(frame)->as<ob::ColorFrame>();
                saveColorFrame(frame, frameIndex);
            });

            frameIndex++;
        }

        // 停止 pipeline
        try {
            pipeline->stop();
        }
        catch (...) {
            std::cerr << "[main] pipeline->stop() threw exception\n";
        }

        // 停止工作线程
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            stopThread = true;
        }
        conditionVar.notify_one();
        worker.join();

        // 等待 stereo 线程
        if (stereoCfg.enable && stereoThread.joinable())
            stereoThread.join();

        std::cout << "Press any key to exit." << std::endl;
        ob_smpl::waitForKeyPressed();

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[main] Exception: " << e.what() << std::endl;
        return -1;
    }
    catch (...) {
        std::cerr << "[main] Unknown exception occurred." << std::endl;
        return -1;
    }
}

// ----------------- 保存函数 & 目录创建 -----------------

void saveDepthFrame(const std::shared_ptr<ob::DepthFrame> depthFrame,
                    const uint32_t frameIndex) {
    std::vector<int> params;
    params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    params.push_back(0);
    params.push_back(cv::IMWRITE_PNG_STRATEGY);
    params.push_back(cv::IMWRITE_PNG_STRATEGY_DEFAULT);

    std::string depthFolder = "depth/";
    createDirectory(depthFolder);

    std::string depthName = depthFolder + "Depth_" +
                            std::to_string(depthFrame->width()) + "x" +
                            std::to_string(depthFrame->height()) + "_" +
                            std::to_string(frameIndex) + "_" +
                            std::to_string(depthFrame->timeStamp()) + "ms.png";

    cv::Mat depthMat(depthFrame->height(), depthFrame->width(), CV_16UC1, depthFrame->data());
    try {
        cv::imwrite(depthName, depthMat, params);
    }
    catch (const cv::Exception &e) {
        std::cerr << "[saveDepthFrame] imwrite error: " << e.what() << std::endl;
    }

    std::cout << "Depth saved: " << depthName << std::endl;
}

void saveColorFrame(const std::shared_ptr<ob::ColorFrame> colorFrame,
                    const uint32_t frameIndex) {
    std::vector<int> params;
    params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    params.push_back(0);
    params.push_back(cv::IMWRITE_PNG_STRATEGY);
    params.push_back(cv::IMWRITE_PNG_STRATEGY_DEFAULT);

    std::string colorFolder = "color/";
    createDirectory(colorFolder);

    std::string colorName = colorFolder + "Color_" +
                            std::to_string(colorFrame->width()) + "x" +
                            std::to_string(colorFrame->height()) + "_" +
                            std::to_string(frameIndex) + "_" +
                            std::to_string(colorFrame->timeStamp()) + "ms.jpg";

    cv::Mat colorMat(colorFrame->height(), colorFrame->width(), CV_8UC3, colorFrame->data());
    try {
        cv::imwrite(colorName, colorMat, params);
    }
    catch (const cv::Exception &e) {
        std::cerr << "[saveColorFrame] imwrite error: " << e.what() << std::endl;
    }

    std::cout << "Color saved: " << colorName << std::endl;
}

void createDirectory(const std::string& path) {
    if (mkdir(path.c_str(), 0777) == -1) {
        // 已存在也会返回 -1，不必视为致命错误
        // std::cerr << "Failed to create directory: " << path << std::endl;
    } else {
        std::cout << "Directory created: " << path << std::endl;
    }
}
