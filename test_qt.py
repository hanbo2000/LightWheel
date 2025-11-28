// Copyright (c) Orbbec Inc. All Rights Reserved.
// Licensed under the MIT License.

#include <libobsensor/ObSensor.hpp>
#include "PipelineHolder.hpp"
#include "FramePairingManager.hpp"
#include "utils.hpp"
#include "utils_opencv.hpp"
#include "utils/cJSON.h"

#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <chrono>
#include <queue>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <filesystem>
namespace fs = std::filesystem;

#define MAX_DEVICE_COUNT 9
#define CONFIG_FILE "./MultiDeviceSyncConfig.json"
#define KEY_ESC 27

template<typename T>
class SafeQueue {
public:
    void push(const T &t) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            q.push(t);
        }
        cv.notify_one();
    }

    bool pop(T &t) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&]{ return !q.empty(); });

        t = q.front();
        q.pop();

        // 哨兵任务：如果 T 是 “空任务”，说明线程需要退出
        return !(t.exitSignal);
    }

    // 向队列中塞入哨兵任务，使线程退出
    void shutdown() {
        T t;
        t.exitSignal = true;   // 设置退出标记
        push(t);
    }

private:
    std::queue<T> q;
    std::mutex mtx;
    std::condition_variable cv;
};

static bool     quitStreamPreview      = false;

void setupTerminalRawMode() {
    termios t;

    // 获取当前终端参数
    tcgetattr(STDIN_FILENO, &t);

    // 修改参数：关闭 canonical 模式 & 回显
    t.c_lflag &= ~(ICANON | ECHO);

    // 设置立即生效
    tcsetattr(STDIN_FILENO, TCSANOW, &t);

    // 设置stdin为非阻塞
    fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
}

void escListener() {
    setupTerminalRawMode();

    while(true) {
        int ch = getchar();
        if(ch == 27) {  // ESC ASCII = 27
            std::cout << "\n[ESC] Exit requested\n";
            quitStreamPreview = true;
            return;
        }
        usleep(1000); // 防止占用CPU过高
    }
}




// static bool waitingForSave = false;

typedef struct DeviceConfigInfo_t {
    std::string             deviceSN;
    OBMultiDeviceSyncConfig syncConfig;
} DeviceConfigInfo;
struct SaveTask {
    bool exitSignal = false;  // true 表示是一个“退出信号任务”
    int deviceID = -1;        // 默认给 deviceID 赋值
    uint64_t timestamp = 0;   // 给 timestamp 提供一个默认值
    std::shared_ptr<ob::DepthFrame> depth;
    std::shared_ptr<ob::ColorFrame> color;
};

SafeQueue<SaveTask> saveQueue;
std::thread saveThread;

std::vector<std::shared_ptr<ob::Device>>       streamDevList;
std::vector<std::shared_ptr<ob::Device>>       configDevList;
std::vector<std::shared_ptr<DeviceConfigInfo>> deviceConfigList;

std::condition_variable                      waitRebootCompleteCondition;
std::mutex                                   rebootingDevInfoListMutex;
std::vector<std::shared_ptr<ob::DeviceInfo>> rebootingDevInfoList;
std::vector<std::shared_ptr<PipelineHolder>> pipelineHolderList;

bool loadConfigFile();
int  configMultiDeviceSync();
int  testMultiDeviceSync();


std::string           OBSyncModeToString(const OBMultiDeviceSyncMode syncMode);
OBMultiDeviceSyncMode stringToOBSyncMode(const std::string &modeString);

std::string readFileContent(const char *filePath);

int  strcmp_nocase(const char *str0, const char *str1);
bool checkDevicesWithDeviceConfigs(const std::vector<std::shared_ptr<ob::Device>> &deviceList);

std::shared_ptr<PipelineHolder> createPipelineHolder(std::shared_ptr<ob::Device> device, OBSensorType sensorType, int deviceIndex);

ob::Context context;
bool isTriggered(const std::shared_ptr<ob::DepthFrame>& depth) {
    // 这里假设我们通过时间戳来判断是否为触发帧，你可以根据实际需要调整
    static uint64_t lastTimestamp = 0;
    uint64_t currentTimestamp = depth->timeStampUs();

    if (currentTimestamp - lastTimestamp > 10000) {  
        lastTimestamp = currentTimestamp;
        return true;
    }
    return false;
}

int main(void) try {
    int                       choice;
    int                       exitValue      = 0;
    constexpr std::streamsize maxInputIgnore = 10000;

    while(true) {
        std::cout << "\n--------------------------------------------------\n";
        std::cout << "Please select options: \n";
        std::cout << " 0 --> config devices sync mode. \n";
        std::cout << " 1 --> start stream \n";
        std::cout << "--------------------------------------------------\n";
        std::cout << "Please select input: ";
        // std::cin >> choice;
        if(!(std::cin >> choice)) {
            std::cin.clear();
            std::cin.ignore(maxInputIgnore, '\n');
            std::cout << "Invalid input. Please enter a number [0~1]" << std::endl;
            continue;
        }
        std::cout << std::endl;

        switch(choice) {
        case 0:
            exitValue = configMultiDeviceSync();
            if(exitValue == 0) {
                std::cout << "Config MultiDeviceSync Success. \n" << std::endl;

                exitValue = testMultiDeviceSync();
            }
            break;
        case 1:
            std::cout << "\nStart Devices video stream." << std::endl;
            exitValue = testMultiDeviceSync();
            break;
        default:
            break;
        }

        if(exitValue == 0) {
            break;
        }
    }
    return exitValue;
}
catch(ob::Error &e) {
    std::cerr << "function:" << e.getFunction() << "\nargs:" << e.getArgs() << "\nmessage:" << e.what() << "\ntype:" << e.getExceptionType() << std::endl;
    std::cout << "\nPress any key to exit.";
    ob_smpl::waitForKeyPressed();
    exit(EXIT_FAILURE);
}

int configMultiDeviceSync() {
    try {
        if(!loadConfigFile()) {
            std::cout << "load config failed" << std::endl;
            return -1;
        }

        if(deviceConfigList.empty()) {
            std::cout << "DeviceConfigList is empty. please check config file: " << CONFIG_FILE << std::endl;
            return -1;
        }

        // Query the list of connected devices
        auto devList  = context.queryDeviceList();
        int  devCount = devList->deviceCount();
        for(int i = 0; i < devCount; i++) {
            std::shared_ptr<ob::Device> device = devList->getDevice(i);
            configDevList.push_back(devList->getDevice(i));
        }

        if(configDevList.empty()) {
            std::cerr << "Device list is empty. please check device connection state" << std::endl;
            return -1;
        }

        // write configuration to device
        for(auto config: deviceConfigList) {
            auto findItr = std::find_if(configDevList.begin(), configDevList.end(), [config](std::shared_ptr<ob::Device> device) {
                auto serialNumber = device->getDeviceInfo()->serialNumber();
                return strcmp_nocase(serialNumber, config->deviceSN.c_str()) == 0;
            });
            if(findItr != configDevList.end()) {
                auto device    = (*findItr);
                auto curConfig = device->getMultiDeviceSyncConfig();
                // Update the configuration items of the configuration file, and keep the original configuration for other items
                curConfig.syncMode             = config->syncConfig.syncMode;
                curConfig.depthDelayUs         = config->syncConfig.depthDelayUs;
                curConfig.colorDelayUs         = config->syncConfig.colorDelayUs;
                curConfig.trigger2ImageDelayUs = config->syncConfig.trigger2ImageDelayUs;
                curConfig.triggerOutEnable     = config->syncConfig.triggerOutEnable;
                curConfig.triggerOutDelayUs    = config->syncConfig.triggerOutDelayUs;
                curConfig.framesPerTrigger     = config->syncConfig.framesPerTrigger;
                std::cout << "-Config Device syncMode:" << curConfig.syncMode << ", syncModeStr:" << OBSyncModeToString(curConfig.syncMode) << std::endl;
                device->setMultiDeviceSyncConfig(curConfig);

                curConfig = device->getMultiDeviceSyncConfig();
                std::cout << "After saving, sync mode: " << OBSyncModeToString(curConfig.syncMode) << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return 0;
    }
    catch(ob::Error &e) {
        std::cerr << "configMultiDeviceSync failed! \n";
        std::cerr << "function:" << e.getName() << "\nargs:" << e.getArgs() << "\nmessage:" << e.getMessage() << "\ntype:" << e.getExceptionType() << std::endl;
        return -1;
    }
}





void saveDepth(const std::shared_ptr<ob::DepthFrame> &frame, int deviceID, uint64_t timestamp){
    char filename[256];
    sprintf(filename, "output/device%d_depth_%llu.png", deviceID, (unsigned long long)timestamp);

    cv::Mat depthMat(frame->height(), frame->width(), CV_16UC1, frame->data());
    cv::imwrite(filename, depthMat);
    std::cout << "[SAVE] " << filename << std::endl;
}

void saveColor(const std::shared_ptr<ob::ColorFrame>& frame,
               const std::string& basePath)
{
    fs::create_directories(fs::path(basePath).parent_path());

    if(frame->format() != OB_FORMAT_MJPG) {
        std::cout << "[WARN] Color format not MJPG, skip.\n";
        return;
    }

    std::string rawPath = basePath + ".jpg";

    const uint8_t* data = reinterpret_cast<const uint8_t*>(frame->data());
    uint32_t size = frame->dataSize();

    std::ofstream fout(rawPath, std::ios::binary);
    fout.write(reinterpret_cast<const char*>(data), size);
    fout.close();
}


// ======================= ADD: Background Worker =======================
void startDeviceStreams(const std::vector<std::shared_ptr<ob::Device>> &devices, int startIndex) {
    std::vector<OBSensorType> sensorTypes = { OB_SENSOR_DEPTH, OB_SENSOR_COLOR };
    for (auto &dev : devices) {
        for (auto sensorType : sensorTypes) {
            auto holder = createPipelineHolder(dev, sensorType, startIndex);
            pipelineHolderList.push_back(holder);
            holder->startStream();
        }
        startIndex++;
    }
    quitStreamPreview = false;
}

void saveWorker() {
    SaveTask task;

    if (!fs::exists("output/depth")) {
        fs::create_directories("output/depth");
    }
    if (!fs::exists("output/color")) {
        fs::create_directories("output/color");
    }

    while (true) {
        if (!saveQueue.pop(task))
            break;
        if (task.exitSignal)
            break;

        uint64_t ts = task.timestamp;

        // 只有在外部触发后才保存图像
        if (isTriggered(task.depth)) { // 判断帧是否由硬件触发
            // --- 深度图路径 ---
            std::string depthPath =
                "output/depth/device" + std::to_string(task.deviceID) +
                "_ts" + std::to_string(ts) + ".png";

            // --- 彩色图路径 ---
            std::string colorBase =
                "output/color/device" + std::to_string(task.deviceID) +
                "_ts" + std::to_string(ts);

            // ===== 保存深度 =====
            cv::Mat depthMat(task.depth->height(), task.depth->width(),
                             CV_16UC1, task.depth->data());
            cv::imwrite(depthPath, depthMat);

            // ===== 保存彩色 =====
            saveColor(task.color, colorBase);

            std::cout << "[SAVE] dev=" << task.deviceID
                      << " ts=" << ts << std::endl;
        }
    }

    std::cout << "[SAVE THREAD] Worker exited.\n";
}


int testMultiDeviceSync() {
    // 启动后台保存线程
    saveThread = std::thread(saveWorker);

    // 启动 ESC 按键监听线程，用于随时退出循环
    std::thread escThread(escListener);

    try {
        // 获取设备列表
        streamDevList.clear();
        auto devList  = context.queryDeviceList();
        int  devCount = devList->deviceCount();

        for (int i = 0; i < devCount; i++)
            streamDevList.push_back(devList->getDevice(i));

        if (streamDevList.empty()) {
            std::cerr << "No devices found.\n";
            quitStreamPreview = true;
            saveQueue.shutdown();
            if (saveThread.joinable()) saveThread.join();
            if (escThread.joinable()) escThread.detach();
            return -1;
        }

        // 启动所有设备的流
        startDeviceStreams(streamDevList, 0);

        // 等待设备流稳定
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        // 尝试同步设备时钟（如果设备支持）
        try {
            context.enableDeviceClockSync(60000);
            std::cout << "[INFO] Device clock sync enabled.\n";
        } catch (...) {
            std::cout << "[WARN] Device clock sync not supported or failed.\n";
        }

        // 创建 FramePairingManager
        auto pairing = std::make_shared<FramePairingManager>();
        pairing->setPipelineHolderList(pipelineHolderList);

        std::cout << "\nWaiting for HARDWARE TRIGGER signals...\n";
        std::cout << "Press [ESC] to stop saving and exit.\n";

        // 主循环：等待触发的帧
        while (!quitStreamPreview) {
            // 获取同步帧组
            auto pairs = pairing->getFramePairs();

            if (pairs.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            for (size_t i = 0; i < pairs.size(); i++) {
                if (!pairs[i].first || !pairs[i].second) continue;

                auto depth = pairs[i].first->as<ob::DepthFrame>();
                auto color = pairs[i].second->as<ob::ColorFrame>();

                // 计算时间戳 (ms)
                uint64_t ts = depth->timeStampUs() / 1000;

                // 塞入保存队列
                SaveTask t;
                t.deviceID  = i;
                t.timestamp = ts;
                t.depth     = depth;
                t.color     = color;

                // 只在触发帧时保存
                if (isTriggered(depth)) {
                    saveQueue.push(t);
                }
            }
        }

        std::cout << "\n[INFO] Stop requested. Cleaning up...\n";

        // 退出清理
        saveQueue.shutdown();
        if (saveThread.joinable()) saveThread.join();
        if (escThread.joinable()) escThread.join();

        // 停止流
        for (auto &holder : pipelineHolderList) {
            holder->stopStream();
        }
        pipelineHolderList.clear();

        return 0;
    } catch (ob::Error &e) {
        std::cerr << "Error: " << e.getMessage() << "\n";
        quitStreamPreview = true;
        saveQueue.shutdown();
        if (saveThread.joinable()) saveThread.join();
        if (escThread.joinable()) escThread.detach();
        return -1;
    }
}

std::shared_ptr<PipelineHolder> createPipelineHolder(std::shared_ptr<ob::Device> device, OBSensorType sensorType, int deviceIndex) {
    auto pipeline    = std::make_shared<ob::Pipeline>(device);
    auto holder         = std::make_shared<PipelineHolder>(pipeline, sensorType, device->getDeviceInfo()->serialNumber(), deviceIndex);
    return holder;
}

std::string readFileContent(const char *filePath) {
    std::ostringstream oss;
    std::ifstream      file(filePath, std::fstream::in);
    if(!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return "";
    }
    oss << file.rdbuf();
    file.close();
    return oss.str();
}

bool loadConfigFile() {
    int                               deviceCount   = 0;
    std::shared_ptr<DeviceConfigInfo> devConfigInfo = nullptr;
    cJSON                            *deviceElem    = nullptr;

    auto content = readFileContent(CONFIG_FILE);
    if(content.empty()) {
        std::cerr << "load config file failed." << std::endl;
        return false;
    }

    cJSON *rootElem = cJSON_Parse(content.c_str());
    if(rootElem == nullptr) {
        const char *errMsg = cJSON_GetErrorPtr();
        std::cout << std::string(errMsg) << std::endl;
        cJSON_Delete(rootElem);
        return true;
    }

    cJSON *devicesElem = cJSON_GetObjectItem(rootElem, "devices");
    cJSON_ArrayForEach(deviceElem, devicesElem) {
        devConfigInfo = std::make_shared<DeviceConfigInfo>();
        memset(&devConfigInfo->syncConfig, 0, sizeof(devConfigInfo->syncConfig));
        devConfigInfo->syncConfig.syncMode = OB_MULTI_DEVICE_SYNC_MODE_FREE_RUN;

        cJSON *snElem = cJSON_GetObjectItem(deviceElem, "sn");
        if(cJSON_IsString(snElem) && snElem->valuestring != nullptr) {
            devConfigInfo->deviceSN = std::string(snElem->valuestring);
        }
        cJSON *deviceConfigElem = cJSON_GetObjectItem(deviceElem, "syncConfig");
        if(cJSON_IsObject(deviceConfigElem)) {
            cJSON *numberElem = nullptr;
            cJSON *strElem    = nullptr;
            cJSON *bElem      = nullptr;
            strElem           = cJSON_GetObjectItemCaseSensitive(deviceConfigElem, "syncMode");
            if(cJSON_IsString(strElem) && strElem->valuestring != nullptr) {
                devConfigInfo->syncConfig.syncMode = stringToOBSyncMode(strElem->valuestring);
                std::cout << "config[" << (deviceCount++) << "]: SN=" << std::string(devConfigInfo->deviceSN) << ", mode=" << strElem->valuestring << std::endl;
            }
            numberElem = cJSON_GetObjectItemCaseSensitive(deviceConfigElem, "depthDelayUs");
            if(cJSON_IsNumber(numberElem)) {
                devConfigInfo->syncConfig.depthDelayUs = numberElem->valueint;
            }
            numberElem = cJSON_GetObjectItemCaseSensitive(deviceConfigElem, "colorDelayUs");
            if(cJSON_IsNumber(numberElem)) {
                devConfigInfo->syncConfig.colorDelayUs = numberElem->valueint;
            }
            numberElem = cJSON_GetObjectItemCaseSensitive(deviceConfigElem, "trigger2ImageDelayUs");
            if(cJSON_IsNumber(numberElem)) {
                devConfigInfo->syncConfig.trigger2ImageDelayUs = numberElem->valueint;
            }
            numberElem = cJSON_GetObjectItemCaseSensitive(deviceConfigElem, "triggerOutDelayUs");
            if(cJSON_IsNumber(numberElem)) {
                devConfigInfo->syncConfig.triggerOutDelayUs = numberElem->valueint;
            }
            bElem = cJSON_GetObjectItemCaseSensitive(deviceConfigElem, "triggerOutEnable");
            if(cJSON_IsBool(bElem)) {
                devConfigInfo->syncConfig.triggerOutEnable = (bool)bElem->valueint;
            }
            bElem = cJSON_GetObjectItemCaseSensitive(deviceConfigElem, "framesPerTrigger");
            if(cJSON_IsNumber(bElem)) {
                devConfigInfo->syncConfig.framesPerTrigger = bElem->valueint;
            }
        }

        if(OB_MULTI_DEVICE_SYNC_MODE_FREE_RUN != devConfigInfo->syncConfig.syncMode) {
            deviceConfigList.push_back(devConfigInfo);
        }
        else {
            std::cerr << "Invalid sync mode of deviceSN: " << devConfigInfo->deviceSN << std::endl;
        }
        devConfigInfo = nullptr;
    }
    cJSON_Delete(rootElem);
    return true;
}

OBMultiDeviceSyncMode stringToOBSyncMode(const std::string &modeString) {
    static const std::unordered_map<std::string, OBMultiDeviceSyncMode> syncModeMap = {
        { "OB_MULTI_DEVICE_SYNC_MODE_FREE_RUN", OB_MULTI_DEVICE_SYNC_MODE_FREE_RUN },
        { "OB_MULTI_DEVICE_SYNC_MODE_STANDALONE", OB_MULTI_DEVICE_SYNC_MODE_STANDALONE },
        { "OB_MULTI_DEVICE_SYNC_MODE_PRIMARY", OB_MULTI_DEVICE_SYNC_MODE_PRIMARY },
        { "OB_MULTI_DEVICE_SYNC_MODE_SECONDARY", OB_MULTI_DEVICE_SYNC_MODE_SECONDARY },
        { "OB_MULTI_DEVICE_SYNC_MODE_SECONDARY_SYNCED", OB_MULTI_DEVICE_SYNC_MODE_SECONDARY_SYNCED },
        { "OB_MULTI_DEVICE_SYNC_MODE_SOFTWARE_TRIGGERING", OB_MULTI_DEVICE_SYNC_MODE_SOFTWARE_TRIGGERING },
        { "OB_MULTI_DEVICE_SYNC_MODE_HARDWARE_TRIGGERING", OB_MULTI_DEVICE_SYNC_MODE_HARDWARE_TRIGGERING }
    };
    auto it = syncModeMap.find(modeString);
    if(it != syncModeMap.end()) {
        return it->second;
    }
    // Constructing exception messages with stringstream
    std::stringstream ss;
    ss << "Unrecognized sync mode: " << modeString;
    throw std::invalid_argument(ss.str());
}

std::string OBSyncModeToString(const OBMultiDeviceSyncMode syncMode) {
    static const std::unordered_map<OBMultiDeviceSyncMode, std::string> modeToStringMap = {
        { OB_MULTI_DEVICE_SYNC_MODE_FREE_RUN, "OB_MULTI_DEVICE_SYNC_MODE_FREE_RUN" },
        { OB_MULTI_DEVICE_SYNC_MODE_STANDALONE, "OB_MULTI_DEVICE_SYNC_MODE_STANDALONE" },
        { OB_MULTI_DEVICE_SYNC_MODE_PRIMARY, "OB_MULTI_DEVICE_SYNC_MODE_PRIMARY" },
        { OB_MULTI_DEVICE_SYNC_MODE_SECONDARY, "OB_MULTI_DEVICE_SYNC_MODE_SECONDARY" },
        { OB_MULTI_DEVICE_SYNC_MODE_SECONDARY_SYNCED, "OB_MULTI_DEVICE_SYNC_MODE_SECONDARY_SYNCED" },
        { OB_MULTI_DEVICE_SYNC_MODE_SOFTWARE_TRIGGERING, "OB_MULTI_DEVICE_SYNC_MODE_SOFTWARE_TRIGGERING" },
        { OB_MULTI_DEVICE_SYNC_MODE_HARDWARE_TRIGGERING, "OB_MULTI_DEVICE_SYNC_MODE_HARDWARE_TRIGGERING" }
    };

    auto it = modeToStringMap.find(syncMode);
    if(it != modeToStringMap.end()) {
        return it->second;
    }
    std::stringstream ss;
    ss << "Unmapped sync mode value: " << static_cast<int>(syncMode) << ". Please check the sync mode value.";
    throw std::invalid_argument(ss.str());
}

int strcmp_nocase(const char *str0, const char *str1) {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    return _strcmpi(str0, str1);
#else
    return strcasecmp(str0, str1);
#endif
}
