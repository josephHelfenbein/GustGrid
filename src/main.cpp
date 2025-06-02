#include <renderer.h>
#include <simulator.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>

bool gpuEnabled = true;
bool topFanEnabled = true;
bool cpuFanEnabled = true;
bool frontFanEnabled = true;
float backFanLocations[3] = {0.0f, -2.5f, 1.0f};

const int gridSizeX = 128;
const int gridSizeY = 128;
const int gridSizeZ = 128;
const int numCells = gridSizeX * gridSizeY * gridSizeZ;
float* velocityField = new float[numCells * 3]();
bool itemChanged = false;
bool running = true;

std::mutex velocityFieldMutex;
std::condition_variable velocityFieldReady;
std::condition_variable itemsReady;
std::atomic<bool> velocityFieldInitialized{false};
std::atomic<bool> itemsReadyFlag{false};

void signalVelocityFieldReady(){
    std::lock_guard<std::mutex> lock(velocityFieldMutex);
    velocityFieldInitialized.store(true);
    velocityFieldReady.notify_all();
}
void waitForVelocityField(){
    std::unique_lock<std::mutex> lock(velocityFieldMutex);
    velocityFieldReady.wait(lock, []{ return velocityFieldInitialized.load(); });
}
void signalItemsReady(){
    std::lock_guard<std::mutex> lock(velocityFieldMutex);
    itemsReadyFlag.store(true);
    itemsReady.notify_all();
}
void waitForItems(){
    std::unique_lock<std::mutex> lock(velocityFieldMutex);
    itemsReady.wait(lock, []{ return itemsReadyFlag.load(); });
}
int rendererWrapper(){
    return startRenderer(gpuEnabled, topFanEnabled, cpuFanEnabled, frontFanEnabled, backFanLocations, velocityField, itemChanged, running, waitForVelocityField, signalItemsReady);
}
int simulatorWrapper(){
    return startSimulator(gpuEnabled, topFanEnabled, cpuFanEnabled, frontFanEnabled, backFanLocations, velocityField, itemChanged, running, signalVelocityFieldReady, waitForItems);
}

int main(int argc, char* argv[]){
    std::thread simulatorThread(simulatorWrapper);
    std::thread rendererThread(rendererWrapper);
    simulatorThread.join();
    rendererThread.join();
    delete[] velocityField;
    return 0;
}