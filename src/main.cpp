#include <renderer.h>
#include <simulator.h>

int main(int argc, char* argv[]){
    bool gpuEnabled = true;
    bool topFanEnabled = true;
    bool cpuFanEnabled = true;
    bool frontFanEnabled = true;
    float backFanLocations[3] = {0.0f, -2.5f, 1.0f};
    float* velocityField = nullptr;
    bool itemChanged = false;
    startRenderer(gpuEnabled, topFanEnabled, cpuFanEnabled, frontFanEnabled, backFanLocations, velocityField, itemChanged);
    startSimulator(gpuEnabled, topFanEnabled, cpuFanEnabled, frontFanEnabled, backFanLocations, velocityField, itemChanged);
    return 0;
}