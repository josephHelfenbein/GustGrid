#include <renderer.h>

int main(int argc, char* argv[]){
    bool gpuEnabled = true;
    bool topFanEnabled = true;
    bool cpuFanEnabled = true;
    bool frontFanEnabled = true;
    float backFanLocations[3] = {0.0f, -2.5f, 1.0f};
    return startRenderer(gpuEnabled, topFanEnabled, cpuFanEnabled, topFanEnabled, backFanLocations);
}