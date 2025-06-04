#include <functional>

int startRenderer(bool &gpuEnabled, bool &topFanEnabled, bool& cpuFanEnabled, bool &frontFanEnabled, float* backFanLocations, float* velocityField, float* pressureField, bool &itemChanged, bool &running, std::function<void()> waitForVelocityField, std::function<void()> signalItemsReady, bool &displayPressure);