#pragma once
#include <cuda_runtime.h>
#include <functional>

int startSimulator(bool &gpuEnabled, bool &topFanEnabled, bool& cpuFanEnabled, bool &frontFanEnabled, float* backFanLocations, float* velocityField, bool& itemChanged, bool& running, std::function<void()> signalVelocityFieldReady, std::function<void()> waitForItems);