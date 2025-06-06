#pragma once
#include <cuda_runtime.h>
#include <functional>

int startSimulator(bool &gpuEnabled, bool &topFanEnabled, bool& cpuFanEnabled, bool &frontFanEnabled, float* backFanLocations, float* volumeField, bool& itemChanged, bool& running, std::function<void()> signalVelocityFieldReady, std::function<void()> waitForItems, bool &displayPressure, float* temperatureField, double& stepsPerSecond);