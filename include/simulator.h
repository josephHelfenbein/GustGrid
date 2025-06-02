#pragma once
#ifndef SIMULATOR_H
#define SIMULATOR_H
#include <cuda_runtime.h>

int startSimulator(bool &gpuEnabled, bool &topFanEnabled, bool& cpuFanEnabled, bool &frontFanEnabled, float* backFanLocations, float* velocityField, bool& itemChanged);

#endif
