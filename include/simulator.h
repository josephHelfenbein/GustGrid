#pragma once
#include <cuda_runtime.h>
#include <functional>

void setupSimulator(bool &gpuEnabled, bool &topFanEnabled, bool &cpuFanEnabled, bool &frontFanEnabled, float* backFanLocations, unsigned char** d_solidGrid, float3** d_fanPositions, float3** d_fanDirections, float** d_heatSources, int &numFans);