#pragma once
#include <cuda_runtime.h>

extern "C" void runFluidSimulation(
    int gridSizeX, int gridSizeY, int gridSizeZ,
    float* d_velocityField,
    float* d_pressureField,
    unsigned char* d_solidGrid,
    float3* d_fanPositions,
    float3* d_fanDirections,
    float* d_heatSources,
    float* d_temperature,
    int numFans,
    float dt
);

extern "C" void initializeConstantsExtern(
    int gridSizeX, int gridSizeY, int gridSizeZ
);