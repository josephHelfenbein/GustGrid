#pragma once
#include <cuda_runtime.h>

extern "C" void runFluidSimulation(
    unsigned char* d_solidGrid,
    float3* d_fanPositions,
    float3* d_fanDirections,
    float* d_heatSources,
    bool shouldResetFanAccess,
    int numFans,
    float dt,
    cudaGraphicsResource* volumeResource,
    cudaGraphicsResource* temperatureResource,
    bool displayPressure
);

extern "C" void initializeConstantsExtern(
    int gridSizeX, int gridSizeY, int gridSizeZ
);