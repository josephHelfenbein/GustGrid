#pragma once
#include <simulator.h>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

extern void runFluidSimulation(
    int gridSizeX, int gridSizeY, int gridSizeZ,
    float* d_velocityField,
    unsigned char* d_solidGrid,
    float3* d_fanPositions,
    float3* d_fanDirections,
    float dt
);

static inline int idx3D(int x, int y, int z, int gridSizeX, int gridSizeY) {
    return x + y * gridSizeX + z * gridSizeX * gridSizeY;
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr<<"CUDA error: "<<cudaGetErrorString(err)<<" in "<< __FILE__ <<" on line "<<__LINE__<<std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define gridSizeX 128
#define gridSizeY 128
#define gridSizeZ 128
#define numCells (gridSizeX * gridSizeY * gridSizeZ)

int startSimulator(bool &gpuEnabled, bool &topFanEnabled, bool& cpuFanEnabled, bool &frontFanEnabled, float* backFanLocations, float* velocityField){
    std::vector<unsigned char> h_solidGrid(numCells, 0);
    auto fillOccupancy = [&](){
        const float cellSizeX = 8.0f / gridSizeX;
        const float cellSizeY = 9.0f / gridSizeY;
        const float cellSizeZ = 4.0f / gridSizeZ;
        for(int z=0; z<gridSizeZ; z++){
            float worldZ = -2.0f + (z + 0.5f) * cellSizeZ;
            for(int y=0; y<gridSizeY; y++){
                float worldY = 0.0f + (y + 0.5f) * cellSizeY;
                for(int x=0; x<gridSizeX; x++){
                    float worldX = -4.0f + (x + 0.5f) * cellSizeX;
                    int index = idx3D(x, y, z, gridSizeX, gridSizeY);
                    bool insideSolid = false;

                    // case
                    if(worldY < 0.28f || (worldY > 8.9f && worldX < 3.65 && worldX > 0.65)) insideSolid = true;
                    if(worldZ < -1.8f || worldZ > 1.8f) insideSolid = true;
                    if(worldX < -3.8f && (worldY < 6.0f || worldZ < -0.6f || worldY > 8.3)) insideSolid = true;

                    // ram
                    if(worldZ < -0.95f && worldY > 5.7f && worldY < 8.10f && worldX > -0.6f && worldX < 0.18f) insideSolid = true;

                    // gpu
                    if(((worldX < 0.53f && worldX < 0.46f) || (worldX < -1.08f && worldX > -1.95) || worldX < -3.5) && worldY > 5.04f && worldY < 5.59f && worldZ < 0.5f) insideSolid = true;

                    h_solidGrid[index] = insideSolid ? 1 : 0;
                }
            }
        }
    };
    fillOccupancy();
    std::vector<float3> h_fanPositions;
    std::vector<float3> h_fanDirections;
    h_fanPositions.reserve(1 + 1 + 2 + 3);
    h_fanDirections.reserve(1 + 1 + 2 + 3);
    if(topFanEnabled){
        float3 topPos = make_float3(-1.6f, 8.7f, -0.22f);
        float3 topDir = make_float3(0.0f, -1.0f, 0.0f);
        h_fanPositions.push_back(topPos);
        h_fanDirections.push_back(topDir);
    }
    if(frontFanEnabled){
        float3 frontPos = make_float3(-3.5f, 7.1f, 0.48f);
        float3 frontDir = make_float3(-1.0f, 0.0f, 0.0f);
        h_fanPositions.push_back(frontPos);
        h_fanDirections.push_back(frontDir);
    }
    if(cpuFanEnabled){
        float3 cpuPos = make_float3(-0.85f, 6.9f, 0.1f);
        float3 cpuDir = make_float3(-1.0f, 0.0f, 0.0f);
        h_fanPositions.push_back(cpuPos);
        h_fanDirections.push_back(cpuDir);
    }
    if(gpuEnabled){
        float3 gpu1Pos = make_float3(-0.34f, 5.3f, -0.36f);
        float3 gpu2Pos = make_float3(-2.71f, 5.3f, -0.36f);
        float3 gpuDir = make_float3(0.0f, 1.0f, 0.0f);
        h_fanPositions.push_back(gpu1Pos);
        h_fanPositions.push_back(gpu2Pos);
        h_fanDirections.push_back(gpuDir);
        h_fanDirections.push_back(gpuDir);
    }
    for(int i=0; i<3; i++){
        if(backFanLocations[i] == 1.0f) continue;
        float3 backFanPos = {3.35f, 7.0f - backFanLocations[i], 0.0f};
        float3 backFanDir = {-1.0f, 0.0f, 0.0f};
        h_fanPositions.push_back(backFanPos);
        h_fanDirections.push_back(backFanDir);
    }

    float* d_velocityField = nullptr;
    size_t velocityFieldSize = numCells * sizeof(float) * 3;
    CUDA_CHECK(cudaMalloc(&d_velocityField, velocityFieldSize));
    CUDA_CHECK(cudaMemset(d_velocityField, 0, velocityFieldSize));

    unsigned char* d_solidGrid = nullptr;
    size_t solidGridSize = numCells * sizeof(unsigned char);
    CUDA_CHECK(cudaMalloc(&d_solidGrid, solidGridSize));
    CUDA_CHECK(cudaMemcpy(d_solidGrid, h_solidGrid.data(), solidGridSize, cudaMemcpyHostToDevice));

    float3* d_fanPositions = nullptr;
    size_t fanPositionsSize = h_fanPositions.size() * sizeof(float3);
    if(h_fanPositions.size() > 0) {
        CUDA_CHECK(cudaMalloc(&d_fanPositions, fanPositionsSize));
        CUDA_CHECK(cudaMemcpy(d_fanPositions, h_fanPositions.data(), fanPositionsSize, cudaMemcpyHostToDevice));
    }
    float3* d_fanDirections = nullptr;
    size_t fanDirectionsSize = h_fanDirections.size() * sizeof(float3);
    if(h_fanDirections.size() > 0) {
        CUDA_CHECK(cudaMalloc(&d_fanDirections, fanDirectionsSize));
        CUDA_CHECK(cudaMemcpy(d_fanDirections, h_fanDirections.data(), fanDirectionsSize, cudaMemcpyHostToDevice));
    }
    float dt = 0.01667f;
    while(!userRequestedExit()){
        runFluidSimulation(
            gridSizeX, gridSizeY, gridSizeZ,
            d_velocityField,
            d_solidGrid,
            d_fanPositions,
            d_fanDirections
            dt
        );
        velocityField = new float[numCells * 3];
        CUDA_CHECK(cudaMemcpy(velocityField, d_velocityField, velocityFieldSize, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(d_velocityField));
    CUDA_CHECK(cudaFree(d_solidGrid));
    if(h_fanPositions.size() > 0) CUDA_CHECK(cudaFree(d_fanPositions));
    if(h_fanDirections.size() > 0) CUDA_CHECK(cudaFree(d_fanDirections));
    return 0;
}