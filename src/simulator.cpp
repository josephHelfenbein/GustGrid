#include <fluidSimulator.h>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <cstring>

extern void runFluidSimulation(
    int gridSizeX, int gridSizeY, int gridSizeZ,
    float* d_velocityField,
    unsigned char* d_solidGrid,
    float3* d_fanPositions,
    float3* d_fanDirections,
    int numFans,
    float dt
);

static inline int idx3D(int x, int y, int z, int gridSizeX, int gridSizeY){
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
const int numCells = (gridSizeX * gridSizeY * gridSizeZ);

int startSimulator(bool &gpuEnabled, bool &topFanEnabled, bool& cpuFanEnabled, bool &frontFanEnabled, float* backFanLocations, float* velocityField, bool& itemChanged){
    std::vector<unsigned char> h_solidGrid(numCells, 0);
    std::vector<float3> h_fanPositions;
    std::vector<float3> h_fanDirections;
    h_fanPositions.reserve(1 + 1 + 2 + 3);
    h_fanDirections.reserve(1 + 1 + 2 + 3);
    float3* d_fanPositions = nullptr;
    float3* d_fanDirections = nullptr;
    float* d_velocityField = nullptr;
    unsigned char* d_solidGrid = nullptr;
    size_t solidGridSize = numCells * sizeof(unsigned char);
    size_t velocityFieldSize = numCells * sizeof(float) * 3;
    int maxFanCount = 1 + 1 + 2 + 3;
    std::vector<float> h_velocity(NUM_CELLS * 3, 0.0f);
    size_t fanPositionsSize = maxFanCount * sizeof(float3);
    size_t fanDirectionsSize = maxFanCount * sizeof(float3);
    int numFans = 0;

    CUDA_CHECK(cudaMalloc(&d_velocityField, velocityFieldSize));
    CUDA_CHECK(cudaMemset(d_velocityField, 0, velocityFieldSize));

    CUDA_CHECK(cudaMalloc(&d_solidGrid, solidGridSize));

    CUDA_CHECK(cudaMalloc(&d_fanPositions, fanPositionsSize));
    CUDA_CHECK(cudaMalloc(&d_fanDirections, fanDirectionsSize));

    const float cellSizeX = 8.0f / gridSizeX;
    const float cellSizeY = 9.0f / gridSizeY;
    const float cellSizeZ = 4.0f / gridSizeZ;

    auto fillOccupancy = [&](){
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
                    if(gpuEnabled && ((worldX < 0.53f && worldX < 0.46f) || (worldX < -1.08f && worldX > -1.95) || worldX < -3.5) && worldY > 5.04f && worldY < 5.59f && worldZ < 0.5f) insideSolid = true;

                    h_solidGrid[index] = insideSolid ? 1 : 0;
                }
            }
        }
    };
    fillOccupancy();
    
    auto buildFanLists = [&](){
        h_fanPositions.clear();
        h_fanDirections.clear();
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

        CUDA_CHECK(cudaMemcpy(d_solidGrid, h_solidGrid.data(), solidGridSize, cudaMemcpyHostToDevice));

        numFans = static_cast<int>(h_fanPositions.size());
        if(numFans > 0) {
            CUDA_CHECK(cudaMemcpy(d_fanPositions, h_fanPositions.data(), fanPositionsSize, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_fanDirections, h_fanDirections.data(), fanDirectionsSize, cudaMemcpyHostToDevice));
        }
    }
    buildFanLists();
    
    bool prevGpuEnabled = gpuEnabled;
    bool prevTopFanEnabled = topFanEnabled;
    bool prevCpuFanEnabled = cpuFanEnabled;
    bool prevFrontFanEnabled = frontFanEnabled;
    float prevBackFanLocations[3] = {backFanLocations[0], backFanLocations[1], backFanLocations[2]};
    float dt = 0.01667f;
    while(!userRequestedExit()){
        if(itemChanged)
        if(!gpuEnabled || !prevGpuEnabled || !topFanEnabled || !prevTopFanEnabled || !cpuFanEnabled || !prevCpuFanEnabled || !frontFanEnabled || !prevFrontFanEnabled
           || backFanLocations[0] != prevBackFanLocations[0] || backFanLocations[1] != prevBackFanLocations[1] || backFanLocations[2] != prevBackFanLocations[2]){
            fillOccupancy();
            buildFanLists();
            prevGpuEnabled = gpuEnabled;
            prevTopFanEnabled = topFanEnabled;
            prevCpuFanEnabled = cpuFanEnabled;
            prevFrontFanEnabled = frontFanEnabled;
            for(int i=0; i<3; i++) prevBackFanLocations[i] = backFanLocations[i];
            itemChanged = false;
        }
        runFluidSimulation(
            gridSizeX, gridSizeY, gridSizeZ,
            d_velocityField,
            d_solidGrid,
            d_fanPositions,
            d_fanDirections,
            numFans,
            dt
        );
        CUDA_CHECK(cudaMemcpy(h_velocity.data(), d_velocityField, velocityFieldSize, cudaMemcpyDeviceToHost));
        std::memcpy(velocityField, h_velocity.data(), sizeof(float) * 3 * NUM_CELLS);
    }
    CUDA_CHECK(cudaFree(d_velocityField));
    CUDA_CHECK(cudaFree(d_solidGrid));
    CUDA_CHECK(cudaFree(d_fanPositions));
    CUDA_CHECK(cudaFree(d_fanDirections));
    return 0;
}