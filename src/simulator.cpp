#include <fluidSimulator.h>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#include <functional>

extern void runFluidSimulation(
    int gridSizeX, int gridSizeY, int gridSizeZ,
    float* d_velocityField,
    float* d_pressureField,
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

#define gridSizeX 64
#define gridSizeY 256
#define gridSizeZ 128
const int numCells = (gridSizeX * gridSizeY * gridSizeZ);

int startSimulator(bool &gpuEnabled, bool &topFanEnabled, bool &cpuFanEnabled, bool &frontFanEnabled, float* backFanLocations, float* velocityField, float* pressureField, bool &itemChanged, bool &running, std::function<void()> signalVelocityFieldReady, std::function<void()> waitForItems, bool &displayPressure, float* temperatureField){
    waitForItems();
    std::vector<unsigned char> h_solidGrid(numCells, 0);
    std::vector<float> h_heatSources(numCells, 0.0f);
    std::vector<float3> h_fanPositions;
    std::vector<float3> h_fanDirections;
    int maxFanCount = 1 + 1 + 1 + 2 + 3;
    h_fanPositions.reserve(maxFanCount);
    h_fanDirections.reserve(maxFanCount);
    float3* d_fanPositions = nullptr;
    float3* d_fanDirections = nullptr;
    float* d_velocityField = nullptr;
    float* d_pressureField = nullptr;
    float* d_heatSources = nullptr;
    float* d_temperature = nullptr;
    unsigned char* d_solidGrid = nullptr;
    size_t solidGridSize = numCells * sizeof(unsigned char);
    size_t velocityFieldSize = numCells * sizeof(float) * 3;
    size_t pressureFieldSize = numCells * sizeof(float);
    std::vector<float> h_velocity(numCells * 3, 0.0f);
    std::vector<float> h_pressure(numCells, 0.0f);
    std::vector<float> h_temperature(numCells, 0.0f);
    size_t fanPositionsSize = maxFanCount * sizeof(float3);
    size_t fanDirectionsSize = maxFanCount * sizeof(float3);
    int numFans = 0;

    CUDA_CHECK(cudaMalloc(&d_velocityField, velocityFieldSize));
    CUDA_CHECK(cudaMemset(d_velocityField, 0, velocityFieldSize));
    CUDA_CHECK(cudaMalloc(&d_pressureField, pressureFieldSize));
    CUDA_CHECK(cudaMemset(d_pressureField, 0, pressureFieldSize));
    CUDA_CHECK(cudaMalloc(&d_heatSources, pressureFieldSize));
    CUDA_CHECK(cudaMemset(d_heatSources, 0, pressureFieldSize));
    CUDA_CHECK(cudaMalloc(&d_temperature, pressureFieldSize));
    CUDA_CHECK(cudaMemset(d_temperature, 0, pressureFieldSize));

    CUDA_CHECK(cudaMalloc(&d_solidGrid, solidGridSize));

    CUDA_CHECK(cudaMalloc(&d_fanPositions, fanPositionsSize));
    CUDA_CHECK(cudaMalloc(&d_fanDirections, fanDirectionsSize));

    const float cellSizeX = 4.0f / gridSizeX;
    const float cellSizeY = 9.0f / gridSizeY;
    const float cellSizeZ = 8.0f / gridSizeZ;

    auto fillOccupancy = [&](){
        for(int z=0; z<gridSizeZ; z++){
            float worldZ = -4.0f + (z + 0.5f) * cellSizeZ;
            for(int y=0; y<gridSizeY; y++){
                float worldY = -4.5f + (y + 0.5f) * cellSizeY;
                for(int x=0; x<gridSizeX; x++){
                    float worldX = -2.0f + (x + 0.5f) * cellSizeX;
                    int index = idx3D(x, y, z, gridSizeX, gridSizeY);
                    bool insideSolid = false;
                    float heatSource = 0.0f;

                    // case
                    if((worldY < -4.22f && worldZ > 0.5f) || (worldZ > -1.8f && worldY < -4.26f && worldY > -4.22f) || (worldY > 4.4f && worldZ > -3.65 && worldZ < -0.65)) insideSolid = true;
                    if(worldX < -1.8f || worldX > 1.8f) insideSolid = true;
                    if(worldZ > 3.8f && (worldY < 1.5f || worldX < -0.6f || worldY > 3.8f)) insideSolid = true;

                    // gpu
                    if(gpuEnabled && ((worldZ > -0.53f && worldZ < -0.46f) || (worldZ > 1.08f && worldZ < 1.85) || worldZ > 3.5) && worldY > 0.54f && worldY < 1.09f && worldX < 0.5f) insideSolid = true;

                    h_solidGrid[index] = insideSolid ? 1 : 0;

                     // ram
                    if(worldX < -0.95f && worldX > -1.57f && worldY < 3.6f && worldY > 1.2f && worldZ < 0.6f && worldZ > 0.18f) heatSource = 1.0f;

                    // gpu
                    if(gpuEnabled && worldZ > -0.53f && worldZ < 3.7 && worldY > 0.8f && worldY < 1.09f && worldX < 0.5f) heatSource = 1.0f;

                    // cpu
                    if(worldZ > 1.2f && worldZ < 1.9f && worldY < 2.7f && worldY > 2.0f && worldX < -1.2 && worldX > -1.7) heatSource = 1.0f;

                    // psu
                    if(worldY < -2.2f && worldY > -2.6f && worldZ > 0.4f && worldZ < 3.6f && worldX < 1.4f && worldX > -1.4f) heatSource = 1.0f;

                    h_heatSources[index] = heatSource;
                }
            }
        }
    };
    fillOccupancy();
    
    auto buildFanLists = [&](){
        h_fanPositions.clear();
        h_fanDirections.clear();
        if(topFanEnabled){
            float3 topPos = make_float3(-0.22f, 4.2f, 1.6f);
            float3 topDir = make_float3(0.0f, 1.0f, 0.0f);
            h_fanPositions.push_back(topPos);
            h_fanDirections.push_back(topDir);
        }
        if(frontFanEnabled){
            float3 frontPos = make_float3(0.48f, 2.6f, 3.5f);
            float3 frontDir = make_float3(0.0f, 0.0f, 1.0f);
            h_fanPositions.push_back(frontPos);
            h_fanDirections.push_back(frontDir);
        }
        if(cpuFanEnabled){
            float3 cpuPos = make_float3(0.1f, 2.4f, 0.85f);
            float3 cpuDir = make_float3(0.0f, 0.0f, 1.0f);
            h_fanPositions.push_back(cpuPos);
            h_fanDirections.push_back(cpuDir);
        }
        if(gpuEnabled){
            float3 gpu1Pos = make_float3(-0.36f, 0.8f, 0.34f);
            float3 gpu2Pos = make_float3(-0.36f, 0.8f, 2.71f);
            float3 gpuDir = make_float3(0.0f, 1.0f, 0.0f);
            h_fanPositions.push_back(gpu1Pos);
            h_fanPositions.push_back(gpu2Pos);
            h_fanDirections.push_back(gpuDir);
            h_fanDirections.push_back(gpuDir);
        }
        for(int i=0; i<3; i++){
            if(backFanLocations[i] == 1.0f) continue;
            float3 backFanPos = {0.0f, 2.5f + backFanLocations[i], -3.35f};
            float3 backFanDir = {0.0f, 0.0f, 1.0f};
            h_fanPositions.push_back(backFanPos);
            h_fanDirections.push_back(backFanDir);
        }

        CUDA_CHECK(cudaMemcpy(d_solidGrid, h_solidGrid.data(), solidGridSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_heatSources, h_heatSources.data(), pressureFieldSize, cudaMemcpyHostToDevice));

        numFans = static_cast<int>(h_fanPositions.size());
        if(numFans > 0) {
            CUDA_CHECK(cudaMemcpy(d_fanPositions, h_fanPositions.data(), fanPositionsSize, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_fanDirections, h_fanDirections.data(), fanDirectionsSize, cudaMemcpyHostToDevice));
        }
    };
    buildFanLists();
    
    bool prevGpuEnabled = gpuEnabled;
    bool prevTopFanEnabled = topFanEnabled;
    bool prevCpuFanEnabled = cpuFanEnabled;
    bool prevFrontFanEnabled = frontFanEnabled;
    float prevBackFanLocations[3] = {backFanLocations[0], backFanLocations[1], backFanLocations[2]};
    float dt = 1 / 60.0f; // 60 FPS limit
    runFluidSimulation(
        gridSizeX, gridSizeY, gridSizeZ,
        d_velocityField,
        d_pressureField,
        d_solidGrid,
        d_fanPositions,
        d_fanDirections,
        d_heatSources,
        d_temperature,
        numFans,
        dt
    );
    CUDA_CHECK(cudaMemcpy(h_temperature.data(), d_temperature, pressureFieldSize, cudaMemcpyDeviceToHost));
    std::memcpy(temperatureField, h_temperature.data(), pressureFieldSize);
    CUDA_CHECK(cudaMemcpy(h_velocity.data(), d_velocityField, velocityFieldSize, cudaMemcpyDeviceToHost));
    std::memcpy(velocityField, h_velocity.data(), velocityFieldSize);
    CUDA_CHECK(cudaMemcpy(h_pressure.data(), d_pressureField, pressureFieldSize, cudaMemcpyDeviceToHost));
    std::memcpy(pressureField, h_pressure.data(), pressureFieldSize);
    signalVelocityFieldReady();
    while(running){
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
            d_pressureField,
            d_solidGrid,
            d_fanPositions,
            d_fanDirections,
            d_heatSources,
            d_temperature,
            numFans,
            dt
        );
        CUDA_CHECK(cudaMemcpy(h_temperature.data(), d_temperature, pressureFieldSize, cudaMemcpyDeviceToHost));
        std::memcpy(temperatureField, h_temperature.data(), pressureFieldSize);
        if(displayPressure){
            CUDA_CHECK(cudaMemcpy(h_pressure.data(), d_pressureField, pressureFieldSize, cudaMemcpyDeviceToHost));
            std::memcpy(pressureField, h_pressure.data(), pressureFieldSize);
        }
        else{
            CUDA_CHECK(cudaMemcpy(h_velocity.data(), d_velocityField, velocityFieldSize, cudaMemcpyDeviceToHost));
            std::memcpy(velocityField, h_velocity.data(), velocityFieldSize);
        }
    }
    CUDA_CHECK(cudaFree(d_velocityField));
    CUDA_CHECK(cudaFree(d_pressureField));
    CUDA_CHECK(cudaFree(d_solidGrid));
    CUDA_CHECK(cudaFree(d_fanPositions));
    CUDA_CHECK(cudaFree(d_fanDirections));
    return 0;
}