#include <iostream>
#include <vector>
#include <unordered_set>
#include <cstring>
#include <cuda_runtime.h>

__constant__ int c_GX;
__constant__ int c_GY;
__constant__ int c_GZ;
__constant__ float c_cellSizeX;
__constant__ float c_cellSizeY;
__constant__ float c_cellSizeZ;
__constant__ float c_worldMinX;
__constant__ float c_worldMinY;
__constant__ float c_worldMinZ;
__constant__ float c_thermalDiffusivity;
__constant__ float c_ambientTemperature;
__constant__ float c_coolingRate;
__constant__ float c_heatSourceStrength;
__constant__ float c_thermalExpansionCoefficient;
__constant__ float c_gravity;
__constant__ float c_buoyancyFactor;
__constant__ float c_referenceDensity;

__host__ __device__ static inline int idx3D(int x, int y, int z, int gridSizeX, int gridSizeY){
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

constexpr float worldMinX = -2.0f;
constexpr float worldMaxX = 2.0f;
constexpr float worldMinY = -4.5f;
constexpr float worldMaxY = 4.5f;
constexpr float worldMinZ = -4.0f;
constexpr float worldMaxZ = 4.0f;

constexpr int maxPressureIterations = 10;
constexpr float pressureTolerance = 1e-4f;

constexpr float thermalDiffusivity = 2.2e-5f;
constexpr float ambientTemperature = 22.0f;
constexpr float coolingRate = 0.01f;
constexpr float heatSourceStrength = 25.0f;

constexpr float thermalExpansionCoefficient = 0.0034f;
constexpr float gravity = 9.81f;
constexpr float buoyancyFactor = 0.5f;
constexpr float referenceDensity = 1.225f;

__host__ void initializeConstants(int gridSizeX, int gridSizeY, int gridSizeZ){
    cudaMemcpyToSymbol(c_GX, &gridSizeX, sizeof(int));
    cudaMemcpyToSymbol(c_GY, &gridSizeY, sizeof(int));
    cudaMemcpyToSymbol(c_GZ, &gridSizeZ, sizeof(int));
    float tempCellSizeX = (worldMaxX - worldMinX) / gridSizeX;
    float tempCellSizeY = (worldMaxY - worldMinY) / gridSizeY;
    float tempCellSizeZ = (worldMaxZ - worldMinZ) / gridSizeZ;
    cudaMemcpyToSymbol(c_cellSizeX, &tempCellSizeX, sizeof(float));
    cudaMemcpyToSymbol(c_cellSizeY, &tempCellSizeY, sizeof(float));
    cudaMemcpyToSymbol(c_cellSizeZ, &tempCellSizeZ, sizeof(float));
    cudaMemcpyToSymbol(c_worldMinX, &worldMinX, sizeof(float));
    cudaMemcpyToSymbol(c_worldMinY, &worldMinY, sizeof(float));
    cudaMemcpyToSymbol(c_worldMinZ, &worldMinZ, sizeof(float));
    cudaMemcpyToSymbol(c_thermalDiffusivity, &thermalDiffusivity, sizeof(float));
    cudaMemcpyToSymbol(c_ambientTemperature, &ambientTemperature, sizeof(float));
    cudaMemcpyToSymbol(c_coolingRate, &coolingRate, sizeof(float));
    cudaMemcpyToSymbol(c_heatSourceStrength, &heatSourceStrength, sizeof(float));
    cudaMemcpyToSymbol(c_thermalExpansionCoefficient, &thermalExpansionCoefficient, sizeof(float));
    cudaMemcpyToSymbol(c_gravity, &gravity, sizeof(float));
    cudaMemcpyToSymbol(c_buoyancyFactor, &buoyancyFactor, sizeof(float));
    cudaMemcpyToSymbol(c_referenceDensity, &referenceDensity, sizeof(float));
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void initializeConstantsExtern(int gridSizeX, int gridSizeY, int gridSizeZ){
    initializeConstants(gridSizeX, gridSizeY, gridSizeZ);
}

class CudaMemoryPool{
private:
    struct Block{
        void* ptr;
        size_t size;
        bool inUse;
        Block(void* p, size_t s) : ptr(p), size(s), inUse(false) {}
    };
    std::vector<Block> blocks;
    std::unordered_set<void*> allocatedPointers;
public:
    ~CudaMemoryPool(){
        for(auto& block : blocks) if(block.ptr) cudaFree(block.ptr);
    }
    void* allocate(size_t size){
        size = ((size + 255) / 256) * 256;
        for(auto& block : blocks){
            if(!block.inUse && block.size >= size){
                block.inUse = true;
                allocatedPointers.insert(block.ptr);
                return block.ptr;
            }
        }
        void* newPtr;
        CUDA_CHECK(cudaMalloc(&newPtr, size));
        blocks.emplace_back(newPtr, size);
        blocks.back().inUse = true;
        allocatedPointers.insert(newPtr);
        return newPtr;
    }
    void deallocate(void* ptr){
        if(allocatedPointers.find(ptr)==allocatedPointers.end()) return;
        for(auto& block : blocks){
            if(block.ptr == ptr && block.inUse){
                block.inUse = false;
                allocatedPointers.erase(ptr);
                return;
            }
        }
    }
    static CudaMemoryPool& getInstance(){
        static CudaMemoryPool instance;
        return instance;
    }
};

class SimulationMemory{
private:
    float* d_divergence = nullptr;
    float* d_pressure = nullptr;
    float* d_pressureOut = nullptr;
    float* d_residual = nullptr;
    float* d_tempVelocity = nullptr;
    float* d_tempTemperature = nullptr;
    int allocatedGridSize = 0;
public:
    ~SimulationMemory(){
        cleanup();
    }
    void cleanup(){
        if(allocatedGridSize==0) return;
        auto& pool = CudaMemoryPool::getInstance();
        pool.deallocate(d_divergence);
        pool.deallocate(d_pressure);
        pool.deallocate(d_pressureOut);
        pool.deallocate(d_residual);
        pool.deallocate(d_tempVelocity);
        pool.deallocate(d_tempTemperature);
        allocatedGridSize = 0;
    }
    void ensureAllocated(int numCells){
        if(allocatedGridSize>=numCells) return;
        cleanup();
        auto& pool = CudaMemoryPool::getInstance();
        d_divergence = static_cast<float*>(pool.allocate(numCells * sizeof(float)));
        d_pressure = static_cast<float*>(pool.allocate(numCells * sizeof(float)));
        d_pressureOut = static_cast<float*>(pool.allocate(numCells * sizeof(float)));
        d_residual = static_cast<float*>(pool.allocate(numCells * sizeof(float)));
        d_tempVelocity = static_cast<float*>(pool.allocate(numCells * 3 * sizeof(float)));
        d_tempTemperature = static_cast<float*>(pool.allocate(numCells * sizeof(float)));
        allocatedGridSize = numCells;
    }
    float* getDivergence() { return d_divergence; }
    float* getPressure() { return d_pressure; }
    float* getPressureOut() { return d_pressureOut; }
    float* getResidual() { return d_residual; }
    float* getTempVelocity() { return d_tempVelocity; }
    float* getTempTemperature() { return d_tempTemperature; }
    static SimulationMemory& getInstance() {
        static SimulationMemory instance;
        return instance;
    }
};

__device__ __forceinline__ bool isValidFluidCell(int x, int y, int z, const unsigned char* __restrict__ solidGrid) {
    if (x < 0 || x >= c_GX || y < 0 || y >= c_GY || z < 0 || z >= c_GZ) return false;
    return solidGrid[idx3D(x, y, z, c_GX, c_GY)] == 0;
}

__global__ void computeDivergenceKernel(
    const float* __restrict__ velocity,
    float* __restrict__ divergence,
    const unsigned char* __restrict__ solidGrid
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= c_GX || j >= c_GY || k >= c_GZ) return;
    int idx = idx3D(i, j, k, c_GX, c_GY);
    if(solidGrid[idx] != 0){
        divergence[idx] = 0.0f;
        return;
    }
    float div = 0.0f;
    if(i < c_GX-1 && i > 0){
        float uRight = isValidFluidCell(i+1, j, k, solidGrid) ? velocity[idx3D(i+1, j, k, c_GX, c_GY) * 3 + 0] : 0.0f;
        float uLeft = isValidFluidCell(i-1, j, k, solidGrid) ? velocity[idx3D(i-1, j, k, c_GX, c_GY) * 3 + 0] : 0.0f;
        div = __fmaf_rn(uRight - uLeft, __frcp_rn(2.0f * c_cellSizeX), div);
    }
    if(j < c_GY-1 && j > 0){
        float vUp = isValidFluidCell(i, j+1, k, solidGrid) ? velocity[idx3D(i, j+1, k, c_GX, c_GY) * 3 + 1] : 0.0f;
        float vDown = isValidFluidCell(i, j-1, k, solidGrid) ? velocity[idx3D(i, j-1, k, c_GX, c_GY) * 3 + 1] : 0.0f;
        div = __fmaf_rn(vUp - vDown, __frcp_rn(2.0f * c_cellSizeY), div);
    }
    if(k < c_GZ-1 && k > 0){
        float wFront = isValidFluidCell(i, j, k+1, solidGrid) ? velocity[idx3D(i, j, k+1, c_GX, c_GY) * 3 + 2] : 0.0f;
        float wBack = isValidFluidCell(i, j, k-1, solidGrid) ? velocity[idx3D(i, j, k-1, c_GX, c_GY) * 3 + 2] : 0.0f;
        div = __fmaf_rn(wFront - wBack, __frcp_rn(2.0f * c_cellSizeZ), div);
    }
    divergence[idx] = div;
}

__global__ void pressureJacobianKernel(
    const float* __restrict__ pressureIn,
    float* __restrict__ pressureOut,
    const float* __restrict__ divergence,
    const unsigned char* __restrict__ solidGrid,
    float dt
){
    __shared__ float s_pressure[10][10][10];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    int tz = threadIdx.z + 1;
    if(i<c_GX && j<c_GY && k<c_GZ){
        s_pressure[tx][ty][tz] = pressureIn[idx3D(i, j, k, c_GX, c_GY)];
        if(threadIdx.x==0&&i>0) s_pressure[0][ty][tz] = pressureIn[idx3D(i-1, j, k, c_GX, c_GY)];
        if(threadIdx.x==blockDim.x-1&&i<c_GX-1) s_pressure[tx+1][ty][tz] = pressureIn[idx3D(i+1, j, k, c_GX, c_GY)];
        if(threadIdx.y==0&&j>0) s_pressure[tx][0][tz] = pressureIn[idx3D(i, j-1, k, c_GX, c_GY)];
        if(threadIdx.y==blockDim.y-1&&j<c_GY-1) s_pressure[tx][ty+1][tz] = pressureIn[idx3D(i, j+1, k, c_GX, c_GY)];
        if(threadIdx.z==0&&k>0) s_pressure[tx][ty][0] = pressureIn[idx3D(i, j, k-1, c_GX, c_GY)];
        if(threadIdx.z==blockDim.z-1&&k<c_GZ-1) s_pressure[tx][ty][tz+1] = pressureIn[idx3D(i, j, k+1, c_GX, c_GY)];
    }
    __syncthreads();
    if (i >= c_GX || j >= c_GY || k >= c_GZ) return;
    int idx = idx3D(i, j, k, c_GX, c_GY);
    if(solidGrid[idx] != 0){
        pressureOut[idx] = 0.0f;
        return;
    }
    float neighborPressureSum = 0.0f;
    int neighborCount = 0;
    if(i>0&&solidGrid[idx3D(i-1, j, k, c_GX, c_GY)] == 0){
        neighborPressureSum += s_pressure[tx-1][ty][tz];
        neighborCount++;
    }
    if(i<c_GX-1&&solidGrid[idx3D(i+1, j, k, c_GX, c_GY)] == 0){
        neighborPressureSum += s_pressure[tx+1][ty][tz];
        neighborCount++;
    }
    if(j>0&&solidGrid[idx3D(i, j-1, k, c_GX, c_GY)] == 0){
        neighborPressureSum += s_pressure[tx][ty-1][tz];
        neighborCount++;
    }
    if(j<c_GY-1&&solidGrid[idx3D(i, j+1, k, c_GX, c_GY)] == 0){
        neighborPressureSum += s_pressure[tx][ty+1][tz];
        neighborCount++;
    }
    if(k>0&&solidGrid[idx3D(i, j, k-1, c_GX, c_GY)] == 0){
        neighborPressureSum += s_pressure[tx][ty][tz-1];
        neighborCount++;
    }
    if(k<c_GZ-1&&solidGrid[idx3D(i, j, k+1, c_GX, c_GY)] == 0){
        neighborPressureSum += s_pressure[tx][ty][tz+1];
        neighborCount++;
    }
    if(neighborCount==0){
        pressureOut[idx] = 0.0f;
        return;
    }
    float beta = 0.6f;
    float avgCellSize = (c_cellSizeX + c_cellSizeY + c_cellSizeZ) / 3.0f;
    float scale = avgCellSize * avgCellSize / dt;
    float newPressure = (neighborPressureSum - divergence[idx] * scale) / neighborCount;
    pressureOut[idx] = (1.0f - beta) * pressureIn[idx] + beta * newPressure;
}

__global__ void subtractPressureGradientKernel(
    float* __restrict__ velocity,
    const float* __restrict__ pressure,
    const unsigned char* __restrict__ solidGrid,
    float dt
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= c_GX || j >= c_GY || k >= c_GZ) return;
    int idx = idx3D(i, j, k, c_GX, c_GY);
    if(solidGrid[idx] != 0){
        velocity[idx * 3 + 0] = 0.0f;
        velocity[idx * 3 + 1] = 0.0f;
        velocity[idx * 3 + 2] = 0.0f;
        return;
    }
    float pressureGradientX = 0.0f;
    float pressureGradientY = 0.0f;
    float pressureGradientZ = 0.0f;
    if(i>0 && i<c_GX-1){
        float pRight = isValidFluidCell(i+1, j, k, solidGrid) ? pressure[idx3D(i+1, j, k, c_GX, c_GY)] : pressure[idx];
        float pLeft = isValidFluidCell(i-1, j, k, solidGrid) ? pressure[idx3D(i-1, j, k, c_GX, c_GY)] : pressure[idx];
        pressureGradientX = __fmul_rn(pRight - pLeft, __frcp_rn(2.0f * c_cellSizeX));
    }
    if(j>0 && j<c_GY-1){
        float pUp = isValidFluidCell(i, j+1, k, solidGrid) ? pressure[idx3D(i, j+1, k, c_GX, c_GY)] : pressure[idx];
        float pDown = isValidFluidCell(i, j-1, k, solidGrid) ? pressure[idx3D(i, j-1, k, c_GX, c_GY)] : pressure[idx];
        pressureGradientY = __fmul_rn(pUp - pDown, __frcp_rn(2.0f * c_cellSizeY));
    }
    if(k>0 && k<c_GZ-1){
        float pFront = isValidFluidCell(i, j, k+1, solidGrid) ? pressure[idx3D(i, j, k+1, c_GX, c_GY)] : pressure[idx];
        float pBack = isValidFluidCell(i, j, k-1, solidGrid) ? pressure[idx3D(i, j, k-1, c_GX, c_GY)] : pressure[idx];
        pressureGradientZ = __fmul_rn(pFront - pBack, __frcp_rn(2.0f * c_cellSizeZ));
    }
    velocity[idx * 3 + 0] = __fmaf_rn(-pressureGradientX, dt, velocity[idx * 3 + 0]);
    velocity[idx * 3 + 1] = __fmaf_rn(-pressureGradientY, dt, velocity[idx * 3 + 1]);
    velocity[idx * 3 + 2] = __fmaf_rn(-pressureGradientZ, dt, velocity[idx * 3 + 2]);
}

__global__ void computeResidualKernel(
    const float* __restrict__ pressure,
    const float* __restrict__ divergence,
    float* __restrict__ residual,
    const unsigned char* __restrict__ solidGrid
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= c_GX || j >= c_GY || k >= c_GZ) return;
    int idx = idx3D(i, j, k, c_GX, c_GY);
    if(solidGrid[idx] != 0){
        residual[idx] = 0.0f;
        return;
    }
    float laplacian = 0.0f;
    float centerPressure = pressure[idx];
    int neighbors[6][3] = {
        {-1, 0, 0},
        {1, 0, 0},
        {0, -1, 0},
        {0, 1, 0},
        {0, 0, -1},
        {0, 0, 1}
    };
    float cellSizes[3] = {c_cellSizeX, c_cellSizeY, c_cellSizeZ};
    for(int n=0; n<6; n++){
        int ni = i + neighbors[n][0];
        int nj = j + neighbors[n][1];
        int nk = k + neighbors[n][2];
        if(ni >= 0 && ni < c_GX && nj >= 0 && nj < c_GY && nk >= 0 && nk < c_GZ){
            int nidx = idx3D(ni, nj, nk, c_GX, c_GY);
            if(solidGrid[nidx] == 0){
                int axis = n/2;
                float h = cellSizes[axis];
                laplacian += (pressure[nidx] - centerPressure) / (h * h);
            }
        }
    }
    float residualValue = laplacian - divergence[idx];
    residual[idx] = residualValue * residualValue;
}

__global__ void advectDiffusionHeatKernel(
    const float* __restrict__ tempIn,
    float* __restrict__ tempOut,
    const float* __restrict__ velocity,
    const float* __restrict__ heatSources,
    unsigned char* solidGrid,
    float dt
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= c_GX || j >= c_GY || k >= c_GZ) return;
    int idx = idx3D(i, j, k, c_GX, c_GY);
    if(solidGrid[idx] != 0){
        float heatExchangeRate = 10.0f;
        float invDenom = __frcp_rn(1.0f + heatExchangeRate * dt);
        tempOut[idx] = __fmul_rn(tempIn[idx] + heatExchangeRate * dt * c_ambientTemperature, invDenom);
        return;
    }
    float cellVolume = c_cellSizeX * c_cellSizeY * c_cellSizeZ;
    float airDensity = 1.225;
    float specificHeat = 1005.0f;
    float heatCapacity = airDensity * specificHeat * cellVolume;
    float vx = velocity[idx * 3 + 0];
    float vy = velocity[idx * 3 + 1];
    float vz = velocity[idx * 3 + 2];
    float advectionTerm = 0.0f;
    float maxCellSize = fmaxf(fmaxf(c_cellSizeX, c_cellSizeY), c_cellSizeZ);
    float advectionCFL = __fmul_rn(fmaxf(fmaxf(fabsf(vx), fabsf(vy)), fabsf(vz)), __fdividef(dt, maxCellSize));
    float fluxLimiter = fminf(1.0f, __fdividef(0.8f, fmaxf(advectionCFL, 1e-6f)));
    if(__float_as_int(vx)>0&&i>0){
        int leftIdx = idx3D(i-1, j, k, c_GX, c_GY);
        if(solidGrid[leftIdx] == 0) advectionTerm -= vx * (tempIn[idx] - tempIn[leftIdx]) / c_cellSizeX * fluxLimiter;
    } else if(__float_as_int(vx)<0&&i<c_GX-1){
        int rightIdx = idx3D(i+1, j, k, c_GX, c_GY);
        if(solidGrid[rightIdx] == 0) advectionTerm -= vx * (tempIn[rightIdx] - tempIn[idx]) / c_cellSizeX * fluxLimiter;
    }
    if(__float_as_int(vy)>0&&j>0){
        int downIdx = idx3D(i, j-1, k, c_GX, c_GY);
        if(solidGrid[downIdx] == 0) advectionTerm -= vy * (tempIn[idx] - tempIn[downIdx]) / c_cellSizeY * fluxLimiter;
    } else if(__float_as_int(vy)<0&&j<c_GY-1){
        int upIdx = idx3D(i, j+1, k, c_GX, c_GY);
        if(solidGrid[upIdx] == 0) advectionTerm -= vy * (tempIn[upIdx] - tempIn[idx]) / c_cellSizeY * fluxLimiter;
    }
    if(__float_as_int(vz)>0&&k>0){
        int backIdx = idx3D(i, j, k-1, c_GX, c_GY);
        if(solidGrid[backIdx] == 0) advectionTerm -= vz * (tempIn[idx] - tempIn[backIdx]) / c_cellSizeZ * fluxLimiter;
    } else if(__float_as_int(vz)<0&&k<c_GZ-1){
        int frontIdx = idx3D(i, j, k+1, c_GX, c_GY);
        if(solidGrid[frontIdx] == 0) advectionTerm -= vz * (tempIn[frontIdx] - tempIn[idx]) / c_cellSizeZ * fluxLimiter;
    }
    float diffusionTerm = 0.0f;
    int neighbors[6][3] = {
        {-1, 0, 0},
        {1, 0, 0},
        {0, -1, 0},
        {0, 1, 0},
        {0, 0, -1},
        {0, 0, 1}
    };
    float cellSizes[3] = {c_cellSizeX, c_cellSizeY, c_cellSizeZ};
    for(int n=0; n<6; n++){
        int ni = i + neighbors[n][0];
        int nj = j + neighbors[n][1];
        int nk = k + neighbors[n][2];
        int axis = n/2;
        float h = cellSizes[axis];
        float neighborTemp = c_ambientTemperature;
        if(ni >= 0 && ni < c_GX && nj >= 0 && nj < c_GY && nk >= 0 && nk < c_GZ){
            int nidx = idx3D(ni, nj, nk, c_GX, c_GY);
            neighborTemp = tempIn[nidx];
        }
        diffusionTerm += (neighborTemp - tempIn[idx]) / (h * h);
    }
    diffusionTerm *= c_thermalDiffusivity;
    float heatPowerWatts = heatSources[idx] * c_heatSourceStrength;
    float heatSourceTerm = heatPowerWatts / heatCapacity;
    float tempDiff = tempIn[idx] - c_ambientTemperature;
    float convectiveCooling = c_coolingRate * tempDiff;
    float beta = convectiveCooling * dt;
    float explicitTerm = tempIn[idx] + dt * (advectionTerm + heatSourceTerm + diffusionTerm);
    float newTemp = explicitTerm / (1.0f + beta);
    float dTdt = newTemp - tempIn[idx];
    float maxTempChange = dt * 500.0f;
    if(fabsf(dTdt) > maxTempChange) newTemp = tempIn[idx] + (dTdt > 0 ? +maxTempChange : -maxTempChange);
    newTemp = fmaxf(newTemp, c_ambientTemperature);
    newTemp = fminf(newTemp, 200.0f);
    tempOut[idx] = newTemp;
}

__global__ void addBuoyancyForcesKernel(
    float* __restrict__ velocity,
    const float* __restrict__ temperature,
    const unsigned char* __restrict__ solidGrid,
    float dt
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= c_GX || j >= c_GY || k >= c_GZ) return;
    int idx = idx3D(i, j, k, c_GX, c_GY);
    if(solidGrid[idx] != 0){
        velocity[idx * 3 + 0] = 0.0f;
        velocity[idx * 3 + 1] = 0.0f;
        velocity[idx * 3 + 2] = 0.0f;
        return;
    }
    float tempDiff = temperature[idx] - c_ambientTemperature;
    if(tempDiff > 2.0f){
        float densityChange = -c_referenceDensity * c_thermalExpansionCoefficient * tempDiff;
        float buoyancyForce = densityChange * c_gravity * c_buoyancyFactor / c_referenceDensity;
        float maxBuoyancyAccel = 20.0f;
        buoyancyForce = fminf(fmaxf(buoyancyForce, -maxBuoyancyAccel), maxBuoyancyAccel);
        velocity[idx * 3 + 1] += buoyancyForce * dt;
        if(tempDiff > 20.0f){
            float thermalSpreadForce = fmin(tempDiff * 0.0002f, 0.01f);
            float gradX = 0.0f;
            float gradZ = 0.0f;
            if(i > 0 && i < c_GX - 1){
                int leftIdx = idx3D(i - 1, j, k, c_GX, c_GY);
                int rightIdx = idx3D(i + 1, j, k, c_GX, c_GY);
                if(solidGrid[leftIdx]==0 && solidGrid[rightIdx]==0) gradX = (temperature[rightIdx] - temperature[leftIdx]) / (2.0f * c_cellSizeX);
            }
            if(k > 0 && k < c_GZ - 1){
                int backIdx = idx3D(i, j, k - 1, c_GX, c_GY);
                int frontIdx = idx3D(i, j, k + 1, c_GX, c_GY);
                if(solidGrid[backIdx]==0 && solidGrid[frontIdx]==0) gradZ = (temperature[frontIdx] - temperature[backIdx]) / (2.0f * c_cellSizeZ);
            }
            velocity[idx * 3 + 0] -= gradX * thermalSpreadForce * dt;
            velocity[idx * 3 + 2] -= gradZ * thermalSpreadForce * dt;
        }
    }
    const float maxVelocity = 20.0f;
    const float damping = 0.95f;
    velocity[idx * 3 + 0] = fminf(fmaxf(velocity[idx * 3 + 0], -maxVelocity), maxVelocity) * damping;
    velocity[idx * 3 + 1] = fminf(fmaxf(velocity[idx * 3 + 1], -maxVelocity), maxVelocity) * damping;
    velocity[idx * 3 + 2] = fminf(fmaxf(velocity[idx * 3 + 2], -maxVelocity), maxVelocity) * damping;
}

__host__ void solvePressureProjection(
    float* d_velocity,
    float* d_pressureField,
    unsigned char* d_solidGrid,
    int GX, int GY, int GZ,
    float dt
){
    const int numCells = GX * GY * GZ;
    auto& simMem = SimulationMemory::getInstance();
    simMem.ensureAllocated(numCells);
    float* d_divergence = simMem.getDivergence();
    float* d_pressure = simMem.getPressure();
    float* d_pressureOut = simMem.getPressureOut();
    float* d_residual = simMem.getResidual();

    CUDA_CHECK(cudaMemset(d_pressure, 0, numCells * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pressureOut, 0, numCells * sizeof(float)));

    dim3 block(8, 8, 8);
    dim3 grid(
        (GX + block.x - 1) / block.x,
        (GY + block.y - 1) / block.y,
        (GZ + block.z - 1) / block.z
    );
    computeDivergenceKernel<<<grid, block>>>(
        d_velocity, d_divergence, d_solidGrid
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    float* d_pressure_in = d_pressure;
    float* d_pressure_out = d_pressureOut;
    for(int iter = 0; iter < maxPressureIterations; iter++){
        pressureJacobianKernel<<<grid, block>>>(
            d_pressure_in, d_pressure_out, d_divergence, d_solidGrid, dt
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        if(iter%5==4 || iter == maxPressureIterations-1){
            computeResidualKernel<<<grid, block>>>(
                d_pressure_in, d_divergence, d_residual, d_solidGrid
            );
            CUDA_CHECK(cudaDeviceSynchronize());
            float residualSum = 0.0f;
            float* h_residual = new float[numCells];
            CUDA_CHECK(cudaMemcpy(h_residual, d_residual, numCells * sizeof(float), cudaMemcpyDeviceToHost));
            for(int i = 0; i < numCells; i++) residualSum += h_residual[i];
            delete[] h_residual;
            float avgResidual = residualSum / numCells;
            if(avgResidual < pressureTolerance){
                std::swap(d_pressure_in, d_pressure_out);
                break;
            }
        }
        std::swap(d_pressure_in, d_pressure_out);
    }
    CUDA_CHECK(cudaMemcpy(d_pressureField, d_pressure_in, numCells * sizeof(float), cudaMemcpyDeviceToDevice));
    subtractPressureGradientKernel<<<grid, block>>>(
        d_velocity, d_pressure_in, d_solidGrid, dt
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void addFanForcesKernel(
    float* __restrict__ velocity,
    const unsigned char* __restrict__ solidGrid,
    const float3* __restrict__ fanPos,
    const float3* __restrict__ fanDir,
    const int numFans,
    const float dampeningFactor
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= c_GX || j >= c_GY || k >= c_GZ) return;
    int idx = idx3D(i, j, k, c_GX, c_GY);
    if(solidGrid[idx] != 0){
        velocity[idx * 3 + 0] = 0.0f;
        velocity[idx * 3 + 1] = 0.0f;
        velocity[idx * 3 + 2] = 0.0f;
        return;
    }
    float worldX = __fmaf_rn(i + 0.5f, c_cellSizeX, c_worldMinX);
    float worldY = __fmaf_rn(j + 0.5f, c_cellSizeY, c_worldMinY);
    float worldZ = __fmaf_rn(k + 0.5f, c_cellSizeZ, c_worldMinZ);
    float3 accum = make_float3(0.0f, 0.0f, 0.0f);
    for(int f = 0; f < numFans; ++f){
        float3 fanPosition = fanPos[f];
        float3 fanDirection = fanDir[f];
        float3 toCell = make_float3(
            worldX - fanPosition.x,
            worldY - fanPosition.y,
            worldZ - fanPosition.z
        );
        float distanceSq = __fmaf_rn(toCell.x, toCell.x, __fmaf_rn(toCell.y, toCell.y, toCell.z * toCell.z));
        if(distanceSq < 1e-6f) continue;
        float invDistance = __frsqrt_rn(distanceSq);
        float3 toCellNormalized = make_float3(
            toCell.x * invDistance,
            toCell.y * invDistance,
            toCell.z * invDistance
        );
        float alignment = __fmaf_rn(fanDirection.x, toCellNormalized.x, __fmaf_rn(fanDirection.y, toCellNormalized.y, fanDirection.z * toCellNormalized.z));

        if(alignment > 0.1f){
            float fanRadiusSq = 1.0f;
            float forceMagnitude = __fmaf_rn(5.0f * alignment, __fdividef(1.0f, 1.0f + distanceSq / fanRadiusSq), 0.0f);
            accum.x = __fmaf_rn(fanDirection.x, forceMagnitude, accum.x);
            accum.y = __fmaf_rn(fanDirection.y, forceMagnitude, accum.y);
            accum.z = __fmaf_rn(fanDirection.z, forceMagnitude, accum.z);
        }
    }
    velocity[idx * 3 + 0] = __fmaf_rn(accum.x, 1.0f, velocity[idx * 3 + 0]);
    velocity[idx * 3 + 1] = __fmaf_rn(accum.y, 1.0f, velocity[idx * 3 + 1]);
    velocity[idx * 3 + 2] = __fmaf_rn(accum.z, 1.0f, velocity[idx * 3 + 2]);
    const float maxVelocity = 10.0f;
    velocity[idx * 3 + 0] = fminf(fmaxf(velocity[idx * 3 + 0], -maxVelocity), maxVelocity) * dampeningFactor;
    velocity[idx * 3 + 1] = fminf(fmaxf(velocity[idx * 3 + 1], -maxVelocity), maxVelocity) * dampeningFactor;
    velocity[idx * 3 + 2] = fminf(fmaxf(velocity[idx * 3 + 2], -maxVelocity), maxVelocity) * dampeningFactor;
}

__global__ void advectVelocityKernel(
    const float* __restrict__ velIn,
    float* __restrict__ velOut,
    const unsigned char* __restrict__ solidGrid,
    float dt
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= c_GX || j >= c_GY || k >= c_GZ) return;
    int idx = idx3D(i, j, k, c_GX, c_GY);
    if(solidGrid[idx] != 0){
        velOut[idx * 3 + 0] = 0.0f;
        velOut[idx * 3 + 1] = 0.0f;
        velOut[idx * 3 + 2] = 0.0f;
        return;
    }
    float vx = velIn[idx * 3 + 0];
    float vy = velIn[idx * 3 + 1];
    float vz = velIn[idx * 3 + 2];
    float advectionStrength = 0.5f;
    float x0 = i - vx * dt * advectionStrength / c_cellSizeX;
    float y0 = j - vy * dt * advectionStrength / c_cellSizeY;
    float z0 = k - vz * dt * advectionStrength / c_cellSizeZ;
    x0 = fminf(fmaxf(x0, 0.5f), c_GX - 1.5f);
    y0 = fminf(fmaxf(y0, 0.5f), c_GY - 1.5f);
    z0 = fminf(fmaxf(z0, 0.5f), c_GZ - 1.5f);
    int xi = int(x0);
    int yi = int(y0);
    int zi = int(z0);
    float fx = x0 - xi;
    float fy = y0 - yi;
    float fz = z0 - zi;
    xi = max(0, min(xi, c_GX - 2));
    yi = max(0, min(yi, c_GY - 2));
    zi = max(0, min(zi, c_GZ - 2));

    for(int comp = 0; comp < 3; comp++){
        float c000 = velIn[idx3D(xi, yi, zi, c_GX, c_GY) * 3 + comp];
        float c001 = velIn[idx3D(xi, yi, zi+1, c_GX, c_GY) * 3 + comp];
        float c010 = velIn[idx3D(xi, yi+1, zi, c_GX, c_GY) * 3 + comp];
        float c011 = velIn[idx3D(xi, yi+1, zi+1, c_GX, c_GY) * 3 + comp];
        float c100 = velIn[idx3D(xi+1, yi, zi, c_GX, c_GY) * 3 + comp];
        float c101 = velIn[idx3D(xi+1, yi, zi+1, c_GX, c_GY) * 3 + comp];
        float c110 = velIn[idx3D(xi+1, yi+1, zi, c_GX, c_GY) * 3 + comp];
        float c111 = velIn[idx3D(xi+1, yi+1, zi+1, c_GX, c_GY) * 3 + comp];
        
        float c00 = c000 * (1.0f - fx) + c100 * fx;
        float c01 = c001 * (1.0f - fx) + c101 * fx;
        float c10 = c010 * (1.0f - fx) + c110 * fx;
        float c11 = c011 * (1.0f - fx) + c111 * fx;
        
        float c0 = c00 * (1.0f - fy) + c10 * fy;
        float c1 = c01 * (1.0f - fy) + c11 * fy;
        
        velOut[idx * 3 + comp] = c0 * (1.0f - fz) + c1 * fz;
    }
}

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
){
    dim3 block(8, 8, 8);
    dim3 grid(
        (gridSizeX + block.x - 1) / block.x,
        (gridSizeY + block.y - 1) / block.y,
        (gridSizeZ + block.z - 1) / block.z
    );
    const int numCells = gridSizeX * gridSizeY * gridSizeZ;
    auto& simMem = SimulationMemory::getInstance();
    simMem.ensureAllocated(numCells);
    float* d_tempVelocity = simMem.getTempVelocity();
    float* d_tempTemperature = simMem.getTempTemperature();
    addFanForcesKernel<<<grid, block>>>(
        d_velocityField, d_solidGrid, d_fanPositions, d_fanDirections, numFans, 0.95
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    addBuoyancyForcesKernel<<<grid, block>>>(
        d_velocityField, d_temperature, d_solidGrid, dt
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    advectVelocityKernel<<<grid, block>>>(
        d_velocityField, d_tempVelocity, d_solidGrid, dt
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_velocityField, d_tempVelocity, numCells * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
    solvePressureProjection(
        d_velocityField, d_pressureField, d_solidGrid, gridSizeX, gridSizeY, gridSizeZ, dt
    );
    advectDiffusionHeatKernel<<<grid, block>>>(
        d_temperature, d_tempTemperature, d_velocityField, d_heatSources, d_solidGrid, dt
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_temperature, d_tempTemperature, numCells * sizeof(float), cudaMemcpyDeviceToDevice));
}