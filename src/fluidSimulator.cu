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
__constant__ float c_worldMaxX;
__constant__ float c_worldMaxY;
__constant__ float c_worldMaxZ;
__constant__ float c_thermalDiffusivity;
__constant__ float c_ambientTemperature;
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

constexpr int maxPressureIterations = 8;
constexpr float pressureTolerance = 1e-4f;

constexpr float thermalDiffusivity = 1.5e-5f;
constexpr float ambientTemperature = 22.0f;
constexpr float heatSourceStrength = 1.0f;

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
    cudaMemcpyToSymbol(c_worldMaxX, &worldMaxX, sizeof(float));
    cudaMemcpyToSymbol(c_worldMaxY, &worldMaxY, sizeof(float));
    cudaMemcpyToSymbol(c_worldMaxZ, &worldMaxZ, sizeof(float));
    cudaMemcpyToSymbol(c_thermalDiffusivity, &thermalDiffusivity, sizeof(float));
    cudaMemcpyToSymbol(c_ambientTemperature, &ambientTemperature, sizeof(float));
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
    float* d_tempSum = nullptr;
    float* d_weightSum = nullptr;
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
        pool.deallocate(d_tempSum);
        pool.deallocate(d_weightSum);
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
        d_tempSum = static_cast<float*>(pool.allocate(numCells * sizeof(float)));
        d_weightSum = static_cast<float*>(pool.allocate(numCells * sizeof(float)));
        allocatedGridSize = numCells;
    }
    float* getDivergence() { return d_divergence; }
    float* getPressure() { return d_pressure; }
    float* getPressureOut() { return d_pressureOut; }
    float* getResidual() { return d_residual; }
    float* getTempVelocity() { return d_tempVelocity; }
    float* getTempTemperature() { return d_tempTemperature; }
    float* getTempSum() { return d_tempSum; }
    float* getWeightSum() { return d_weightSum; }
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
    __shared__ unsigned char s_solid[10][10][10];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    int tz = threadIdx.z + 1;
    if(i<c_GX && j<c_GY && k<c_GZ){
        int idx = idx3D(i, j, k, c_GX, c_GY);
        s_pressure[tx][ty][tz] = pressureIn[idx];
        s_solid[tx][ty][tz] = solidGrid[idx];
        if(threadIdx.x == 0 && i > 0){
            int leftIdx = idx3D(i-1, j, k, c_GX, c_GY);
            s_pressure[0][ty][tz] = pressureIn[leftIdx];
            s_solid[0][ty][tz] = solidGrid[leftIdx];
        }
        if(threadIdx.x == blockDim.x-1 && i < c_GX-1){
            int rightIdx = idx3D(i+1, j, k, c_GX, c_GY);
            s_pressure[tx+1][ty][tz] = pressureIn[rightIdx];
            s_solid[tx+1][ty][tz] = solidGrid[rightIdx];
        }
        if(threadIdx.y == 0 && j > 0){
            int downIdx = idx3D(i, j-1, k, c_GX, c_GY);
            s_pressure[tx][0][tz] = pressureIn[downIdx];
            s_solid[tx][0][tz] = solidGrid[downIdx];
        }
        if(threadIdx.y == blockDim.y-1 && j < c_GY-1){
            int upIdx = idx3D(i, j+1, k, c_GX, c_GY);
            s_pressure[tx][ty+1][tz] = pressureIn[upIdx];
            s_solid[tx][ty+1][tz] = solidGrid[upIdx];
        }
        if(threadIdx.z == 0 && k > 0){
            int backIdx = idx3D(i, j, k-1, c_GX, c_GY);
            s_pressure[tx][ty][0] = pressureIn[backIdx];
            s_solid[tx][ty][0] = solidGrid[backIdx];
        }
        if(threadIdx.z == blockDim.z-1 && k < c_GZ-1){
            int frontIdx = idx3D(i, j, k+1, c_GX, c_GY);
            s_pressure[tx][ty][tz+1] = pressureIn[frontIdx];
            s_solid[tx][ty][tz+1] = solidGrid[frontIdx];
        }
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
    if(i > 0 && s_solid[tx-1][ty][tz] == 0){
        neighborPressureSum += s_solid[tx-1][ty][tz] == 0 ? s_pressure[tx-1][ty][tz] : s_pressure[tx][ty][tz];
        neighborCount++;
    }
    if(i < c_GX-1 && s_solid[tx+1][ty][tz] == 0){
        neighborPressureSum += s_solid[tx+1][ty][tz] == 0 ? s_pressure[tx+1][ty][tz] : s_pressure[tx][ty][tz];
        neighborCount++;
    }
    if(j > 0 && s_solid[tx][ty-1][tz] == 0){
        neighborPressureSum += s_solid[tx][ty-1][tz] == 0 ? s_pressure[tx][ty-1][tz] : s_pressure[tx][ty][tz];
        neighborCount++;
    }
    if(j < c_GY-1 && s_solid[tx][ty+1][tz] == 0){
        neighborPressureSum += s_solid[tx][ty+1][tz] == 0 ? s_pressure[tx][ty+1][tz] : s_pressure[tx][ty][tz];
        neighborCount++; 
    }
    if(k > 0 && s_solid[tx][ty][tz-1] == 0){
        neighborPressureSum += s_solid[tx][ty][tz-1] == 0 ? s_pressure[tx][ty][tz-1] : s_pressure[tx][ty][tz];
        neighborCount++;
    }
    if(k < c_GZ-1 && s_solid[tx][ty][tz+1] == 0){
        neighborPressureSum += s_solid[tx][ty][tz+1] == 0 ? s_pressure[tx][ty][tz+1] : s_pressure[tx][ty][tz];
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
    float pressureGradientX = 0.0f;
    float pressureGradientY = 0.0f;
    float pressureGradientZ = 0.0f;
    float tempDiff = temperature[idx] - c_ambientTemperature;
    if(i>0 && i<c_GX-1){
        int leftIdx = idx3D(i-1, j, k, c_GX, c_GY);
        int rightIdx = idx3D(i+1, j, k, c_GX, c_GY);
        float pLeft = pressure[leftIdx];
        float pRight = pressure[rightIdx];
        if(solidGrid[leftIdx] != 0) pLeft = pressure[idx] + tempDiff * 0.2f;
        else pLeft += (temperature[leftIdx] - c_ambientTemperature) * 0.1f;
        if(solidGrid[rightIdx] != 0) pRight = pressure[idx] + tempDiff * 0.2f;
        else pRight += (temperature[rightIdx] - c_ambientTemperature) * 0.1f;
        pressureGradientX = (pRight - pLeft) / (2.0f * c_cellSizeX);
    }
    if(j>0 && j<c_GY-1){
        int downIdx = idx3D(i, j-1, k, c_GX, c_GY);
        int upIdx = idx3D(i, j+1, k, c_GX, c_GY);
        float pDown = pressure[downIdx];
        float pUp = pressure[upIdx];
        if(solidGrid[downIdx] != 0) pDown = pressure[idx] + tempDiff * 0.2f;
        else pDown += (temperature[downIdx] - c_ambientTemperature) * 0.1f;
        if(solidGrid[upIdx] != 0) pUp = pressure[idx] + tempDiff * 0.2f;
        else pUp += (temperature[upIdx] - c_ambientTemperature) * 0.1f;
        pressureGradientY = (pUp - pDown) / (2.0f * c_cellSizeY);
    }
    if(k>0 && k<c_GZ-1){
        int backIdx = idx3D(i, j, k-1, c_GX, c_GY);
        int frontIdx = idx3D(i, j, k+1, c_GX, c_GY);
        float pBack = pressure[backIdx];
        float pFront = pressure[frontIdx];
        if(solidGrid[backIdx] != 0) pBack = pressure[idx] + tempDiff * 0.2f;
        else pBack += (temperature[backIdx] - c_ambientTemperature) * 0.1f;
        if(solidGrid[frontIdx] != 0) pFront = pressure[idx] + tempDiff * 0.2f;
        else pFront += (temperature[frontIdx] - c_ambientTemperature) * 0.1f;
        pressureGradientZ = (pFront - pBack) / (2.0f * c_cellSizeZ);
    }
    velocity[idx * 3 + 0] -= pressureGradientX * dt;
    velocity[idx * 3 + 1] -= pressureGradientY * dt;
    velocity[idx * 3 + 2] -= pressureGradientZ * dt;
}

__global__ void enforceBoundaryConditionsKernel(
    float* __restrict__ velocity,
    const unsigned char* __restrict__ solidGrid,
    const float* __restrict__ temperature
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
    float3 v = reinterpret_cast<float3*>(velocity)[idx];
    float tempDiff = temperature[idx] - c_ambientTemperature;
    float pressureMultiplier = 1.0f + fminf(tempDiff * 0.02f , 2.0f);
    if(i>0 && solidGrid[idx3D(i-1, j, k, c_GX, c_GY)] != 0 && v.x < 0.0f) v.x = -v.x * 0.8 * pressureMultiplier;
    if(i<c_GX-1 && solidGrid[idx3D(i+1, j, k, c_GX, c_GY)] != 0 && v.x > 0.0f) v.x = -v.x * 0.8 * pressureMultiplier;
    if(j>0 && solidGrid[idx3D(i, j-1, k, c_GX, c_GY)] != 0 && v.y < 0.0f) v.y = -v.y * 0.8 * pressureMultiplier;
    if(j<c_GY-1 && solidGrid[idx3D(i, j+1, k, c_GX, c_GY)] != 0 && v.y > 0.0f) v.y = -v.y * 0.8 * pressureMultiplier;
    if(k>0 && solidGrid[idx3D(i, j, k-1, c_GX, c_GY)] != 0 && v.z < 0.0f) v.z = -v.z * 0.8 * pressureMultiplier;
    if(k<c_GZ-1 && solidGrid[idx3D(i, j, k+1, c_GX, c_GY)] != 0 && v.z > 0.0f) v.z = -v.z * 0.8 * pressureMultiplier;
    reinterpret_cast<float3*>(velocity)[idx] = v;
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

__global__ void advectKernel(
    const float* __restrict__ tempIn,
    const float* __restrict__ velocity,
    float* __restrict__ speed,
    float* __restrict__ tempSum,
    float* __restrict__ weightSum,
    const unsigned char* __restrict__ solidGrid,
    float dt
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= c_GX || j >= c_GY || k >= c_GZ) return;
    int idx = idx3D(i, j, k, c_GX, c_GY);
    float temp = tempIn[idx];
    float3 v = reinterpret_cast<const float3*>(velocity)[idx];
    speed[idx] = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    float x0 = (float) i;
    float y0 = (float) j;
    float z0 = (float) k;
    float fx = x0 + v.x * dt / c_cellSizeX;
    float fy = y0 + v.y * dt / c_cellSizeY;
    float fz = z0 + v.z * dt / c_cellSizeZ;
    fx = fminf(fmaxf(fx, 0.0f), (float) (c_GX - 1));
    fy = fminf(fmaxf(fy, 0.0f), (float) (c_GY - 1));
    fz = fminf(fmaxf(fz, 0.0f), (float) (c_GZ - 1));
    int xi = (int) floorf(fx);
    int yi = (int) floorf(fy);
    int zi = (int) floorf(fz);
    xi = max(0, min(xi, c_GX - 2));
    yi = max(0, min(yi, c_GY - 2));
    zi = max(0, min(zi, c_GZ - 2));
    float wx = fx - (float) xi;
    float wy = fy - (float) yi;
    float wz = fz - (float) zi;
    wx = fminf(fmaxf(wx, 0.0f), 1.0f);
    wy = fminf(fmaxf(wy, 0.0f), 1.0f);
    wz = fminf(fmaxf(wz, 0.0f), 1.0f);
    for(int dx=0; dx<=1; dx++){
        for(int dy=0; dy<=1; dy++){
            for(int dz=0; dz<=1; dz++){
                int nx = xi + dx;
                int ny = yi + dy;
                int nz = zi + dz;
                if(nx<0 || nx>=c_GX || ny<0 || ny>=c_GY || nz<0 || nz>=c_GZ) continue;
                int nidx = idx3D(nx, ny, nz, c_GX, c_GY);
                if(solidGrid[nidx]!=0) continue;
                float weight = (dx ? wx : (1.0f-wx)) * (dy ? wy : (1.0f-wy)) * (dz ? wz : (1.0f-wz));
                if(weight > 1e-6f){
                    atomicAdd(&tempSum[nidx], temp * weight);
                    atomicAdd(&weightSum[nidx], weight);
                }
            }
        }
    }
}

__global__ void normalizeForwardAdvected(
    float* __restrict__ tempOut,
    const float* __restrict__ tempSum,
    const float* __restrict__ weightSum,
    const float* __restrict__ tempIn,
    const unsigned char* __restrict__ solidGrid
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= c_GX || j >= c_GY || k >= c_GZ) return;
    int idx = idx3D(i, j, k, c_GX, c_GY);
    float w = weightSum[idx];
    tempOut[idx] = (w > 1e-6f) ? tempSum[idx] / w : c_ambientTemperature;
}

__global__ void diffuseKernel(
    const float* __restrict__ tempAdv,
    float* __restrict__ tempOut,
    const float* __restrict__ heatSources,
    const unsigned char* __restrict__ solidGrid,
    float dt
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= c_GX || j >= c_GY || k >= c_GZ) return;
    int idx = idx3D(i, j, k, c_GX, c_GY);
    if(solidGrid[idx] != 0){
        const float coolingCoefficient = 0.1f;
        float inv = 1.0f / (1.0f + coolingCoefficient * dt);
        float cooled = (tempAdv[idx] + coolingCoefficient * dt * c_ambientTemperature) * inv;
        tempOut[idx] = cooled;
        return;
    }
    float T0 = tempAdv[idx];
    const int neighbors[6][3] = {
        {-1, 0, 0},
        {1, 0, 0},
        {0, -1, 0},
        {0, 1, 0},
        {0, 0, -1},
        {0, 0, 1}
    };
    float cellVolume = c_cellSizeX * c_cellSizeY * c_cellSizeZ;
    float heat = c_heatSourceStrength * heatSources[idx] / (1.225f * 1005.0f * cellVolume);
    float diffusionSum = 0.0f;
    float totalDiffusionCoeff = 0.0f;
    float cellSizes[3] = {c_cellSizeX, c_cellSizeY, c_cellSizeZ};
    for(int n = 0; n < 6; n++){
        int ni = i + neighbors[n][0];
        int nj = j + neighbors[n][1];
        int nk = k + neighbors[n][2];
        if(ni >= 0 && ni < c_GX && nj >= 0 && nj < c_GY && nk >= 0 && nk < c_GZ){
            int nidx = idx3D(ni, nj, nk, c_GX, c_GY);
            float neighborTemp;
            if(solidGrid[nidx] != 0) neighborTemp = c_ambientTemperature;
            else neighborTemp = tempAdv[nidx];
            int axis = n / 2;
            float h = cellSizes[axis];
            float diffusionCoeff = c_thermalDiffusivity / (h * h);
            diffusionSum += diffusionCoeff * (neighborTemp - T0);
            totalDiffusionCoeff += diffusionCoeff;
        }
    }
    float alpha = totalDiffusionCoeff * dt;
    float newTemp = (T0 + alpha * (diffusionSum / totalDiffusionCoeff) + heat) / (1.0f + alpha);
    newTemp = fmaxf(newTemp, c_ambientTemperature - 10.0f);
    newTemp = fminf(newTemp, c_ambientTemperature + 200.0f);
    tempOut[idx] = newTemp;
}

__host__ void solvePressureProjection(
    float* d_velocity,
    float* d_pressureField,
    float* d_temperature,
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
        if(iter%4==3 || iter == maxPressureIterations-1){
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
        d_velocity, d_pressure_in, d_temperature, d_solidGrid, dt
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    enforceBoundaryConditionsKernel<<<grid, block>>>(
        d_velocity, d_solidGrid, d_temperature
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void velocityUpdateKernel(
    const float* __restrict__ velIn,
    float* __restrict__ velOut,
    const float* __restrict__ temperature,
    const unsigned char* __restrict__ solidGrid,
    const float3* __restrict__ fanPos,
    const float3* __restrict__ fanDir,
    const int numFans,
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

    float newVel[3];
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
        
        newVel[comp] = c0 * (1.0f - fz) + c1 * fz;
    }
    float worldX = (i + 0.5f) * c_cellSizeX + c_worldMinX;
    float worldY = (j + 0.5f) * c_cellSizeY + c_worldMinY;
    float worldZ = (k + 0.5f) * c_cellSizeZ + c_worldMinZ;
    float3 fanAccum = make_float3(0.0f, 0.0f, 0.0f);
    for(int f=0; f<numFans; f++){
        float3 fanPosition = fanPos[f];
        float3 fanDirection = fanDir[f];
        float3 toCell = make_float3(
            worldX - fanPosition.x,
            worldY - fanPosition.y,
            worldZ - fanPosition.z
        );
        float distanceSq = toCell.x * toCell.x + toCell.y * toCell.y + toCell.z * toCell.z;
        if(distanceSq < 1e-6f) continue;
        float invDistance = rsqrtf(distanceSq);
        float distance = sqrtf(distanceSq);
        float3 toCellNormalized = make_float3(
            toCell.x * invDistance,
            toCell.y * invDistance,
            toCell.z * invDistance
        );
        float alignment = fanDirection.x * toCellNormalized.x + fanDirection.y * toCellNormalized.y + fanDirection.z * toCellNormalized.z;
        float fanRadiusSq = 1.0f;
        float forceMagnitude = __fmaf_rn(5.0f * alignment, __fdividef(1.0f, 1.0f + distanceSq / fanRadiusSq), 0.0f);
        if(alignment > 0.1f){
            fanAccum.x += fanDirection.x * forceMagnitude;
            fanAccum.y += fanDirection.y * forceMagnitude;
            fanAccum.z += fanDirection.z * forceMagnitude;
        } else if(alignment < 0.1f){
            fanAccum.x -= fanDirection.x * forceMagnitude;
            fanAccum.y -= fanDirection.y * forceMagnitude;
            fanAccum.z -= fanDirection.z * forceMagnitude;
        }
        if(fabsf(alignment) > 0.1f && distance > 0.2f){
            float axialDistance = alignment * distance;
            float3 axialPoint = make_float3(
                fanPosition.x + fanDirection.x * axialDistance,
                fanPosition.y + fanDirection.y * axialDistance,
                fanPosition.z + fanDirection.z * axialDistance
            );
            float3 radialVector = make_float3(
                worldX - axialPoint.x,
                worldY - axialPoint.y,
                worldZ - axialPoint.z
            );
            float radialDistanceSq = radialVector.x * radialVector.x + radialVector.y * radialVector.y + radialVector.z * radialVector.z;
            if(radialDistanceSq > 1e-6f){
                float invRadialDistance = rsqrtf(radialDistanceSq);
                float radialStrength = 0.5f / (distanceSq + 0.5f);
                if(alignment > 0){
                    fanAccum.x += radialVector.x * invRadialDistance * radialStrength;
                    fanAccum.y += radialVector.y * invRadialDistance * radialStrength;
                    fanAccum.z += radialVector.z * invRadialDistance * radialStrength;
                } else{
                    fanAccum.x -= radialVector.x * invRadialDistance * radialStrength;
                    fanAccum.y -= radialVector.y * invRadialDistance * radialStrength;
                    fanAccum.z -= radialVector.z * invRadialDistance * radialStrength;
                }
            }
        }
    }
    newVel[0] += fanAccum.x;
    newVel[1] += fanAccum.y;
    newVel[2] += fanAccum.z;
    float tempDiff = temperature[idx] - c_ambientTemperature;
    if(tempDiff > 2.0f){
        float densityChange = -c_referenceDensity * c_thermalExpansionCoefficient * tempDiff;
        float buoyancyForce = densityChange * c_gravity * c_buoyancyFactor / c_referenceDensity;
        float maxBuoyancyAccel = 20.0f;
        buoyancyForce = fminf(fmaxf(buoyancyForce, -maxBuoyancyAccel), maxBuoyancyAccel);
        newVel[1] -= buoyancyForce * dt;
        if(tempDiff > 20.0f){
            float thermalSpreadForce = fminf(tempDiff * 0.0002f, 0.01f);
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
            newVel[0] -= gradX * thermalSpreadForce * dt;
            newVel[2] -= gradZ * thermalSpreadForce * dt;
        }
    }
    if(tempDiff > 5.0f){
        float gradX = 0.0f;
        float gradY = 0.0f;
        float gradZ = 0.0f;
        if(i>0 && i<c_GX-1){
            int leftIdx = idx3D(i-1, j, k, c_GX, c_GY);
            int rightIdx = idx3D(i+1, j, k, c_GX, c_GY);
            if(solidGrid[leftIdx]==0 && solidGrid[rightIdx]==0) gradX = (temperature[rightIdx] - temperature[leftIdx]) / (2.0f * c_cellSizeX);
        }
        if(j>0 && j<c_GY-1){
            int downIdx = idx3D(i, j-1, k, c_GX, c_GY);
            int upIdx = idx3D(i, j+1, k, c_GX, c_GY);
            if(solidGrid[downIdx]==0 && solidGrid[upIdx]==0) gradX = (temperature[upIdx] - temperature[downIdx]) / (2.0f * c_cellSizeY);
        }
        if(k>0 && k<c_GZ-1){
            int backIdx = idx3D(i, j, k-1, c_GX, c_GY);
            int frontIdx = idx3D(i, j, k+1, c_GX, c_GY);
            if(solidGrid[backIdx]==0 && solidGrid[frontIdx]==0) gradZ = (temperature[frontIdx] - temperature[backIdx]) / (2.0f * c_cellSizeZ);
        }
        float convectionStrength = fminf(tempDiff * tempDiff * 0.001f, 0.5f);
        float thermalVelocityX = -gradX * convectionStrength;
        float thermalVelocityY = -gradY * convectionStrength;
        float thermalVelocityZ = -gradZ * convectionStrength;
        if(tempDiff > 20.0f){
            float circX = (worldZ - c_worldMinZ - (c_worldMaxZ - c_worldMinZ) * 0.5f) * 0.02f;
            float circZ = -(worldX - c_worldMinX - (c_worldMaxX - c_worldMinX) * 0.5f) * 0.02f;
            thermalVelocityX += circX;
            thermalVelocityZ += circZ;
        }
        newVel[0] += thermalVelocityX * dt;
        newVel[1] += thermalVelocityY * dt;
        newVel[2] += thermalVelocityZ * dt;
    }
    const float maxVelocity = 20.0f;
    const float damping = 0.95f;
    velOut[idx * 3 + 0] = fminf(fmaxf(newVel[0], -maxVelocity), maxVelocity) * damping;
    velOut[idx * 3 + 1] = fminf(fmaxf(newVel[1], -maxVelocity), maxVelocity) * damping;
    velOut[idx * 3 + 2] = fminf(fmaxf(newVel[2], -maxVelocity), maxVelocity) * damping;
}

extern "C" void runFluidSimulation(
    int gridSizeX, int gridSizeY, int gridSizeZ,
    float* d_velocityField,
    float* d_speedField,
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
    float* d_tempSum = simMem.getTempSum();
    float* d_weightSum = simMem.getWeightSum();
    CUDA_CHECK(cudaMemset(d_tempSum, 0, numCells * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_weightSum, 0, numCells * sizeof(float)));
    velocityUpdateKernel<<<grid, block>>>(
        d_velocityField, d_tempVelocity, d_temperature, d_solidGrid,
        d_fanPositions, d_fanDirections, numFans, dt
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_velocityField, d_tempVelocity, numCells * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
    solvePressureProjection(
        d_velocityField, d_pressureField, d_temperature, d_solidGrid, gridSizeX, gridSizeY, gridSizeZ, dt
    );
    advectKernel<<<grid, block>>>(
        d_temperature, d_velocityField, d_speedField, d_tempSum, d_weightSum, d_solidGrid, dt
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    normalizeForwardAdvected<<<grid, block>>>(
        d_tempTemperature, d_tempSum, d_weightSum, d_temperature, d_solidGrid
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    diffuseKernel<<<grid, block>>>(
        d_tempTemperature, d_temperature, d_heatSources, d_solidGrid, dt
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}