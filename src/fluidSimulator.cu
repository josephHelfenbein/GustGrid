#include <iostream>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

__global__ void addFanForcesKernel(
    float* velocity,
    unsigned char* solidGrid,
    float3* fanPos,
    float3* fanDir,
    int numFans,
    int GX, int GY, int GZ
);

__global__ void advectVelocityKernel(
    float* velIn,
    float* velOut,
    unsigned char* solidGrid,
    int GX, int GY, int GZ,
    float dt
);

extern "C" void runFluidSimulation(
    int gridSizeX, int gridSizeY, int gridSizeZ,
    float* d_velocityField,
    unsigned char* d_solidGrid,
    float3* d_fanPositions,
    float3* d_fanDirections,
    int numFans,
    float dt
){
    const int numCells = gridSizeX * gridSizeY * gridSizeZ;
    float* d_tempVel = nullptr;
    size_t velocitySize = sizeof(float) * 3 * numCells;
    cudaMalloc(&d_tempVel, velocitySize);
    cudaMemset(d_tempVel, 0, velocitySize);
    {
        dim3 block(8, 8, 8);
        dim3 grid(
            (gridSizeX + block.x - 1) / block.x,
            (gridSizeY + block.y - 1) / block.y,
            (gridSizeZ + block.z - 1) / block.z
        );
        addFanForcesKernel<<<grid, block>>>(
            d_tempVel,
            d_solidGrid,
            d_fanPositions,
            d_fanDirections,
            numFans,
            gridSizeX, gridSizeY, gridSizeZ
        );
        cudaDeviceSynchronize();
        {
            dim3 block(8, 8, 8);
            dim3 grid(
                (gridSizeX + block.x - 1) / block.x,
                (gridSizeY + block.y - 1) / block.y,
                (gridSizeZ + block.z - 1) / block.z
            );
            advectVelocityKernel<<<grid, block>>>(
                d_tempVel,
                d_velocityField,
                d_solidGrid,
                gridSizeX, gridSizeY, gridSizeZ,
                dt
            );
            cudaDeviceSynchronize();
        }
        cudaFree(d_tempVel);
    }
}

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

constexpr int gridSizeX = 128;
constexpr int gridSizeY = 128;
constexpr int gridSizeZ = 128;
constexpr float worldMinX = -2.0f;
constexpr float worldMaxX = 2.0f;
constexpr float worldMinY = -4.5f;
constexpr float worldMaxY = 4.5f;
constexpr float worldMinZ = -4.0f;
constexpr float worldMaxZ = 4.0f;

constexpr float cellSizeX = (worldMaxX - worldMinX) / gridSizeX;
constexpr float cellSizeY = (worldMaxY - worldMinY) / gridSizeY;
constexpr float cellSizeZ = (worldMaxZ - worldMinZ) / gridSizeZ;

__global__ void addFanForcesKernel(
    float* velocity,
    unsigned char* solidGrid,
    float3* fanPos,
    float3* fanDir,
    int numFans,
    int GX, int GY, int GZ
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= GX || j >= GY || k >= GZ) return;
    int idx = idx3D(i, j, k, GX, GY);
    if(solidGrid[idx] != 0){
        velocity[idx * 3 + 0] = 0.0f;
        velocity[idx * 3 + 1] = 0.0f;
        velocity[idx * 3 + 2] = 0.0f;
        return;
    }
    float worldX = worldMinX + (i + 0.5f) * cellSizeX;
    float worldY = worldMinY + (j + 0.5f) * cellSizeY;
    float worldZ = worldMinZ + (k + 0.5f) * cellSizeZ;
    float3 accum = make_float3(0.0f, 0.0f, 0.0f);
    for(int f = 0; f < numFans; ++f){
        float3 fanPosition = fanPos[f];
        float3 fanDirection = fanDir[f];
        float dx = worldX - fanPosition.x;
        float dy = worldY - fanPosition.y;
        float dz = worldZ - fanPosition.z;
        float distSq = dx * dx + dy * dy + dz * dz + 1e-3f;
        float forceMagnitude = 1.0f / distSq;
        accum.x += fanDirection.x * forceMagnitude;
        accum.y += fanDirection.y * forceMagnitude;
        accum.z += fanDirection.z * forceMagnitude;
    }
    velocity[idx * 3 + 0] += accum.x;
    velocity[idx * 3 + 1] += accum.y;
    velocity[idx * 3 + 2] += accum.z;
}

__global__ void advectVelocityKernel(
    float* velIn,
    float* velOut,
    unsigned char* solidGrid,
    int GX, int GY, int GZ,
    float dt
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= GX || j >= GY || k >= GZ) return;
    int idx = idx3D(i, j, k, GX, GY);
    if(solidGrid[idx] != 0){
        velOut[idx * 3 + 0] = 0.0f;
        velOut[idx * 3 + 1] = 0.0f;
        velOut[idx * 3 + 2] = 0.0f;
        return;
    }
    float vx = velIn[idx * 3 + 0];
    float vy = velIn[idx * 3 + 1];
    float vz = velIn[idx * 3 + 2];
    float x0 = i - vx * dt / cellSizeX;
    float y0 = j - vy * dt / cellSizeY;
    float z0 = k - vz * dt / cellSizeZ;
    x0 = fminf(fmaxf(x0, 0.0f), GX - 1.0f);
    y0 = fminf(fmaxf(y0, 0.0f), GY - 1.0f);
    z0 = fminf(fmaxf(z0, 0.0f), GZ - 1.0f);
    int xi = int(roundf(x0));
    int yi = int(roundf(y0));
    int zi = int(roundf(z0));
    int iidx = idx3D(xi, yi, zi, GX, GY);
    velOut[idx * 3 + 0] = velIn[iidx * 3 + 0];
    velOut[idx * 3 + 1] = velIn[iidx * 3 + 1];
    velOut[idx * 3 + 2] = velIn[iidx * 3 + 2];
}
