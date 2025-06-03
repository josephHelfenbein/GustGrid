#include <iostream>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

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

constexpr int maxPressureIterations = 25;
constexpr float pressureTolerance = 1e-4f;

__device__ bool isValidFluidCell(int x, int y, int z, int GX, int GY, int GZ, unsigned char* solidGrid) {
    if (x < 0 || x >= GX || y < 0 || y >= GY || z < 0 || z >= GZ) return false;
    return solidGrid[idx3D(x, y, z, GX, GY)] == 0;
}

__global__ void computeDivergenceKernel(
    float* velocity,
    float* divergence,
    unsigned char* solidGrid,
    int GX, int GY, int GZ
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= GX || j >= GY || k >= GZ) return;
    int idx = idx3D(i, j, k, GX, GY);
    if(solidGrid[idx] != 0){
        divergence[idx] = 0.0f;
        return;
    }
    float div = 0.0f;
    if(i < GX-1 && i > 0){
        float u_right = isValidFluidCell(i+1, j, k, GX, GY, GZ, solidGrid) ? velocity[idx3D(i+1, j, k, GX, GY) * 3 + 0] : 0.0f;
        float u_left = isValidFluidCell(i-1, j, k, GX, GY, GZ, solidGrid) ? velocity[idx3D(i-1, j, k, GX, GY) * 3 + 0] : 0.0f;
        div += (u_right - u_left) / (2.0f * cellSizeX);
    }
    if(j < GY-1 && j > 0){
        float v_up = isValidFluidCell(i, j+1, k, GX, GY, GZ, solidGrid) ? velocity[idx3D(i, j+1, k, GX, GY) * 3 + 1] : 0.0f;
        float v_down = isValidFluidCell(i, j-1, k, GX, GY, GZ, solidGrid) ? velocity[idx3D(i, j-1, k, GX, GY) * 3 + 1] : 0.0f;
        div += (v_up - v_down) / (2.0f * cellSizeY);
    }
    if(k < GZ-1 && k > 0){
        float w_front = isValidFluidCell(i, j, k+1, GX, GY, GZ, solidGrid) ? velocity[idx3D(i, j, k+1, GX, GY) * 3 + 2] : 0.0f;
        float w_back = isValidFluidCell(i, j, k-1, GX, GY, GZ, solidGrid) ? velocity[idx3D(i, j, k-1, GX, GY) * 3 + 2] : 0.0f;
        div += (w_front - w_back) / (2.0f * cellSizeZ);
    }
    divergence[idx] = div;
}

__global__ void pressureJacobianKernel(
    float* pressureIn,
    float* pressureOut,
    float* divergence,
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
        pressureOut[idx] = 0.0f;
        return;
    }
    float neighborPressureSum = 0.0f;
    int neighborCount = 0;
    int neighbors[6][3] = {
        {-1, 0, 0},
        {1, 0, 0},
        {0, -1, 0},
        {0, 1, 0},
        {0, 0, -1},
        {0, 0, 1}
    };
    for(int n=0; n<6; n++){
        int ni = i + neighbors[n][0];
        int nj = j + neighbors[n][1];
        int nk = k + neighbors[n][2];
        if(ni >= 0 && ni < GX && nj >= 0 && nj < GY && nk >= 0 && nk < GZ){
            int nidx = idx3D(ni, nj, nk, GX, GY);
            if(solidGrid[nidx] == 0){
                neighborPressureSum += pressureIn[nidx];
                neighborCount++;
            }
        }
    }
    if(neighborCount==0){
        pressureOut[idx] = 0.0f;
        return;
    }
    float beta = 0.6f;
    float avgCellSize = (cellSizeX + cellSizeY + cellSizeZ) / 3.0f;
    float scale = avgCellSize * avgCellSize / dt;
    float newPressure = (neighborPressureSum - divergence[idx] * scale) / neighborCount;
    pressureOut[idx] = (1.0f - beta) * pressureIn[idx] + beta * newPressure;
}

__global__ void subtractPressureGradientKernel(
    float* velocity,
    float* pressure,
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
        velocity[idx * 3 + 0] = 0.0f;
        velocity[idx * 3 + 1] = 0.0f;
        velocity[idx * 3 + 2] = 0.0f;
        return;
    }
    float pressureGradientX = 0.0f;
    float pressureGradientY = 0.0f;
    float pressureGradientZ = 0.0f;
    if(i>0 && i<GX-1){
        float pRight = isValidFluidCell(i+1, j, k, GX, GY, GZ, solidGrid) ? pressure[idx3D(i+1, j, k, GX, GY)] : pressure[idx];
        float pLeft = isValidFluidCell(i-1, j, k, GX, GY, GZ, solidGrid) ? pressure[idx3D(i-1, j, k, GX, GY)] : pressure[idx];
        pressureGradientX = (pRight - pLeft) / (2.0f * cellSizeX);
    }
    if(j>0 && j<GY-1){
        float pUp = isValidFluidCell(i, j+1, k, GX, GY, GZ, solidGrid) ? pressure[idx3D(i, j+1, k, GX, GY)] : pressure[idx];
        float pDown = isValidFluidCell(i, j-1, k, GX, GY, GZ, solidGrid) ? pressure[idx3D(i, j-1, k, GX, GY)] : pressure[idx];
        pressureGradientY = (pUp - pDown) / (2.0f * cellSizeY);
    }
    if(k>0 && k<GZ-1){
        float pFront = isValidFluidCell(i, j, k+1, GX, GY, GZ, solidGrid) ? pressure[idx3D(i, j, k+1, GX, GY)] : pressure[idx];
        float pBack = isValidFluidCell(i, j, k-1, GX, GY, GZ, solidGrid) ? pressure[idx3D(i, j, k-1, GX, GY)] : pressure[idx];
        pressureGradientZ = (pFront - pBack) / (2.0f * cellSizeZ);
    }
    velocity[idx * 3 + 0] -= pressureGradientX * dt;
    velocity[idx * 3 + 1] -= pressureGradientY * dt;
    velocity[idx * 3 + 2] -= pressureGradientZ * dt;
}

__global__ void computeResidualKernel(
    float* pressure,
    float* divergence,
    float* residual,
    unsigned char* solidGrid,
    int GX, int GY, int GZ
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= GX || j >= GY || k >= GZ) return;
    int idx = idx3D(i, j, k, GX, GY);
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
    float cellSizes[3] = {cellSizeX, cellSizeY, cellSizeZ};
    for(int n=0; n<6; n++){
        int ni = i + neighbors[n][0];
        int nj = j + neighbors[n][1];
        int nk = k + neighbors[n][2];
        if(ni >= 0 && ni < GX && nj >= 0 && nj < GY && nk >= 0 && nk < GZ){
            int nidx = idx3D(ni, nj, nk, GX, GY);
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

__host__ void solvePressureProjection(
    float* d_velocity,
    float* d_pressureField,
    unsigned char* d_solidGrid,
    int GX, int GY, int GZ,
    float dt
){
    const int numCells = GX * GY * GZ;
    float *d_divergence, *d_pressure, *d_pressureOut, *d_residual;
    CUDA_CHECK(cudaMalloc(&d_divergence, numCells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pressure, numCells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pressureOut, numCells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residual, numCells * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_pressure, 0, numCells * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pressureOut, 0, numCells * sizeof(float)));

    dim3 block(8, 8, 8);
    dim3 grid(
        (GX + block.x - 1) / block.x,
        (GY + block.y - 1) / block.y,
        (GZ + block.z - 1) / block.z
    );
    computeDivergenceKernel<<<grid, block>>>(
        d_velocity, d_divergence, d_solidGrid, GX, GY, GZ
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    float* d_pressure_in = d_pressure;
    float* d_pressure_out = d_pressureOut;
    for(int iter = 0; iter < maxPressureIterations; iter++){
        pressureJacobianKernel<<<grid, block>>>(
            d_pressure_in, d_pressure_out, d_divergence, d_solidGrid, GX, GY, GZ, dt
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        if(iter%5==4 || iter == maxPressureIterations-1){
            computeResidualKernel<<<grid, block>>>(
                d_pressure_in, d_divergence, d_residual, d_solidGrid, GX, GY, GZ
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
        d_velocity, d_pressure_in, d_solidGrid, GX, GY, GZ, dt
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_divergence));
    CUDA_CHECK(cudaFree(d_pressure));
    CUDA_CHECK(cudaFree(d_pressureOut));
    CUDA_CHECK(cudaFree(d_residual));
}

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
        float3 toCell = make_float3(
            worldX - fanPosition.x,
            worldY - fanPosition.y,
            worldZ - fanPosition.z
        );
        float distance = sqrtf(
            toCell.x * toCell.x + 
            toCell.y * toCell.y + 
            toCell.z * toCell.z
        );
        if(distance < 1e-3f) continue;
        float3 toCellNormalized = make_float3(
            toCell.x / distance,
            toCell.y / distance,
            toCell.z / distance
        );
        float alignment = 
            fanDirection.x * toCellNormalized.x +
            fanDirection.y * toCellNormalized.y +
            fanDirection.z * toCellNormalized.z;
        if(alignment > 0.1f){
            float fanRadius = 1.0f;
            float forceMagnitude = 5.0f * alignment / (1.0f + (distance * distance) / (fanRadius * fanRadius));
            accum.x += fanDirection.x * forceMagnitude;
            accum.y += fanDirection.y * forceMagnitude;
            accum.z += fanDirection.z * forceMagnitude;
        }
    }
    velocity[idx * 3 + 0] += accum.x;
    velocity[idx * 3 + 1] += accum.y;
    velocity[idx * 3 + 2] += accum.z;
    const float maxVelocity = 10.0f;
    velocity[idx * 3 + 0] = fminf(fmaxf(velocity[idx * 3 + 0], -maxVelocity), maxVelocity);
    velocity[idx * 3 + 1] = fminf(fmaxf(velocity[idx * 3 + 1], -maxVelocity), maxVelocity);
    velocity[idx * 3 + 2] = fminf(fmaxf(velocity[idx * 3 + 2], -maxVelocity), maxVelocity);
}

__global__ void applyDampeningKernel(
    float* velocity,
    unsigned char* solidGrid,
    int GX, int GY, int GZ,
    float dampeningFactor
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
    velocity[idx * 3 + 0] *= dampeningFactor;
    velocity[idx * 3 + 1] *= dampeningFactor;
    velocity[idx * 3 + 2] *= dampeningFactor;
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
    float advectionStrength = 0.5f;
    float x0 = i - vx * dt * advectionStrength / cellSizeX;
    float y0 = j - vy * dt * advectionStrength / cellSizeY;
    float z0 = k - vz * dt * advectionStrength / cellSizeZ;
    x0 = fminf(fmaxf(x0, 0.5f), GX - 1.5f);
    y0 = fminf(fmaxf(y0, 0.5f), GY - 1.5f);
    z0 = fminf(fmaxf(z0, 0.5f), GZ - 1.5f);
    int xi = int(x0);
    int yi = int(y0);
    int zi = int(z0);
    float fx = x0 - xi;
    float fy = y0 - yi;
    float fz = z0 - zi;
    xi = max(0, min(xi, GX - 2));
    yi = max(0, min(yi, GY - 2));
    zi = max(0, min(zi, GZ - 2));

    for(int comp = 0; comp < 3; comp++){
        float c000 = velIn[idx3D(xi, yi, zi, GX, GY) * 3 + comp];
        float c001 = velIn[idx3D(xi, yi, zi+1, GX, GY) * 3 + comp];
        float c010 = velIn[idx3D(xi, yi+1, zi, GX, GY) * 3 + comp];
        float c011 = velIn[idx3D(xi, yi+1, zi+1, GX, GY) * 3 + comp];
        float c100 = velIn[idx3D(xi+1, yi, zi, GX, GY) * 3 + comp];
        float c101 = velIn[idx3D(xi+1, yi, zi+1, GX, GY) * 3 + comp];
        float c110 = velIn[idx3D(xi+1, yi+1, zi, GX, GY) * 3 + comp];
        float c111 = velIn[idx3D(xi+1, yi+1, zi+1, GX, GY) * 3 + comp];
        
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
    float* d_tempVelocity;
    CUDA_CHECK(cudaMalloc(&d_tempVelocity, numCells * 3 * sizeof(float)));
    addFanForcesKernel<<<grid, block>>>(
        d_velocityField, d_solidGrid, d_fanPositions, d_fanDirections, numFans, gridSizeX, gridSizeY, gridSizeZ
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    applyDampeningKernel<<<grid, block>>>(
        d_velocityField, d_solidGrid, gridSizeX, gridSizeY, gridSizeZ, 0.95f
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    advectVelocityKernel<<<grid, block>>>(
        d_velocityField, d_tempVelocity, d_solidGrid, gridSizeX, gridSizeY, gridSizeZ, dt
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_velocityField, d_tempVelocity, numCells * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
    float* d_pressureFieldTemp;
    CUDA_CHECK(cudaMalloc(&d_pressureFieldTemp, numCells * sizeof(float)));
    solvePressureProjection(
        d_velocityField, d_pressureFieldTemp, d_solidGrid, gridSizeX, gridSizeY, gridSizeZ, dt
    );
    CUDA_CHECK(cudaMemcpy(d_pressureField, d_pressureFieldTemp, numCells * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(d_tempVelocity));
}