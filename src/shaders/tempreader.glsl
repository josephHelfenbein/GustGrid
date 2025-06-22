#version 430 core

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

uniform sampler3D temperatureTex;
ivec3 gridSize = {64, 256, 128};

layout(std430, binding = 0) buffer TemperatureBuffer {
    float temperatures[];
};

//  (-1.45f+2.0f)/(4.0f/gridSizeX)-0.5f, (2.45f+4.5f)/(9.0f/gridSizeY)-0.5f, (1.45f+4.0f)/(8.0f/gridSizeZ)-0.5f)
ivec3 cpuIndex = ivec3(8, 197, 87);
//  (0.25f+2.0f)/(4.0f/gridSizeX)-0.5f, (0.0f+4.5f)/(9.0f/gridSizeY)-0.5f, (3.5f+4.0f)/(8.0f/gridSizeZ)-0.5f)
ivec3 gpuIndex = ivec3(35, 154, 119);

void main() {
    uint index = gl_GlobalInvocationID.x;
    
    if (index == 0) {
        vec3 cpuTexCoord = (vec3(cpuIndex) + 0.5) / vec3(gridSize);
        temperatures[0] = texture(temperatureTex, cpuTexCoord).r;
    }
    else if (index == 1) {
        vec3 gpuTexCoord = (vec3(gpuIndex) + 0.5) / vec3(gridSize);
        temperatures[1] = texture(temperatureTex, gpuTexCoord).r;
    }
}