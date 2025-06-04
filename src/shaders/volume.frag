#version 330 core

in vec3 FragPos;
in vec3 texCoord;
out vec4 FragColor;

uniform sampler3D volumeTex;
uniform int displayPressure;
uniform vec3 camPos;
uniform vec3 gridSize;
uniform float stepSize;
uniform vec3 worldMin;
uniform vec3 worldMax;

void main(){
    vec3 rayDir = normalize(FragPos - camPos);
    float accumColor = 0.0;

    vec3 worldSize = worldMax - worldMin;
    vec3 voxelSize = worldSize / gridSize;

    float minVoxelDim = min(min(voxelSize.x, voxelSize.y), voxelSize.z);
    float worldStep = minVoxelDim * stepSize;
    vec3 currentWorldPos = FragPos;
    vec3 currentTexCoord = texCoord;

    int maxSteps = int(length(worldSize) / worldStep) + 1;

    float opacity = displayPressure == 1 ? 0.05 : 0.002;

    for(int i = 0; i < maxSteps; i++){
        if(any(lessThan(currentTexCoord, vec3(0.0))) || any(greaterThan(currentTexCoord, vec3(1.0)))) break;
        
        float volumeVal = texture(volumeTex, currentTexCoord).r;
        float col = clamp(volumeVal * stepSize * opacity, 0.0, 1.0);
        
        accumColor += (1.0 - accumColor) * col;
        
        if(accumColor > 0.95) break;
        
        currentWorldPos += rayDir * worldStep;
        currentTexCoord = (currentWorldPos - worldMin) / (worldMax - worldMin);
    }
    FragColor = vec4(accumColor);
}