#version 330 core

in vec3 FragPos;
in vec3 texCoord;
out vec4 FragColor;

uniform sampler3D volumeTex;
uniform vec3 camPos;
uniform vec3 gridSize;
uniform float stepSize;
uniform vec3 worldMin;
uniform vec3 worldMax;

void main(){
    vec3 rayDir = normalize(FragPos - camPos);
    float accumColor = 0.0;
    float accumAlpha = 0.0;

    vec3 worldSize = worldMax - worldMin;
    vec3 voxelSize = worldSize / gridSize;
    float worldStep = float(voxelSize) * stepSize;
    vec3 currentWorldPos = FragPos;
    vec3 currentTexCoord = texCoord;

    int maxSteps = int(length(worldSize) / worldStep) + 1;

    for(int i = 0; i < maxSteps; i++){
        if(any(lessThan(currentTexCoord, vec3(0.0))) || any(greaterThan(currentTexCoord, vec3(1.0)))) break;
        
        float val = texture(volumeTex, currentTexCoord).r;
        float density = clamp(val * 0.5, 0.0, 1.0);
        float a = density * stepSize;
        
        accumColor += (1.0 - accumAlpha) * a * density;
        accumAlpha += (1.0 - accumAlpha) * a;
        
        if(accumAlpha > 0.95) break;
        
        currentWorldPos += rayDir * worldStep;
        currentTexCoord = (currentWorldPos - worldMin) / (worldMax - worldMin);
    }
    FragColor = vec4(vec3(accumColor), accumAlpha);
}