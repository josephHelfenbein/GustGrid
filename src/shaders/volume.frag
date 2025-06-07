#version 330 core

in vec3 FragPos;
in vec3 texCoord;
out vec4 FragColor;

uniform sampler3D volumeTex;
uniform sampler3D temperatureTex;
uniform int displayPressure;
uniform vec3 camPos;
uniform vec3 gridSize;
uniform float stepSize;
uniform vec3 worldMin;
uniform vec3 worldMax;

void main(){
    vec3 rayDir = normalize(FragPos - camPos);
    float accumAlpha = 0.0;
    vec3 accumColor = vec3(0.0);

    vec3 worldSize = worldMax - worldMin;
    vec3 voxelSize = worldSize / gridSize;

    float maxVoxelDim = max(max(voxelSize.x, voxelSize.y), voxelSize.z);
    float baseWorldStep = maxVoxelDim * stepSize;
    float worldStep = baseWorldStep;
    int emptySteps = 0;

    vec3 currentWorldPos = FragPos;
    vec3 currentTexCoord = texCoord;

    int maxSteps = int(length(worldSize) / baseWorldStep) + 1;

    float opacity = displayPressure == 1 ? 0.05 : 0.002;

    float ambientTemp = 22.0;
    float maxExpectedTemp = 100.0;
    float midTemp = (ambientTemp + maxExpectedTemp) * 0.5;

    for(int i = 0; i < maxSteps; i++){
        if(any(lessThan(currentTexCoord, vec3(0.0))) || any(greaterThan(currentTexCoord, vec3(1.0)))) break;
        
        float volumeVal = texture(volumeTex, currentTexCoord).r;
        float tempValue = texture(temperatureTex, currentTexCoord).r;
        if(volumeVal < 1e-4 && tempValue < ambientTemp + 0.1){
            emptySteps++;
            if(emptySteps > 2.0) worldStep = min(baseWorldStep * 8.0, baseWorldStep * (1.0 + float(emptySteps - 2)));
            currentWorldPos += rayDir * worldStep;
            currentTexCoord = (currentWorldPos - worldMin) / (worldMax - worldMin);
            continue;
        } else{
            emptySteps = 0;
            worldStep = baseWorldStep;
        }

        float normalizedTemp = clamp((tempValue - ambientTemp) / (maxExpectedTemp - ambientTemp), 0.0, 1.0);

        float tempAlpha = normalizedTemp * 0.01;
        float volumeAlpha = clamp(volumeVal * stepSize * opacity, 0.0, 1.0);
        float totalAlpha = max(volumeAlpha, tempAlpha);
        vec3 color = vec3(0.0);
        if(normalizedTemp<0.5){
            color += mix(vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), normalizedTemp*2.0);
            color += mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, 0.0), normalizedTemp*2.0);
        }
        else{
            color += mix(vec3(0.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), (normalizedTemp - 0.5) * 2.0);
            color += mix(vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 0.0), (normalizedTemp - 0.5) * 2.0);
        }
        float alphaToAdd = totalAlpha * (1.0 - accumAlpha);
        accumAlpha += alphaToAdd;
        accumColor += color * alphaToAdd;
        
        if(accumAlpha > 0.95) break;
        
        currentWorldPos += rayDir * worldStep;
        currentTexCoord = (currentWorldPos - worldMin) / (worldMax - worldMin);
    }
    FragColor = vec4(vec3(accumColor), accumAlpha);
}