#version 330 core
layout(location = 0) in vec3 aPos;

out vec3 FragPos;
out vec3 texCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform vec3 worldMin;
uniform vec3 worldMax;

void main(){
    vec4 worldPosition = model * vec4(aPos, 1.0);
    FragPos = worldPosition.xyz;
    texCoord = (FragPos - worldMin) / (worldMax - worldMin);
    gl_Position = projection * view * worldPosition;
}