#version 330 core
in vec3 normalVec;
in vec2 texCoord;
in vec3 FragPos;

const float PI = 3.14159265358979323846;

uniform vec3 camPos;

out vec4 FragColor;

void main(){
    vec3 albedo = vec3(0.5, 0.5, 0.75);
    vec3 normal = normalize(normalVec);

    vec3 viewDir = normalize(camPos - FragPos);
    vec3 reflectionDir = reflect(-viewDir, normal);

    vec3 lightDir = normalize(vec3(2.0, 4.0, 3.0));
    highp float diffuse = max(dot(normal, lightDir), 0.015);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
    vec3 specular = vec3(spec) * 0.1;

    float gamma = 2.2;
    FragColor = vec4(pow(specular + albedo * vec3(diffuse), vec3(1.0 / gamma)), 1.0);
}