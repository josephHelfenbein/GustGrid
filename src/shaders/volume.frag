#version 330 core

in vec3 FragPos;
out vec4 FragColor;

uniform sampler3D volumeTex;
uniform vec3 camPos;
uniform vec3 gridSize;
uniform float stepSize;

void main(){
    vec3 rayDir = normalize(FragPos - camPos);
    float accumColor = 0.0;
    float accumAlpha = 0.0f;

    vec3 pos = FragPos;
    for(int i=0; i<int(gridSize.x); i++){
        if(pos.x < 0.0 || pos.x >= 1.0 ||
           pos.y < 0.0 || pos.y >= 1.0 ||
           pos.z < 0.0 || pos.z >= 1.0) {
            break;
        }
        float val = texture(volumeTex, pos).r;
        float density = clamp(val * 2.0, 0.0, 1.0);
        float a = density * stepSize;
        accumColor += (1.0 - accumAlpha) * a * density;
        accumAlpha += (1.0 - accumAlpha) * a;
        if(accumAlpha > 0.95) break;
        pos += rayDir * stepSize;
    }
    FragColor = vec4(vec3(accumColor), accumAlpha);
}