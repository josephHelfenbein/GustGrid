#include <iostream>
#include <glad/glad.h>
#include <glfw/include/GLFW/glfw3.h>
#include <glm/glm/glm.hpp>
#include <glm/glm/gtc/matrix_transform.hpp>
#include <glm/glm/gtc/type_ptr.hpp>
#include <OBJ-Loader/Source/OBJ_Loader.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <freetype/include/ft2build.h>
#include FT_FREETYPE_H
#include <map>
#include <vector>
#include <thread>

#define uiVertexPath "./src/shaders/ui.vert"
#define uiFragmentPath "./src/shaders/ui.frag"
#define textFragmentPath "./src/shaders/text.frag"
#define checkboxCheckedSource "./src/textures/ui/checkbox/checked.png"
#define checkboxUncheckedSource "./src/textures/ui/checkbox/unchecked.png"
#define sliderKnobSource "./src/textures/ui/slider/knob.png"
#define arrowUpSource "./src/textures/ui/arrow/up.png"
#define arrowDownSource "./src/textures/ui/arrow/down.png"
#define uiWindowSource "./src/textures/ui/uiWindow.png"
#define uiCloseSource "./src/textures/ui/toggle/close.png"
#define uiOpenSource "./src/textures/ui/toggle/open.png"
#define caseSource "./src/models/case.obj"
#define cpuSource "./src/models/cpu.obj"
#define gpuSource "./src/models/gpu.obj"
#define ramSource "./src/models/ram.obj"
#define motherboardSource "./src/models/motherboard.obj"
#define psuSource "./src/models/psu.obj"
#define ioShieldSource "./src/models/ioshield.obj"
#define shieldSource "./src/models/shield.obj"
#define glassSource "./src/models/glass.obj"
#define topFanSource "./src/models/topfan.obj"
#define frontFanSource "./src/models/frontfan.obj"
#define backFanSource "./src/models/backfan.obj"
#define cpuFanSource "./src/models/cpufan.obj"
#define volumeVertexShaderPath "./src/shaders/volume.vert"
#define volumeFragmentShaderPath "./src/shaders/volume.frag"
#define vertexShaderPath "./src/shaders/main.vert"
#define fragmentShaderPath "./src/shaders/main.frag"
#define PI 3.14159265358979323846f
#define worldMinX -2.0f
#define worldMaxX 2.0f
#define worldMinY -4.5f
#define worldMaxY 4.5f
#define worldMinZ -4.0f
#define worldMaxZ 4.0f
#define renderMinX -1.65f
#define renderMaxX 1.7f
#define renderMinY -3.0f
#define renderMaxY 4.22f
#define renderMinZ -3.4f
#define renderMaxZ 3.4f
#define gridSizeX 64
#define gridSizeY 256
#define gridSizeZ 128

unsigned int SCR_WIDTH = 800;
unsigned int SCR_HEIGHT = 600;
float camYaw = PI / 8;
float camPitch = PI / 12;
float camRadius = 15.0f;
glm::vec3 camPos = glm::vec3(sin(camYaw) * camRadius, sin(camPitch) * camRadius, cos(camYaw) * camRadius);
float camFOV = 45.0f;

float deltaTime = 0.0f;
float lastFrame = 0.0f;
bool firstMouse = true;
bool mousePressed = false;
float lastMouseX = SCR_WIDTH / 2.0f;
float lastMouseY = SCR_HEIGHT / 2.0f;
float mouseSensitivity = 0.007f;
bool showUI = true;
bool* itemChangedPtr = nullptr;
bool* runningPtr = nullptr;

std::vector<unsigned int> VAOs;
std::vector<unsigned int> buffers;

const char* caseTexturesSource[4] = {
    "./src/textures/case/basecolor.png",
    "./src/textures/case/metallic.png",
    "./src/textures/case/roughness.png",
    "./src/textures/case/normal.png"
};
const char* cpuTexturesSource[4] = {
    "./src/textures/cpu/basecolor.png",
    "./src/textures/cpu/metallic.png",
    "./src/textures/cpu/roughness.png",
    "./src/textures/cpu/normal.png"
};
const char* gpuTexturesSource[4] = {
    "./src/textures/gpu/basecolor.png",
    "./src/textures/gpu/metallic.png",
    "./src/textures/gpu/roughness.png",
    "./src/textures/gpu/normal.png"
};
const char* ramTexturesSource[4] = {
    "./src/textures/ram/basecolor.png",
    "./src/textures/ram/metallic.png",
    "./src/textures/ram/roughness.png",
    "./src/textures/ram/normal.png"
};
const char* motherboardTexturesSource[4] = {
    "./src/textures/motherboard/basecolor.png",
    "./src/textures/motherboard/metallic.png",
    "./src/textures/motherboard/roughness.png",
    "./src/textures/motherboard/normal.png"
};
const char* psuTexturesSource[4] = {
    "./src/textures/psu/basecolor.png",
    "./src/textures/psu/metallic.png",
    "./src/textures/psu/roughness.png",
    "./src/textures/psu/normal.png"
};
const char* ioShieldTexturesSource[4] = {
    "./src/textures/ioshield/basecolor.png",
    "./src/textures/ioshield/metallic.png",
    "./src/textures/ioshield/roughness.png",
    "./src/textures/ioshield/normal.png"
};
const char* shieldTexturesSource[4] = {
    "./src/textures/shield/basecolor.png",
    "./src/textures/shield/metallic.png",
    "./src/textures/shield/roughness.png",
    "./src/textures/shield/normal.png"
};
const char* glassTexturesSource[4] = {
    "./src/textures/glass/basecolor.png",
    "./src/textures/glass/metallic.png",
    "./src/textures/glass/roughness.png",
    "./src/textures/glass/normal.png"
};

float checkboxYPositions[4] = {100.0f, 130.0f, 160.0f, 190.0f};
bool *checkboxItems[4];
float sliderYPositions[3];
float *sliderXValues[3];
float possibleSliderXValues[3];
int hoverElement = -1;
bool *displayingPressure = nullptr;

void framebufferSizeCallback(GLFWwindow* window, int width, int height){
    glViewport(0, 0, width, height);
    SCR_WIDTH = width;
    SCR_HEIGHT = height;
}
void mouseCallback(GLFWwindow* window, double xpos, double ypos){
    float xposFloat = static_cast<float>(xpos);
    float yposFloat = static_cast<float>(ypos);

    if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE){
        lastMouseX = xposFloat;
        lastMouseY = yposFloat;
        firstMouse = true;

        if(showUI){
            bool hoveringElement = false;
            for(int i=0; i<4; i++){
                if(
                    yposFloat < checkboxYPositions[i]
                &&  yposFloat > checkboxYPositions[i] - 20.0f
                &&  xposFloat > SCR_WIDTH - 75.0f
                &&  xposFloat < SCR_WIDTH - 55.0f) {
                    hoverElement = i;
                    hoveringElement = true;
                    break;
                }
            }
            if(!hoveringElement) hoverElement = -1;
            if(
                yposFloat > 200.0f
            &&  yposFloat < 210.0f
            &&  xposFloat > SCR_WIDTH - 100.0f
            &&  xposFloat < SCR_WIDTH - 80.0f) hoverElement = 4;
            else if(
                yposFloat > 210.0f
            &&  yposFloat < 220.0f
            &&  xposFloat > SCR_WIDTH - 100.0f
            &&  xposFloat < SCR_WIDTH - 80.0f) hoverElement = 5;
            else if(
                yposFloat > 30.0f
            &&  yposFloat < 50.0f
            &&  xposFloat > SCR_WIDTH - 245.0f
            &&  xposFloat < SCR_WIDTH - 225.0f) hoverElement = 6;
            else if(
                yposFloat > 430.0f
            &&  yposFloat < 450.0f
            &&  xposFloat > SCR_WIDTH - 75.0f
            &&  xposFloat < SCR_WIDTH - 55.0f) hoverElement = 7;
        } else if(
            yposFloat > 30.0f
        &&  yposFloat < 50.0f
        &&  xposFloat > SCR_WIDTH - 25.0f
        &&  xposFloat < SCR_WIDTH - 5.0f) hoverElement = 6;

        return;
    }
    else if(showUI){
        for(int i=0; i<3; i++){
            if(sliderYPositions[i] == 1.0) continue;
            float minValue = i==0 ? 0.0f : *sliderXValues[i-1] - 2.5f;
            float maxValue = i==2 || *sliderXValues[i+1] == 1.0 ? -5.0f : *sliderXValues[i+1] + 2.5f;
            if(
                yposFloat < sliderYPositions[i]
            &&  yposFloat > sliderYPositions[i] - 20.0f
            &&  xposFloat > SCR_WIDTH - 140.0f
            &&  xposFloat < SCR_WIDTH - 60.0f) {
                possibleSliderXValues[i] = 5 * (-xposFloat + (SCR_WIDTH - 140.0f)) / 80.0f;
                if(possibleSliderXValues[i] > minValue) possibleSliderXValues[i] = minValue;
                else if(possibleSliderXValues[i] < maxValue) possibleSliderXValues[i] = maxValue;
                *sliderXValues[i] = possibleSliderXValues[i];
                *itemChangedPtr = true;
                return;
            }
            else possibleSliderXValues[i] = *sliderXValues[i];
        }
        if(
            yposFloat > 50.0f
        &&  yposFloat < 500.0f
        &&  xposFloat > SCR_WIDTH - 220.0f
        &&  xposFloat < SCR_WIDTH - 20.0f
        ) return;
    } else if(
        yposFloat > 30.0f
    &&  yposFloat < 50.0f
    &&  xposFloat > SCR_WIDTH - 25.0f
    &&  xposFloat < SCR_WIDTH - 5.0f) return;

    if(firstMouse){
        lastMouseX = xposFloat;
        lastMouseY = yposFloat;
        firstMouse = false;
    }
    
    float xOffset = (xposFloat - lastMouseX) * mouseSensitivity;
    float yOffset = (lastMouseY - yposFloat) * mouseSensitivity;
    lastMouseX = xposFloat;
    lastMouseY = yposFloat;
    camYaw += xOffset;
    camPitch += yOffset;
    if(camPitch > 1.6f) camPitch = 1.6f;
    else if(camPitch < -1.6f) camPitch = -1.6f;
    camPos = glm::vec3(sin(camYaw) * camRadius, sin(camPitch) * camRadius, cos(camYaw) * camRadius);
}
void processInput(GLFWwindow* window){
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        *runningPtr = false;
        glfwSetWindowShouldClose(window, true);
    }
    else if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS){
        if(!firstMouse) return;
        if(hoverElement == -1) return;
        if(hoverElement < 4) *checkboxItems[hoverElement] = !(*checkboxItems[hoverElement]);
        else if(hoverElement == 4){
            int lastIndex = 0;
            if(*sliderXValues[0] != 1.0f) lastIndex = 1;
            if(*sliderXValues[1] != 1.0f) lastIndex = 2;
            if(*sliderXValues[2] != 1.0f) lastIndex = 3;
            if(lastIndex<3) for(int i=0; i<=lastIndex; i++) *sliderXValues[i] = i==0 ? 0.0f : *sliderXValues[i-1] - 2.5f;
        }
        else if(hoverElement == 5){
            int lastIndex = 2;
            if(*sliderXValues[2] == 1.0f) lastIndex = 1;
            if(*sliderXValues[1] == 1.0f) lastIndex = 0;
            *sliderXValues[lastIndex] = 1.0f;
        }
        else if(hoverElement == 6) {
            showUI = !showUI;
            hoverElement = -1;
        }
        else if(hoverElement == 7) *displayingPressure = !(*displayingPressure);
        firstMouse = false;
        *itemChangedPtr = true;
    }
    else if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) firstMouse = true;   
}
char* getShaders(const char* file){
    FILE* shaderFile = fopen(file, "r");
    if(!shaderFile){
        std::cerr<<"Error opening shader file at "<<file<<"\n";
        return nullptr;
    }
    int fileSize = 0;
    fseek(shaderFile, 0, SEEK_END);
    fileSize = ftell(shaderFile);
    rewind(shaderFile);
    char* shader = (char*) malloc(sizeof(char) * (fileSize + 1));
    fread(shader, sizeof(char), fileSize, shaderFile);
    shader[fileSize] = '\0';
    fclose(shaderFile);
    return shader;
}
unsigned int createShader(const char* vertSource, const char* fragSource){
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertSource, nullptr);
    glCompileShader(vertexShader);
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if(!success){
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        std::cerr<<"A vertex shader compilation failed.\n"<<infoLog<<std::endl;
    }
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragSource, nullptr);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if(!success){
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        std::cerr<<"A fragment shader compilation failed.\n"<<infoLog<<std::endl;
    }
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if(!success){
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr<<"Linking shader program failed.\n"<<infoLog<<std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return shaderProgram;
}
struct ModelData{
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    bool loaded = false;
};
void loadModelData(const char* file, ModelData &data){
    objl::Loader loader;
    if(!loader.LoadFile(file)){
        std::cerr<<"Failed to load OBJ file"<<std::endl;
        return;
    }
    objl::Mesh mesh = loader.LoadedMeshes[0];
    data.vertices.resize(mesh.Vertices.size() * 8);
    int i = 0;
    for(const auto &vertex : mesh.Vertices){
        data.vertices[i++] = vertex.Position.X;
        data.vertices[i++] = vertex.Position.Y;
        data.vertices[i++] = vertex.Position.Z;
        data.vertices[i++] = vertex.TextureCoordinate.X;
        data.vertices[i++] = 1.0f - vertex.TextureCoordinate.Y;
        data.vertices[i++] = vertex.Normal.X;
        data.vertices[i++] = vertex.Normal.Y;
        data.vertices[i++] = vertex.Normal.Z;
    }
    data.indices = mesh.Indices;
    data.loaded = true;
}
void createGLObjects(const ModelData &data, unsigned int &VAO, unsigned int &VBO, unsigned int &EBO, unsigned int &indexCount){
    if(!data.loaded){
        std::cerr<<"Model data not loaded."<<std::endl;
        return;
    }
    indexCount = data.indices.size();
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, data.vertices.size() * sizeof(float), &data.vertices[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexCount * sizeof(unsigned int), &data.indices[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*) 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*) (3 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*) (5 * sizeof(float)));
    VAOs.push_back(VAO);
    buffers.push_back(VBO);
    buffers.push_back(EBO);
}
unsigned int loadTexture(const char* file){
    unsigned int textureID;
    glGenTextures(1, &textureID);
    int width, height, channels;
    unsigned char* data = stbi_load(file, &width, &height, &channels, 0);
    if(data){
        GLenum format;
        if(channels == 1) format = GL_RED;
        else if(channels == 3) format = GL_RGB;
        else if(channels == 4) format = GL_RGBA;
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        stbi_image_free(data);
    } else{
        std::cerr<<"Failed to load texture image at path "<<file<<std::endl;
        stbi_image_free(data);
    }
    return textureID;
}
void drawObject(unsigned int textureIDs[], unsigned int shaderProgram, unsigned int VAO, unsigned int indexCount){
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureIDs[0]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, textureIDs[1]);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, textureIDs[2]);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, textureIDs[3]);
    glUniform1i(glGetUniformLocation(shaderProgram, "albedoMap"), 0);
    glUniform1i(glGetUniformLocation(shaderProgram, "metallicMap"), 1);
    glUniform1i(glGetUniformLocation(shaderProgram, "roughnessMap"), 2);
    glUniform1i(glGetUniformLocation(shaderProgram, "normalMap"), 3);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
}
struct Character{
    unsigned int textureID;
    glm::ivec2 size;
    glm::ivec2 bearing;
    unsigned int advance;
};
std::map<char, Character> Characters;
void prepareCharacters(){
    FT_Library ft;
    if(FT_Init_FreeType(&ft)){
        std::cerr<<"Could not initialize FreeType Library"<<std::endl;
        return;
    }
    FT_Face face;
    if(FT_New_Face(ft, "./src/fonts/Lato.ttf", 0, &face)){
        std::cerr<<"Could not load font at ./src/fonts/Lato.ttf"<<std::endl;
        return;
    }
    FT_Set_Pixel_Sizes(face, 0, 48);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    for(unsigned char c = 0; c < 128; c++){
        if(FT_Load_Char(face, c, FT_LOAD_RENDER)){
            std::cerr<<"Failed to load glyph: "<<c<<std::endl;
            continue;
        }
        unsigned int textureID;
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, face->glyph->bitmap.width, face->glyph->bitmap.rows, 0, GL_RED, GL_UNSIGNED_BYTE, face->glyph->bitmap.buffer);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        Character character = { 
                                textureID, 
                                glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
                                glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
                                static_cast<unsigned int>(face->glyph->advance.x)
                              };
        Characters[c] = character;
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    FT_Done_Face(face);
    FT_Done_FreeType(ft);
}
void drawText(unsigned int shader, unsigned int VAO, unsigned int VBO, const char* text, glm::vec2 position, float scale, glm::vec3 color){
    glUseProgram(shader);
    glm::mat4 textProj = glm::ortho(0.0f, (float) SCR_WIDTH, 0.0f, (float) SCR_HEIGHT);
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, &textProj[0][0]);
    glUniform3f(glGetUniformLocation(shader, "textColor"), color.x, color.y, color.z);
    glm::mat4 spriteModel = glm::mat4(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, &spriteModel[0][0]);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(VAO);
    for(const char* p = text; *p; p++){
        Character ch = Characters[*p];
        float xPos = position.x + ch.bearing.x * scale;
        float yPos = position.y - (ch.size.y - ch.bearing.y) * scale;
        float w = ch.size.x * scale;
        float h = ch.size.y * scale;
        float vertices[6][4] = {
            {xPos, yPos + h, 0.0f, 0.0f},
            {xPos, yPos, 0.0f, 1.0f},
            {xPos + w, yPos, 1.0f, 1.0f},
            {xPos, yPos + h, 0.0f, 0.0f},
            {xPos + w, yPos, 1.0f, 1.0f},
            {xPos + w, yPos + h, 1.0f, 0.0f}
        };
        glBindTexture(GL_TEXTURE_2D, ch.textureID);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        position.x += (ch.advance >> 6) * scale;
    }
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}
void drawSprite(unsigned int shader, unsigned int VAO, unsigned int VBO, glm::vec2 position, glm::vec2 size, glm::vec3 color, unsigned int textureID){
    glUseProgram(shader);
    glm::mat4 spriteProj = glm::ortho(0.0f, (float) SCR_WIDTH, 0.0f, (float) SCR_HEIGHT);
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, &spriteProj[0][0]);
    glUniform3f(glGetUniformLocation(shader, "tintColor"), color.x, color.y, color.z);
    glm::mat4 spriteModel = glm::mat4(1.0f);
    spriteModel = glm::translate(spriteModel, glm::vec3(position.x, position.y, 0.0f));
    spriteModel = glm::scale(spriteModel, glm::vec3(size.x, size.y, 1.0f));
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, &spriteModel[0][0]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glUniform1i(glGetUniformLocation(shader, "image"), 0);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}
void drawCheckbox(unsigned int shader, unsigned int VAO, unsigned int VBO, glm::vec2 position, glm::vec2 size, bool checked, unsigned int checkedTextureID, unsigned int uncheckedTextureID){
    glUseProgram(shader);
    glm::mat4 checkboxProj = glm::ortho(0.0f, (float) SCR_WIDTH, 0.0f, (float) SCR_HEIGHT);
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, &checkboxProj[0][0]);
    glUniform3f(glGetUniformLocation(shader, "tintColor"), 1.0f, 1.0f, 1.0f);
    glm::mat4 checkboxModel = glm::mat4(1.0f);
    checkboxModel = glm::translate(checkboxModel, glm::vec3(position.x, position.y, 0.0f));
    checkboxModel = glm::scale(checkboxModel, glm::vec3(size.x, size.y, 1.0f));
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, &checkboxModel[0][0]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, checked ? checkedTextureID : uncheckedTextureID);
    glUniform1i(glGetUniformLocation(shader, "image"), 0);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}
void drawSlider(unsigned int shader, unsigned int VAO, unsigned int VBO, glm::vec2 position, glm::vec2 size, float value, unsigned int sliderTextureID, unsigned int knobTextureID){
    glUseProgram(shader);
    float knobX = position.x - value / 5 * size.x - size.y / 2;
    glm::mat4 knobModel = glm::mat4(1.0f);
    knobModel = glm::translate(knobModel, glm::vec3(knobX, position.y, 0.0f));
    knobModel = glm::scale(knobModel, glm::vec3(size.y, size.y, 1.0f));
    glUniform3f(glGetUniformLocation(shader, "tintColor"), 1.0f, 1.0f, 1.0f);
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, &knobModel[0][0]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, knobTextureID);
    glUniform1i(glGetUniformLocation(shader, "image"), 0);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glm::mat4 sliderProj = glm::ortho(0.0f, (float) SCR_WIDTH, 0.0f, (float) SCR_HEIGHT);
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, &sliderProj[0][0]);
    glm::mat4 sliderModel = glm::mat4(1.0f);
    sliderModel = glm::translate(sliderModel, glm::vec3(position.x, position.y, 0.0f));
    sliderModel = glm::scale(sliderModel, glm::vec3(size.x, size.y, 1.0f));
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, &sliderModel[0][0]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, sliderTextureID);
    glUniform1i(glGetUniformLocation(shader, "image"), 1);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}
void drawArrowInput(unsigned int shader, unsigned int VAO, unsigned int VBO, glm::vec2 position, glm::vec2 size, unsigned int arrowUpTexture, unsigned int arrowDownTexture){
    glUseProgram(shader);
    glm::mat4 arrowProj = glm::ortho(0.0f, (float) SCR_WIDTH, 0.0f, (float) SCR_HEIGHT);
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, &arrowProj[0][0]);
    glUniform3f(glGetUniformLocation(shader, "tintColor"), 1.0f, 1.0f, 1.0f);
    glm::mat4 arrowModel = glm::mat4(1.0f);
    arrowModel = glm::translate(arrowModel, glm::vec3(position.x, position.y + size.y / 2.0f, 0.0f));
    arrowModel = glm::scale(arrowModel, glm::vec3(size.x, size.y / 2.0f, 1.0f));
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, &arrowModel[0][0]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, arrowDownTexture);
    glUniform1i(glGetUniformLocation(shader, "image"), 0);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    arrowModel = glm::translate(arrowModel, glm::vec3(0.0f, -1.0f, 0.0f));
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, &arrowModel[0][0]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, arrowUpTexture);
    glUniform1i(glGetUniformLocation(shader, "image"), 0);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}
int startRenderer(bool &gpuEnabled, bool &topFanEnabled, bool &cpuFanEnabled, bool &frontFanEnabled, float* backFanLocations, float* velocityField, float* pressureField, bool &itemChanged, bool &running, std::function<void()> waitForVelocityField, std::function<void()> signalItemsReady, bool &displayPressure, float* temperatureField){
    if(!glfwInit()){
        std::cerr<<"Failed to initialize GLFW"<<std::endl;
        return -1;
    }
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window;
    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "GustGrid", nullptr, nullptr);
    if(window == nullptr){
        std::cerr<<"Failed to create GLFW window"<<std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetCursorPosCallback(window, mouseCallback);

    if(!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)){
        std::cerr<<"Failed to initialize GLAD"<<std::endl;
        return -1;
    }
    
    glEnable(GL_DEPTH_TEST);

    std::array<ModelData, 13> models;

    unsigned int caseVAO, caseVBO, caseEBO, caseIndexCount;
    unsigned int cpuVAO, cpuVBO, cpuEBO, cpuIndexCount;
    unsigned int gpuVAO, gpuVBO, gpuEBO, gpuIndexCount;
    unsigned int ramVAO, ramVBO, ramEBO, ramIndexCount;
    unsigned int motherboardVAO, motherboardVBO, motherboardEBO, motherboardIndexCount;
    unsigned int psuVAO, psuVBO, psuEBO, psuIndexCount;
    unsigned int ioShieldVAO, ioShieldVBO, ioShieldEBO, ioShieldIndexCount;
    unsigned int shieldVAO, shieldVBO, shieldEBO, shieldIndexCount;
    unsigned int glassVAO, glassVBO, glassEBO, glassIndexCount;
    unsigned int topFanVAO, topFanVBO, topFanEBO, topFanIndexCount;
    unsigned int frontFanVAO, frontFanVBO, frontFanEBO, frontFanIndexCount;
    unsigned int backFanVAO, backFanVBO, backFanEBO, backFanIndexCount;
    unsigned int cpuFanVAO, cpuFanVBO, cpuFanEBO, cpuFanIndexCount;
    unsigned int* modelInfos[52] = {
        &caseVAO, &caseVBO, &caseEBO, &caseIndexCount,
        &cpuVAO, &cpuVBO, &cpuEBO, &cpuIndexCount,
        &gpuVAO, &gpuVBO, &gpuEBO, &gpuIndexCount,
        &ramVAO, &ramVBO, &ramEBO, &ramIndexCount,
        &motherboardVAO, &motherboardVBO, &motherboardEBO, &motherboardIndexCount,
        &psuVAO, &psuVBO, &psuEBO, &psuIndexCount,
        &ioShieldVAO, &ioShieldVBO, &ioShieldEBO, &ioShieldIndexCount,
        &shieldVAO, &shieldVBO, &shieldEBO, &shieldIndexCount,
        &glassVAO, &glassVBO, &glassEBO, &glassIndexCount,
        &topFanVAO, &topFanVBO, &topFanEBO, &topFanIndexCount,
        &frontFanVAO, &frontFanVBO, &frontFanEBO, &frontFanIndexCount,
        &backFanVAO, &backFanVBO, &backFanEBO, &backFanIndexCount,
        &cpuFanVAO, &cpuFanVBO, &cpuFanEBO, &cpuFanIndexCount
    };
    std::thread caseThread([&](){ loadModelData(caseSource, models[0]); });
    std::thread cpuThread([&](){ loadModelData(cpuSource, models[1]); });
    std::thread gpuThread([&](){ loadModelData(gpuSource, models[2]); });
    std::thread ramThread([&](){ loadModelData(ramSource, models[3]); });
    std::thread motherboardThread([&](){ loadModelData(motherboardSource, models[4]); });
    std::thread psuThread([&](){ loadModelData(psuSource, models[5]); });
    std::thread ioShieldThread([&](){ loadModelData(ioShieldSource, models[6]); });
    std::thread shieldThread([&](){ loadModelData(shieldSource, models[7]); });
    std::thread glassThread([&](){ loadModelData(glassSource, models[8]); });
    std::thread topFanThread([&](){ loadModelData(topFanSource, models[9]); });
    std::thread frontFanThread([&](){ loadModelData(frontFanSource, models[10]); });
    std::thread backFanThread([&](){ loadModelData(backFanSource, models[11]); });
    std::thread cpuFanThread([&](){ loadModelData(cpuFanSource, models[12]); });
    caseThread.join();
    cpuThread.join();
    gpuThread.join();
    ramThread.join();
    motherboardThread.join();
    psuThread.join();
    ioShieldThread.join();
    shieldThread.join();
    glassThread.join();
    topFanThread.join();
    frontFanThread.join();
    backFanThread.join();
    cpuFanThread.join();
    for(int i=0; i<13; i++){
        if(!models[i].loaded){
            std::cerr<<"Model data for model "<<i<<" not loaded."<<std::endl;
            return -1;
        }
        createGLObjects(models[i], *modelInfos[i*4], *modelInfos[i*4+1], *modelInfos[i*4+2], *modelInfos[i*4+3]);
    }

    unsigned int caseTextures[4];
    for(int i = 0; i < 4; i++) caseTextures[i] = loadTexture(caseTexturesSource[i]);
    unsigned int cpuTextures[4];
    for(int i = 0; i < 4; i++) cpuTextures[i] = loadTexture(cpuTexturesSource[i]);
    unsigned int gpuTextures[4];
    for(int i = 0; i < 4; i++) gpuTextures[i] = loadTexture(gpuTexturesSource[i]);
    unsigned int ramTextures[4];
    for(int i = 0; i < 4; i++) ramTextures[i] = loadTexture(ramTexturesSource[i]);
    unsigned int motherboardTextures[4];
    for(int i = 0; i < 4; i++) motherboardTextures[i] = loadTexture(motherboardTexturesSource[i]);
    unsigned int psuTextures[4];
    for(int i = 0; i < 4; i++) psuTextures[i] = loadTexture(psuTexturesSource[i]);
    unsigned int ioShieldTextures[4];
    for(int i = 0; i < 4; i++) ioShieldTextures[i] = loadTexture(ioShieldTexturesSource[i]);
    unsigned int shieldTextures[4];
    for(int i = 0; i < 4; i++) shieldTextures[i] = loadTexture(shieldTexturesSource[i]);
    unsigned int glassTextures[4];
    for(int i = 0; i < 4; i++) glassTextures[i] = loadTexture(glassTexturesSource[i]);

    float spriteVertices[] = {
        0.0f, 1.0f,  0.0f, 1.0f,
        1.0f, 0.0f,  1.0f, 0.0f,
        0.0f, 0.0f,  0.0f, 0.0f,
        0.0f, 1.0f,  0.0f, 1.0f,
        1.0f, 1.0f,  1.0f, 1.0f,
        1.0f, 0.0f,  1.0f, 0.0f
    };
    unsigned int spriteVAO, spriteVBO;
    glGenVertexArrays(1, &spriteVAO);
    glGenBuffers(1, &spriteVBO);
    glBindBuffer(GL_ARRAY_BUFFER, spriteVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(spriteVertices), spriteVertices, GL_STATIC_DRAW);
    glBindVertexArray(spriteVAO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*) 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    VAOs.push_back(spriteVAO);
    buffers.push_back(spriteVBO);
    unsigned int textVAO, textVBO;
    glGenVertexArrays(1, &textVAO);
    glGenBuffers(1, &textVBO);
    glBindVertexArray(textVAO);
    glBindBuffer(GL_ARRAY_BUFFER, textVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*) 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    VAOs.push_back(textVAO);
    buffers.push_back(textVBO);
    prepareCharacters();

    char* uiVertexShaderSource = getShaders(uiVertexPath);
    char* uiFragmentShaderSource = getShaders(uiFragmentPath);
    char* textFragmentShaderSource = getShaders(textFragmentPath);
    unsigned int textProgram = createShader(uiVertexShaderSource, textFragmentShaderSource);
    unsigned int uiProgram = createShader(uiVertexShaderSource, uiFragmentShaderSource);

    unsigned int checkboxCheckedTexture = loadTexture(checkboxCheckedSource);
    unsigned int checkboxUncheckedTexture = loadTexture(checkboxUncheckedSource);
    unsigned int sliderKnobTexture = loadTexture(sliderKnobSource);
    unsigned int arrowUpTexture = loadTexture(arrowUpSource);
    unsigned int arrowDownTexture = loadTexture(arrowDownSource);
    unsigned int windowTexture = loadTexture(uiWindowSource);
    unsigned int uiCloseTexture = loadTexture(uiCloseSource);
    unsigned int uiOpenTexture = loadTexture(uiOpenSource);

    unsigned int volume3DTexture;
    glGenTextures(1, &volume3DTexture);
    glBindTexture(GL_TEXTURE_3D, volume3DTexture);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32F, gridSizeX, gridSizeY, gridSizeZ);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    float borderColor[] = {0.0f, 0.0f, 0.0f, 1.0f};
    glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, borderColor);
    unsigned int temperature3DTexture;
    glGenTextures(1, &temperature3DTexture);
    glBindTexture(GL_TEXTURE_3D, temperature3DTexture);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32F, gridSizeX, gridSizeY, gridSizeZ);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glBindTexture(GL_TEXTURE_3D, 0);
    unsigned int volumeVAO, volumeVBO;
    float cubeVertex[] = {
        renderMinX, renderMinY, renderMinZ,
        renderMaxX, renderMinY, renderMinZ,
        renderMaxX, renderMaxY, renderMinZ,
        renderMinX, renderMaxY, renderMinZ,
        renderMinX, renderMinY, renderMaxZ,
        renderMaxX, renderMinY, renderMaxZ,
        renderMaxX, renderMaxY, renderMaxZ,
        renderMinX, renderMaxY, renderMaxZ
    };
    unsigned int cubeIndices[] = {
        0, 1, 2,  2, 3, 0,
        4, 5, 6,  6, 7, 4,
        0, 4, 7,  7, 3, 0,
        1, 5, 6,  6, 2, 1,
        0, 1, 5,  5, 4, 0,
        3, 2, 6,  6, 7, 3
    };
    glGenVertexArrays(1, &volumeVAO);
    glBindVertexArray(volumeVAO);
    unsigned int posVBO = 0, idxEBO = 0;
    glGenBuffers(1, &posVBO);
    glBindBuffer(GL_ARRAY_BUFFER, posVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertex), cubeVertex, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*) 0);
    glGenBuffers(1, &idxEBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, idxEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cubeIndices), cubeIndices, GL_STATIC_DRAW);
    glBindVertexArray(0);
    char* vertexVolumeShaderSource = getShaders(volumeVertexShaderPath);
    char* fragmentVolumeShaderSource = getShaders(volumeFragmentShaderPath);
    unsigned int volumeShaderProgram = createShader(vertexVolumeShaderSource, fragmentVolumeShaderSource);
    VAOs.push_back(volumeVAO);
    buffers.push_back(volumeVBO);
    buffers.push_back(idxEBO);
    buffers.push_back(posVBO);

    checkboxItems[0] = &gpuEnabled;
    checkboxItems[1] = &topFanEnabled;
    checkboxItems[2] = &cpuFanEnabled;
    checkboxItems[3] = &frontFanEnabled;

    for(int i=0; i<3; i++) sliderXValues[i] = &backFanLocations[i];

    itemChangedPtr = &itemChanged;
    runningPtr = &running;
    displayingPressure = &displayPressure;

    char* vertexShaderSource = getShaders(vertexShaderPath);
    char* fragmentShaderSource = getShaders(fragmentShaderPath);
    unsigned int shaderProgram = createShader(vertexShaderSource, fragmentShaderSource);

    signalItemsReady();
    waitForVelocityField();

    float dt = 1.0f / 60.0f; // 60 FPS limit
    while(!glfwWindowShouldClose(window)){
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        processInput(window);
        if(deltaTime < dt) continue;
        lastFrame = currentFrame;

        glClearColor(0.1f, 0.1f, 0.25f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDisable(GL_CULL_FACE);
        glm::mat4 projection = glm::perspective(glm::radians(camFOV), (float) SCR_WIDTH / (float) SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = glm::lookAt(camPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 model = glm::mat4(1.0f);

        glUseProgram(shaderProgram);

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &model[0][0]);
        glUniform3fv(glGetUniformLocation(shaderProgram, "camPos"), 1, &camPos[0]);
        glUniform1i(glGetUniformLocation(shaderProgram, "isEmissive"), 0);

        glDepthMask(GL_TRUE);

        if(gpuEnabled) drawObject(gpuTextures, shaderProgram, gpuVAO, gpuIndexCount);
        drawObject(cpuTextures, shaderProgram, cpuVAO, cpuIndexCount);
        drawObject(ramTextures, shaderProgram, ramVAO, ramIndexCount);
        drawObject(motherboardTextures, shaderProgram, motherboardVAO, motherboardIndexCount);
        drawObject(psuTextures, shaderProgram, psuVAO, psuIndexCount);
        drawObject(ioShieldTextures, shaderProgram, ioShieldVAO, ioShieldIndexCount);
        if(cpuFanEnabled) drawObject(caseTextures, shaderProgram, cpuFanVAO, cpuFanIndexCount);
        if(topFanEnabled) drawObject(caseTextures, shaderProgram, topFanVAO, topFanIndexCount);
        if(frontFanEnabled) drawObject(caseTextures, shaderProgram, frontFanVAO, frontFanIndexCount);

        for(int i=0; i<3; i++){
            if(backFanLocations[i]>0.0f) continue;
            glm::mat4 backFanLocation = glm::mat4(1.0f);
            backFanLocation = glm::translate(backFanLocation, glm::vec3(0, backFanLocations[i], 0));
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &backFanLocation[0][0]);
            drawObject(caseTextures, shaderProgram, backFanVAO, backFanIndexCount);
        }
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &model[0][0]);
        drawObject(caseTextures, shaderProgram, caseVAO, caseIndexCount);
        glDepthMask(GL_FALSE);
        glEnable(GL_DEPTH_TEST);
        drawObject(shieldTextures, shaderProgram, shieldVAO, shieldIndexCount);
        glDepthMask(GL_TRUE);

        static std::vector<float> volumeData(gridSizeX * gridSizeY * gridSizeZ);
        for(int z=0; z<gridSizeZ; z++){
            for(int y=0; y<gridSizeY; y++){
                for(int x=0; x<gridSizeX; x++){
                    int index = z * gridSizeX * gridSizeY + y * gridSizeX + x;
                    if(displayPressure) volumeData[index] = pressureField[index];
                    else{
                        float vx = velocityField[index * 3 + 0];
                        float vy = velocityField[index * 3 + 1];
                        float vz = velocityField[index * 3 + 2];
                        volumeData[index] = sqrt(vx * vx + vy * vy + vz * vz);
                    }
                }
            }
        }
        glBindTexture(GL_TEXTURE_3D, volume3DTexture);
        glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, gridSizeX, gridSizeY, gridSizeZ, GL_RED, GL_FLOAT, volumeData.data());
        static std::vector<float> temperatureData(gridSizeX * gridSizeY * gridSizeZ);
        for(int z=0; z<gridSizeZ; z++){
            for(int y=0; y<gridSizeY; y++){
                for(int x=0; x<gridSizeX; x++){
                    int index = z * gridSizeX * gridSizeY + y * gridSizeX + x;
                    temperatureData[index] = temperatureField[index];
                }
            }
        }
        glBindTexture(GL_TEXTURE_3D, temperature3DTexture);
        glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, gridSizeX, gridSizeY, gridSizeZ, GL_RED, GL_FLOAT, temperatureData.data());
        glBindTexture(GL_TEXTURE_3D, 0);

        glUseProgram(volumeShaderProgram);

        glUniformMatrix4fv(glGetUniformLocation(volumeShaderProgram, "projection"), 1, GL_FALSE, &projection[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(volumeShaderProgram, "view"), 1, GL_FALSE, &view[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(volumeShaderProgram, "model"), 1, GL_FALSE, &model[0][0]);
        glUniform3fv(glGetUniformLocation(volumeShaderProgram, "camPos"), 1, &camPos[0]);
        glUniform1i(glGetUniformLocation(volumeShaderProgram, "volumeTex"), 0);
        glUniform1i(glGetUniformLocation(volumeShaderProgram, "temperatureTex"), 1);
        glUniform1i(glGetUniformLocation(volumeShaderProgram, "displayPressure"), displayPressure ? 1 : 0);
        glUniform3f(glGetUniformLocation(volumeShaderProgram, "gridSize"), (float) gridSizeX, (float) gridSizeY, (float) gridSizeZ);
        glUniform1f(glGetUniformLocation(volumeShaderProgram, "stepSize"), 1.0 / 128.0f);

        glUniform3f(glGetUniformLocation(volumeShaderProgram, "worldMin"), worldMinX, worldMinY, worldMinZ);
        glUniform3f(glGetUniformLocation(volumeShaderProgram, "worldMax"), worldMaxX, worldMaxY, worldMaxZ);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, volume3DTexture);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_3D, temperature3DTexture);
        glBindVertexArray(volumeVAO);
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        glUseProgram(shaderProgram);

        drawObject(shieldTextures, shaderProgram, shieldVAO, shieldIndexCount);
        drawObject(glassTextures, shaderProgram, glassVAO, glassIndexCount);

        if(showUI){
            drawSprite(uiProgram, spriteVAO, spriteVBO, glm::vec2(SCR_WIDTH-245.0f, SCR_HEIGHT - 50.0f), glm::vec2(20.0f), glm::vec3(1.0f), uiCloseTexture);
            drawText(textProgram, textVAO, textVBO, "Menu", glm::vec2(SCR_WIDTH - 200.0f, SCR_HEIGHT - 70.0f), 0.4f, glm::vec3(0.0f));
            drawText(textProgram, textVAO, textVBO, "GPU Enabled", glm::vec2(SCR_WIDTH - 200.0f, SCR_HEIGHT - 100.0f), 0.3f, glm::vec3(0.0f));
            drawCheckbox(uiProgram, spriteVAO, spriteVBO, glm::vec2(SCR_WIDTH - 75.0f, SCR_HEIGHT - checkboxYPositions[0]), glm::vec2(20.0f), gpuEnabled, checkboxCheckedTexture, checkboxUncheckedTexture);
            drawText(textProgram, textVAO, textVBO, gpuEnabled ? "ON" : "OFF", glm::vec2(SCR_WIDTH - 50.0f, SCR_HEIGHT - 100.0f), 0.3f, glm::vec3(0.0f));
            drawText(textProgram, textVAO, textVBO, "Top Fan Enabled", glm::vec2(SCR_WIDTH - 200.0f, SCR_HEIGHT - 130.0f), 0.3f, glm::vec3(0.0f));
            drawCheckbox(uiProgram, spriteVAO, spriteVBO, glm::vec2(SCR_WIDTH - 75.0f, SCR_HEIGHT - checkboxYPositions[1]), glm::vec2(20.0f), topFanEnabled, checkboxCheckedTexture, checkboxUncheckedTexture);
            drawText(textProgram, textVAO, textVBO, topFanEnabled ? "ON" : "OFF", glm::vec2(SCR_WIDTH - 50.0f, SCR_HEIGHT - 130.0f), 0.3f, glm::vec3(0.0f));
            drawText(textProgram, textVAO, textVBO, "CPU Fan Enabled", glm::vec2(SCR_WIDTH - 200.0f, SCR_HEIGHT - 160.0f), 0.3f, glm::vec3(0.0f));
            drawCheckbox(uiProgram, spriteVAO, spriteVBO, glm::vec2(SCR_WIDTH - 75.0f, SCR_HEIGHT - checkboxYPositions[2]), glm::vec2(20.0f), cpuFanEnabled, checkboxCheckedTexture, checkboxUncheckedTexture);
            drawText(textProgram, textVAO, textVBO, cpuFanEnabled ? "ON" : "OFF", glm::vec2(SCR_WIDTH - 50.0f, SCR_HEIGHT - 160.0f), 0.3f, glm::vec3(0.0f));
            drawText(textProgram, textVAO, textVBO, "Front Fan Enabled", glm::vec2(SCR_WIDTH - 200.0f, SCR_HEIGHT - 190.0f), 0.3f, glm::vec3(0.0f));
            drawCheckbox(uiProgram, spriteVAO, spriteVBO, glm::vec2(SCR_WIDTH - 75.0f, SCR_HEIGHT - checkboxYPositions[3]), glm::vec2(20.0f), frontFanEnabled, checkboxCheckedTexture, checkboxUncheckedTexture);
            drawText(textProgram, textVAO, textVBO, frontFanEnabled ? "ON" : "OFF", glm::vec2(SCR_WIDTH - 50.0f, SCR_HEIGHT - 190.0f), 0.3f, glm::vec3(0.0f));
            char numberOfFans = '0';
            for(int i=0; i<3; i++) if(backFanLocations[i]<=0.0f) numberOfFans++;
            const char* backFanTextPrefix = "Back Fans: ";
            char backFanText[13];
            strcpy(backFanText, backFanTextPrefix);
            backFanText[11] = numberOfFans;
            drawText(textProgram, textVAO, textVBO, backFanText, glm::vec2(SCR_WIDTH - 200.0f, SCR_HEIGHT - 220.0f), 0.3f, glm::vec3(0.0f));
            drawArrowInput(uiProgram, spriteVAO, spriteVBO, glm::vec2(SCR_WIDTH - 100.0f, SCR_HEIGHT - 220.0f), glm::vec2(20.0f), arrowUpTexture, arrowDownTexture);
            int lastSliderY = 250;
            int existingSliders = 0;
            for(int i=0; i<3; i++){
                if(backFanLocations[i]<=0.0f) {
                    char fanNumber = '0';
                    fanNumber += i + 1;
                    const char* fanTextPrefix = "Fan  :";
                    char fanText[7];
                    strcpy(fanText, fanTextPrefix);
                    fanText[4] = fanNumber;
                    drawText(textProgram, textVAO, textVBO, fanText, glm::vec2(SCR_WIDTH - 200.0f, SCR_HEIGHT - lastSliderY), 0.3f, glm::vec3(0.0f));
                    drawSlider(uiProgram, spriteVAO, spriteVBO, glm::vec2(SCR_WIDTH - 140.0f, SCR_HEIGHT - lastSliderY), glm::vec2(80.0f, 20.0f), backFanLocations[i], glassTextures[2], sliderKnobTexture);
                    sliderYPositions[existingSliders++] = lastSliderY;
                    lastSliderY += 30;
                }
                else sliderYPositions[i] = -1;
            }
            drawText(textProgram, textVAO, textVBO, "Show Pressure", glm::vec2(SCR_WIDTH - 200.0f, SCR_HEIGHT - 450.0f), 0.3f, glm::vec3(0.0f));
            drawCheckbox(uiProgram, spriteVAO, spriteVBO, glm::vec2(SCR_WIDTH - 75.0f, SCR_HEIGHT - 450.0f), glm::vec2(20.0f), displayPressure, checkboxCheckedTexture, checkboxUncheckedTexture);
            drawText(textProgram, textVAO, textVBO, displayPressure ? "ON" : "OFF", glm::vec2(SCR_WIDTH - 50.0f, SCR_HEIGHT - 450.0f), 0.3f, glm::vec3(0.0f));
            drawSprite(uiProgram, spriteVAO, spriteVBO, glm::vec2(SCR_WIDTH - 220.0f, SCR_HEIGHT - 500.0f), glm::vec2(200.0f, 450.0f), glm::vec3(1.0f), windowTexture);
        }
        else drawSprite(uiProgram, spriteVAO, spriteVBO, glm::vec2(SCR_WIDTH - 25.0f, SCR_HEIGHT - 50.0f), glm::vec2(20.0f), glm::vec3(1.0f), uiOpenTexture);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    for(unsigned int VAO : VAOs) glDeleteVertexArrays(1, &VAO);
    for(unsigned int buffer : buffers) glDeleteBuffers(1, &buffer);

    glfwTerminate();
    return 0;
}