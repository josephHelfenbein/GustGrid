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
#define vertexShaderPath "./src/shaders/main.vert"
#define fragmentShaderPath "./src/shaders/main.frag"
#define PI 3.14159265358979323846f

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
                &&  xposFloat > SCR_WIDTH - 100.0f
                &&  xposFloat < SCR_WIDTH - 80.0f) {
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
            &&  xposFloat > SCR_WIDTH - 220.0f
            &&  xposFloat < SCR_WIDTH - 200.0f) hoverElement = 6;
        } else if(
            yposFloat > 30.0f
        &&  yposFloat < 50.0f
        &&  xposFloat > SCR_WIDTH - 25.0f
        &&  xposFloat < SCR_WIDTH - 5.0f) hoverElement = 6;

        return;
    }
    else if(showUI){
        for(int i=0; i<3; i++){
            float minValue = i==0 ? 0.0f : *sliderXValues[i-1] - 2.5f;
            float maxValue = i==2 ? -5.0f : *sliderXValues[i+1] + 2.5f;
            if(
                sliderYPositions[i] != -1
            &&  yposFloat < sliderYPositions[i]
            &&  yposFloat > sliderYPositions[i] - 20.0f
            &&  xposFloat > SCR_WIDTH - 140.0f
            &&  xposFloat < SCR_WIDTH - 60.0f) {
                possibleSliderXValues[i] = 5 * (-xposFloat + (SCR_WIDTH - 140.0f)) / 80.0f;
                if(possibleSliderXValues[i] > minValue) possibleSliderXValues[i] = minValue;
                else if(possibleSliderXValues[i] < maxValue) possibleSliderXValues[i] = maxValue;
                *sliderXValues[i] = possibleSliderXValues[i];
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
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);
    else if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS){
        if(hoverElement != -1 && hoverElement < 4) *checkboxItems[hoverElement] = !(*checkboxItems[hoverElement]);
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
        else if(hoverElement == 6) showUI = !showUI;
    }
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
void loadModel(const char* file, unsigned int &VAO, unsigned int &VBO, unsigned int &EBO, unsigned int &indexCount){
    VAOs.push_back(VAO);
    buffers.push_back(VBO);
    buffers.push_back(EBO);
    objl::Loader loader;
    if(!loader.LoadFile(file)){
        std::cerr<<"Failed to load OBJ file"<<std::endl;
        return;
    }
    objl::Mesh mesh = loader.LoadedMeshes[0];
    float vertices[mesh.Vertices.size() * 8];
    int i = 0;
    for(const auto &vertex : mesh.Vertices){
        vertices[i++] = vertex.Position.X;
        vertices[i++] = vertex.Position.Y;
        vertices[i++] = vertex.Position.Z;
        vertices[i++] = vertex.TextureCoordinate.X;
        vertices[i++] = 1.0f - vertex.TextureCoordinate.Y;
        vertices[i++] = vertex.Normal.X;
        vertices[i++] = vertex.Normal.Y;
        vertices[i++] = vertex.Normal.Z;
    }
    indexCount = mesh.Indices.size();
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, mesh.Vertices.size() * 8 * sizeof(float), &vertices[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexCount * sizeof(unsigned int), &mesh.Indices[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*) 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*) (3 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*) (5 * sizeof(float)));
    return;
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
    arrowModel = glm::translate(arrowModel, glm::vec3(position.x, position.y, 0.0f));
    arrowModel = glm::scale(arrowModel, glm::vec3(size.x, size.y, 1.0f));
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, &arrowModel[0][0]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, arrowUpTexture);
    glUniform1i(glGetUniformLocation(shader, "image"), 0);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    arrowModel = glm::translate(arrowModel, glm::vec3(0.0f, -size.y * 2.0f, 0.0f));
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, &arrowModel[0][0]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, arrowDownTexture);
    glUniform1i(glGetUniformLocation(shader, "image"), 1);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}
int startRenderer(bool& gpuEnabled, bool& topFanEnabled, bool& cpuFanEnabled, bool& frontFanEnabled, float* backFanLocations){
    if(!glfwInit()){
        std::cerr<<"Failed to initialize GLFW"<<std::endl;
        return -1;
    }
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
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

    unsigned int caseVAO, caseVBO, caseEBO, caseIndexCount;
    loadModel(caseSource, caseVAO, caseVBO, caseEBO, caseIndexCount);
    unsigned int caseTextures[4];
    for(int i = 0; i < 4; i++) caseTextures[i] = loadTexture(caseTexturesSource[i]);

    unsigned int cpuVAO, cpuVBO, cpuEBO, cpuIndexCount;
    loadModel(cpuSource, cpuVAO, cpuVBO, cpuEBO, cpuIndexCount);
    unsigned int cpuTextures[4];
    for(int i = 0; i < 4; i++) cpuTextures[i] = loadTexture(cpuTexturesSource[i]);

    unsigned int gpuVAO, gpuVBO, gpuEBO, gpuIndexCount;
    loadModel(gpuSource, gpuVAO, gpuVBO, gpuEBO, gpuIndexCount);
    unsigned int gpuTextures[4];
    for(int i = 0; i < 4; i++) gpuTextures[i] = loadTexture(gpuTexturesSource[i]);

    unsigned int ramVAO, ramVBO, ramEBO, ramIndexCount;
    loadModel(ramSource, ramVAO, ramVBO, ramEBO, ramIndexCount);
    unsigned int ramTextures[4];
    for(int i = 0; i < 4; i++) ramTextures[i] = loadTexture(ramTexturesSource[i]);

    unsigned int motherboardVAO, motherboardVBO, motherboardEBO, motherboardIndexCount;
    loadModel(motherboardSource, motherboardVAO, motherboardVBO, motherboardEBO, motherboardIndexCount);
    unsigned int motherboardTextures[4];
    for(int i = 0; i < 4; i++) motherboardTextures[i] = loadTexture(motherboardTexturesSource[i]);

    unsigned int psuVAO, psuVBO, psuEBO, psuIndexCount;
    loadModel(psuSource, psuVAO, psuVBO, psuEBO, psuIndexCount);
    unsigned int psuTextures[4];
    for(int i = 0; i < 4; i++) psuTextures[i] = loadTexture(psuTexturesSource[i]);

    unsigned int ioShieldVAO, ioShieldVBO, ioShieldEBO, ioShieldIndexCount;
    loadModel(ioShieldSource, ioShieldVAO, ioShieldVBO, ioShieldEBO, ioShieldIndexCount);
    unsigned int ioShieldTextures[4];
    for(int i = 0; i < 4; i++) ioShieldTextures[i] = loadTexture(ioShieldTexturesSource[i]);

    unsigned int shieldVAO, shieldVBO, shieldEBO, shieldIndexCount;
    loadModel(shieldSource, shieldVAO, shieldVBO, shieldEBO, shieldIndexCount);
    unsigned int shieldTextures[4];
    for(int i = 0; i < 4; i++) shieldTextures[i] = loadTexture(shieldTexturesSource[i]);

    unsigned int glassVAO, glassVBO, glassEBO, glassIndexCount;
    loadModel(glassSource, glassVAO, glassVBO, glassEBO, glassIndexCount);
    unsigned int glassTextures[4];
    for(int i = 0; i < 4; i++) glassTextures[i] = loadTexture(glassTexturesSource[i]);

    unsigned int topFanVAO, topFanVBO, topFanEBO, topFanIndexCount;
    loadModel(topFanSource, topFanVAO, topFanVBO, topFanEBO, topFanIndexCount);
    unsigned int frontFanVAO, frontFanVBO, frontFanEBO, frontFanIndexCount;
    loadModel(frontFanSource, frontFanVAO, frontFanVBO, frontFanEBO, frontFanIndexCount);
    unsigned int backFanVAO, backFanVBO, backFanEBO, backFanIndexCount;
    loadModel(backFanSource, backFanVAO, backFanVBO, backFanEBO, backFanIndexCount);
    unsigned int cpuFanVAO, cpuFanVBO, cpuFanEBO, cpuFanIndexCount;
    loadModel(cpuFanSource, cpuFanVAO, cpuFanVBO, cpuFanEBO, cpuFanIndexCount);

    float spriteVertices[] = {
        0.0f, 1.0f,  0.0f, 1.0f,
        1.0f, 0.0f,  1.0f, 0.0f,
        0.0f, 0.0f,  0.0f, 0.0f,
        0.0f, 1.0f,  0.0f, 1.0f,
        1.0f, 1.0f,  1.0f, 1.0f,
        1.0f, 0.0f,  1.0f, 0.0f
    };
    unsigned int spriteVAO, spriteVBO;
    VAOs.push_back(spriteVAO);
    buffers.push_back(spriteVBO);
    glGenVertexArrays(1, &spriteVAO);
    glGenBuffers(1, &spriteVBO);
    glBindBuffer(GL_ARRAY_BUFFER, spriteVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(spriteVertices), spriteVertices, GL_STATIC_DRAW);
    glBindVertexArray(spriteVAO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*) 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    unsigned int textVAO, textVBO;
    VAOs.push_back(textVAO);
    buffers.push_back(textVBO);
    glGenVertexArrays(1, &textVAO);
    glGenBuffers(1, &textVBO);
    glBindVertexArray(textVAO);
    glBindBuffer(GL_ARRAY_BUFFER, textVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*) 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
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

    checkboxItems[0] = &gpuEnabled;
    checkboxItems[1] = &topFanEnabled;
    checkboxItems[2] = &cpuFanEnabled;
    checkboxItems[3] = &frontFanEnabled;

    for(int i=0; i<3; i++) sliderXValues[i] = &backFanLocations[i];

    char* vertexShaderSource = getShaders(vertexShaderPath);
    char* fragmentShaderSource = getShaders(fragmentShaderPath);
    unsigned int shaderProgram = createShader(vertexShaderSource, fragmentShaderSource);
    
    while(!glfwWindowShouldClose(window)){
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        processInput(window);

        glClearColor(0.1f, 0.1f, 0.25f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        glUseProgram(shaderProgram);
        glEnable(GL_BLEND);
        glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

        glm::mat4 projection = glm::perspective(glm::radians(camFOV), (float) SCR_WIDTH / (float) SCR_HEIGHT, 0.1f, 100.0f);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection[0][0]);

        glm::mat4 view = glm::lookAt(camPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view[0][0]);

        glUniform3fv(glGetUniformLocation(shaderProgram, "camPos"), 1, &camPos[0]);

        glm::mat4 model = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &model[0][0]);

        glUniform1i(glGetUniformLocation(shaderProgram, "isEmissive"), 0);
        
        if(gpuEnabled) drawObject(gpuTextures, shaderProgram, gpuVAO, gpuIndexCount);
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

        if(cpuFanEnabled) drawObject(caseTextures, shaderProgram, cpuFanVAO, cpuFanIndexCount);

        drawObject(cpuTextures, shaderProgram, cpuVAO, cpuIndexCount);
        drawObject(ramTextures, shaderProgram, ramVAO, ramIndexCount);
        drawObject(motherboardTextures, shaderProgram, motherboardVAO, motherboardIndexCount);
        drawObject(psuTextures, shaderProgram, psuVAO, psuIndexCount);
        drawObject(ioShieldTextures, shaderProgram, ioShieldVAO, ioShieldIndexCount);
        drawObject(caseTextures, shaderProgram, caseVAO, caseIndexCount);
        drawObject(shieldTextures, shaderProgram, shieldVAO, shieldIndexCount);
        drawObject(glassTextures, shaderProgram, glassVAO, glassIndexCount);

        if(showUI){
            drawSprite(uiProgram, spriteVAO, spriteVBO, glm::vec2(SCR_WIDTH-245.0f, SCR_HEIGHT - 50.0f), glm::vec2(20.0f), glm::vec3(1.0f), uiCloseTexture);
            drawText(textProgram, textVAO, textVBO, "Menu", glm::vec2(SCR_WIDTH - 200.0f, SCR_HEIGHT - 70.0f), 0.3f, glm::vec3(0.0f));
            drawText(textProgram, textVAO, textVBO, "GPU Enabled", glm::vec2(SCR_WIDTH - 200.0f, SCR_HEIGHT - 100.0f), 0.3f, glm::vec3(0.0f));
            drawCheckbox(uiProgram, spriteVAO, spriteVBO, glm::vec2(SCR_WIDTH - 100.0f, SCR_HEIGHT - checkboxYPositions[0]), glm::vec2(20.0f), gpuEnabled, checkboxCheckedTexture, checkboxUncheckedTexture);
            drawText(textProgram, textVAO, textVBO, gpuEnabled ? "ON" : "OFF", glm::vec2(SCR_WIDTH - 75.0f, SCR_HEIGHT - 100.0f), 0.3f, glm::vec3(0.0f));
            drawText(textProgram, textVAO, textVBO, "Top Fan Enabled", glm::vec2(SCR_WIDTH - 200.0f, SCR_HEIGHT - 130.0f), 0.3f, glm::vec3(0.0f));
            drawCheckbox(uiProgram, spriteVAO, spriteVBO, glm::vec2(SCR_WIDTH - 100.0f, SCR_HEIGHT - checkboxYPositions[1]), glm::vec2(20.0f), topFanEnabled, checkboxCheckedTexture, checkboxUncheckedTexture);
            drawText(textProgram, textVAO, textVBO, topFanEnabled ? "ON" : "OFF", glm::vec2(SCR_WIDTH - 75.0f, SCR_HEIGHT - 130.0f), 0.3f, glm::vec3(0.0f));
            drawText(textProgram, textVAO, textVBO, "CPU Fan Enabled", glm::vec2(SCR_WIDTH - 200.0f, SCR_HEIGHT - 160.0f), 0.3f, glm::vec3(0.0f));
            drawCheckbox(uiProgram, spriteVAO, spriteVBO, glm::vec2(SCR_WIDTH - 100.0f, SCR_HEIGHT - checkboxYPositions[2]), glm::vec2(20.0f), cpuFanEnabled, checkboxCheckedTexture, checkboxUncheckedTexture);
            drawText(textProgram, textVAO, textVBO, cpuFanEnabled ? "ON" : "OFF", glm::vec2(SCR_WIDTH - 75.0f, SCR_HEIGHT - 160.0f), 0.3f, glm::vec3(0.0f));
            drawText(textProgram, textVAO, textVBO, "Front Fan Enabled", glm::vec2(SCR_WIDTH - 200.0f, SCR_HEIGHT - 190.0f), 0.3f, glm::vec3(0.0f));
            drawCheckbox(uiProgram, spriteVAO, spriteVBO, glm::vec2(SCR_WIDTH - 100.0f, SCR_HEIGHT - checkboxYPositions[3]), glm::vec2(20.0f), frontFanEnabled, checkboxCheckedTexture, checkboxUncheckedTexture);
            drawText(textProgram, textVAO, textVBO, frontFanEnabled ? "ON" : "OFF", glm::vec2(SCR_WIDTH - 75.0f, SCR_HEIGHT - 190.0f), 0.3f, glm::vec3(0.0f));
            char numberOfFans = '0';
            for(int i=0; i<3; i++) if(backFanLocations[i]<=0.0f) numberOfFans++;
            const char backFanText[] = {'B', 'a', 'c', 'k', ' ', 'F', 'a', 'n', 's', ':', ' ', numberOfFans};
            drawText(textProgram, textVAO, textVBO, backFanText, glm::vec2(SCR_WIDTH - 200.0f, SCR_HEIGHT - 220.0f), 0.3f, glm::vec3(0.0f));
            drawArrowInput(uiProgram, spriteVAO, spriteVBO, glm::vec2(SCR_WIDTH - 100.0f, SCR_HEIGHT - 220.0f), glm::vec2(20.0f), arrowUpTexture, arrowDownTexture);
            int lastSliderY = 250;
            int existingSliders = 0;
            for(int i=0; i<3; i++){
                if(backFanLocations[i]<=0.0f) {
                    char fanNumber = '0';
                    fanNumber += i + 1;
                    const char fanText[] = {'F', 'a', 'n',  ' ', fanNumber, ':'};
                    drawText(textProgram, textVAO, textVBO, fanText, glm::vec2(SCR_WIDTH - 180.0f, SCR_HEIGHT - lastSliderY), 0.3f, glm::vec3(0.0f));
                    drawSlider(uiProgram, spriteVAO, spriteVBO, glm::vec2(SCR_WIDTH - 140.0f, SCR_HEIGHT - lastSliderY), glm::vec2(80.0f, 20.0f), backFanLocations[i], glassTextures[2], sliderKnobTexture);
                    sliderYPositions[existingSliders++] = lastSliderY;
                    lastSliderY += 30;
                }
                else sliderYPositions[i] = -1;
            }
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