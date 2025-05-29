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

unsigned int SCR_WIDTH = 800;
unsigned int SCR_HEIGHT = 600;
const float PI = 3.14159265358979323846f;
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

std::vector<unsigned int> VAOs;
std::vector<unsigned int> buffers;

const char* uiVertexPath = "./src/shaders/ui.vert";
const char* uiFragmentPath = "./src/shaders/ui.frag";
const char* textFragmentPath = "./src/shaders/text.frag";

const char* caseSource = "./src/models/case.obj";
const char* caseTexturesSource[4] = {
    "./src/textures/case/basecolor.png",
    "./src/textures/case/metallic.png",
    "./src/textures/case/roughness.png",
    "./src/textures/case/normal.png"
};
const char* cpuSource = "./src/models/cpu.obj";
const char* cpuTexturesSource[4] = {
    "./src/textures/cpu/basecolor.png",
    "./src/textures/cpu/metallic.png",
    "./src/textures/cpu/roughness.png",
    "./src/textures/cpu/normal.png"
};
const char* gpuSource = "./src/models/gpu.obj";
const char* gpuTexturesSource[4] = {
    "./src/textures/gpu/basecolor.png",
    "./src/textures/gpu/metallic.png",
    "./src/textures/gpu/roughness.png",
    "./src/textures/gpu/normal.png"
};
const char* ramSource = "./src/models/ram.obj";
const char* ramTexturesSource[4] = {
    "./src/textures/ram/basecolor.png",
    "./src/textures/ram/metallic.png",
    "./src/textures/ram/roughness.png",
    "./src/textures/ram/normal.png"
};
const char* motherboardSource = "./src/models/motherboard.obj";
const char* motherboardTexturesSource[4] = {
    "./src/textures/motherboard/basecolor.png",
    "./src/textures/motherboard/metallic.png",
    "./src/textures/motherboard/roughness.png",
    "./src/textures/motherboard/normal.png"
};
const char* psuSource = "./src/models/psu.obj";
const char* psuTexturesSource[4] = {
    "./src/textures/psu/basecolor.png",
    "./src/textures/psu/metallic.png",
    "./src/textures/psu/roughness.png",
    "./src/textures/psu/normal.png"
};
const char* ioShieldSource = "./src/models/ioshield.obj";
const char* ioShieldTexturesSource[4] = {
    "./src/textures/ioshield/basecolor.png",
    "./src/textures/ioshield/metallic.png",
    "./src/textures/ioshield/roughness.png",
    "./src/textures/ioshield/normal.png"
};
const char* shieldSource = "./src/models/shield.obj";
const char* shieldTexturesSource[4] = {
    "./src/textures/shield/basecolor.png",
    "./src/textures/shield/metallic.png",
    "./src/textures/shield/roughness.png",
    "./src/textures/shield/normal.png"
};
const char* glassSource = "./src/models/glass.obj";
const char* glassTexturesSource[4] = {
    "./src/textures/glass/basecolor.png",
    "./src/textures/glass/metallic.png",
    "./src/textures/glass/roughness.png",
    "./src/textures/glass/normal.png"
};

const char* topFanSource = "./src/models/topfan.obj";
const char* frontFanSource = "./src/models/frontfan.obj";
const char* backFanSource = "./src/models/backfan.obj";
const char* cpuFanSource = "./src/models/cpufan.obj";

const char* zArrowSource = "./src/models/upwardvector.obj";
const char* arrowTexturesSource[6] = {
    "./src/textures/arrow/basecolor-blue.png",
    "./src/textures/arrow/metallic.png",
    "./src/textures/arrow/roughness.png",
    "./src/textures/arrow/normal.png",
    "./src/textures/arrow/basecolor-green.png",
    "./src/textures/arrow/basecolor-red.png",
};
const char* xArrowSource = "./src/models/xvector.obj";
const char* yArrowSource = "./src/models/yvector.obj";

const char* vertexShaderPath = "./src/shaders/main.vert";
const char* fragmentShaderPath = "./src/shaders/main.frag";

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
        return;
    }

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
void RenderText(unsigned int shader, unsigned int VAO, unsigned int VBO, char* text, float x, float y, float scale, glm::vec3 color){
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
        float xPos = x + ch.bearing.x * scale;
        float yPos = y - (ch.size.y - ch.bearing.y) * scale;
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
        x += (ch.advance >> 6) * scale;
    }
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
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

    unsigned int zArrowVAO, zArrowVBO, zArrowEBO, zArrowIndexCount;
    loadModel(zArrowSource, zArrowVAO, zArrowVBO, zArrowEBO, zArrowIndexCount);
    unsigned int arrowTextures[4];
    for(int i = 0; i < 4; i++) arrowTextures[i] = loadTexture(arrowTexturesSource[i]);
    unsigned int xArrowVAO, xArrowVBO, xArrowEBO, xArrowIndexCount;
    loadModel(xArrowSource, xArrowVAO, xArrowVBO, xArrowEBO, xArrowIndexCount);
    unsigned int yArrowVAO, yArrowVBO, yArrowEBO, yArrowIndexCount;
    loadModel(yArrowSource, yArrowVAO, yArrowVBO, yArrowEBO, yArrowIndexCount);
    unsigned int otherArrowColors[2] = {loadTexture(arrowTexturesSource[4]), loadTexture(arrowTexturesSource[5])};

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


    char* vertexShaderSource = getShaders(vertexShaderPath);
    char* fragmentShaderSource = getShaders(fragmentShaderPath);
    unsigned int shaderProgram = createShader(vertexShaderSource, fragmentShaderSource);
    
    while(!glfwWindowShouldClose(window)){
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        glClearColor(0.1f, 0.1f, 0.25f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        glUseProgram(shaderProgram);
        glEnable(GL_BLEND);
        glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

        glm::mat4 projection = glm::perspective(glm::radians(camFOV), (float) SCR_WIDTH / (float) SCR_HEIGHT, 0.1f, 100.0f);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection[0][0]);

        glm::mat4 view = glm::lookAt(camPos, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
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

        glm::mat4 arrowsLocation = glm::mat4(1.0f);
        arrowsLocation = glm::translate(arrowsLocation, glm::vec3(0.0f, 5.0f, 0.0f));
        arrowsLocation = glm::scale(arrowsLocation, glm::vec3(0.5f, 0.5f, 0.5f));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &arrowsLocation[0][0]);
        glUniform1i(glGetUniformLocation(shaderProgram, "isEmissive"), 1);
        drawObject(arrowTextures, shaderProgram, zArrowVAO, zArrowIndexCount);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, otherArrowColors[0]);
        glUniform1i(glGetUniformLocation(shaderProgram, "albedoMap"), 0);
        glBindVertexArray(xArrowVAO);
        glDrawElements(GL_TRIANGLES, xArrowIndexCount, GL_UNSIGNED_INT, 0);
        glBindTexture(GL_TEXTURE_2D, otherArrowColors[1]);
        glUniform1i(glGetUniformLocation(shaderProgram, "albedoMap"), 0);
        glBindVertexArray(yArrowVAO);
        glDrawElements(GL_TRIANGLES, yArrowIndexCount, GL_UNSIGNED_INT, 0);

        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    for(unsigned int VAO : VAOs) glDeleteVertexArrays(1, &VAO);
    for(unsigned int buffer : buffers) glDeleteBuffers(1, &buffer);

    glfwTerminate();
    return 0;
}