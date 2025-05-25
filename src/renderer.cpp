#include <iostream>
#include <glad/glad.h>
#include <glfw/include/GLFW/glfw3.h>
#include <glm/glm/glm.hpp>
#include <glm/glm/gtc/matrix_transform.hpp>
#include <glm/glm/gtc/type_ptr.hpp>
#include <OBJ-Loader/Source/OBJ_Loader.h>

unsigned int SCR_WIDTH = 800;
unsigned int SCR_HEIGHT = 600;
const float PI = 3.14159265358979323846f;
float camYaw = PI / 8;
float camPitch = PI / 12;
float camRadius = 10.0f;
glm::vec3 camPos = glm::vec3(sin(camYaw) * camRadius, sin(camPitch) * camRadius, cos(camYaw) * camRadius);
float camFOV = 45.0f;

float deltaTime = 0.0f;
float lastFrame = 0.0f;
bool firstMouse = true;
bool mousePressed = false;
float lastMouseX = SCR_WIDTH / 2.0f;
float lastMouseY = SCR_HEIGHT / 2.0f;
float mouseSensitivity = 0.007f;

const char* teapot = "./src/models/teapot.obj";

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
    objl::Loader loader;
    if(!loader.LoadFile(file)){
        std::cerr<<"Failed to load OBJ file"<<std::endl;
        return;
    }
    objl::Mesh mesh = loader.LoadedMeshes[0];
    glm::vec3 minBound(FLT_MAX, FLT_MAX, FLT_MAX);
    glm::vec3 maxBound(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    float maxDistance = 0.0f;
    for(const auto &vertex : mesh.Vertices){
        glm::vec3 vertexPos(vertex.Position.X, vertex.Position.Y, vertex.Position.Z);
        minBound = glm::min(minBound, vertexPos);
        maxBound = glm::max(maxBound, vertexPos);
    }
    glm::vec3 center = (minBound + maxBound) * 0.5f;
    for(const auto &vertex : mesh.Vertices){
        glm::vec3 vertexPos(vertex.Position.X, vertex.Position.Y, vertex.Position.Z);
        float distance = glm::length(vertexPos - center);
        maxDistance = std::max(maxDistance, distance);
    }
    glm::vec3 size = maxBound - minBound;
    float aspectRatio = size.x / size.y;
    float scaleFactor = 1.5f / maxDistance;
    float vertices[mesh.Vertices.size() * 8];
    int i = 0;
    for(const auto &vertex : mesh.Vertices){
        glm::vec3 scaledPosition(
            (vertex.Position.X - center.x) * scaleFactor,
            (vertex.Position.Y - center.y) * scaleFactor,
            (vertex.Position.Z - center.z) * scaleFactor
        );
        vertices[i++] = scaledPosition.x;
        vertices[i++] = scaledPosition.y;
        vertices[i++] = scaledPosition.z;
        vertices[i++] = vertex.TextureCoordinate.X;
        vertices[i++] = vertex.TextureCoordinate.Y;
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
int startRenderer(){
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

    unsigned int teapotVAO, teapotVBO, teapotEBO, teapotIndexCount;
    loadModel(teapot, teapotVAO, teapotVBO, teapotEBO, teapotIndexCount);

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

        glm::mat4 projection = glm::perspective(glm::radians(camFOV), (float) SCR_WIDTH / (float) SCR_HEIGHT, 0.1f, 100.0f);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection[0][0]);

        glm::mat4 view = glm::lookAt(camPos, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view[0][0]);

        glUniform3fv(glGetUniformLocation(shaderProgram, "camPos"), 1, &camPos[0]);

        glm::mat4 model = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &model[0][0]);

        glBindVertexArray(teapotVAO);
        
        glDrawElements(GL_TRIANGLES, teapotIndexCount, GL_UNSIGNED_INT, 0);

        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &teapotVAO);
    glDeleteBuffers(1, &teapotVBO);
    glDeleteBuffers(1, &teapotEBO);

    glfwTerminate();
    return 0;
}