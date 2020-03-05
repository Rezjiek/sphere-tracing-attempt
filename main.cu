#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <chrono>
#include "render.h"


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}



//vertices and texture in domain [-1,1] of the screen
float data[] = 
{ 
	-1.0f, -1.0f,	0.0f, 0.0f,
	-1.0f, 1.0f,	0.0f, 1.0f,
	1.0f, -1.0f,	1.0f, 0.0f,
	1.0f, 1.0f,		1.0f, 1.0f
};

//vertex shader written in glsl
const char* vertexShader = 
"#version 330 core\n"
"layout(location = 1) in vec2 vertexUV;\n"
"layout(location = 0) in vec4 position;\n"
"out vec2 UV;\n"
"void main()\n"
"{"
"gl_Position = position;\n"
"UV = vertexUV;\n"
"}";

//fragment shader written in glsl
const char* fragmentShader =
"#version 330 core\n"
"out vec4 color;\n"
"uniform sampler2D textureSampler;\n"
"in vec2 UV;\n"
"void main()\n"
"{\n"
"color = vec4(texture(textureSampler, UV).r, 0.0f, 0.0f, 1.0f);\n"
"}\n";



int tex;

//setting up fragment and vertex shader
static unsigned int CreateShader(const char* &vertexShader, const char* &fragmentShader)
{
	unsigned int program = glCreateProgram(); //initialize program

	unsigned int vertex_shader = glCreateShader(GL_VERTEX_SHADER); //initialize vertex shader
	glShaderSource(vertex_shader, 1, &vertexShader, nullptr); //gives the vertex shader the corresponding glsl code
	glCompileShader(vertex_shader); //compiles the glsl code

	unsigned int fragment_shader = glCreateShader(GL_FRAGMENT_SHADER); //initialize fragment shader
	glShaderSource(fragment_shader, 1, &fragmentShader, nullptr); //gives the fragment shader the corresponding glsl code
	glCompileShader(fragment_shader); //compiles the glsl code

	int result;
	glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &result);
	if (result == GL_FALSE) 
	{
		int length;
		glGetShaderiv(fragment_shader, GL_INFO_LOG_LENGTH, &length);
		char* message = new char[length];
		glGetShaderInfoLog(fragment_shader, length, &length, message);
		std::cout << message << std::endl;
	}



	glAttachShader(program, vertex_shader); //adds the vertex shader to the program
	glAttachShader(program, fragment_shader); //adds the fragment shader to the program

	glLinkProgram(program); //links the program
	glValidateProgram(program); //checks if the program works

	tex = glGetUniformLocation(program, "textureSampler");

	glDeleteShader(vertex_shader); //deletes the vertex shader (since it has been compiled)
	glDeleteShader(fragment_shader); //deletes the fragment shader (since it has been compiled)

	return program;
}

int main()
{
	unsigned char* pixels = new unsigned char[W*H];
	Camera camera(vec3(0.0f, 0.0f, -3.0f) , vec3(0.0f, 0.0f, 0.5f));

	cudaMallocManaged(&pixels, W*H * sizeof(unsigned char));

	unsigned int threads = 256;
	unsigned int blocks = 1000;

	//dim3 threads(16, 16);
	//dim3 blocks(W / threads.x + 1, H / threads.y + 1);


	glfwInit(); //initialize glfw
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); //version of glfw 3.x
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); //version of glfw x.3 (so 3.3 since the first one was also a 3)
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); //something with profile (?)

	GLFWwindow* window = glfwCreateWindow(W, H, "test", NULL, NULL); //initialize window
	if (window == NULL) //check if window is created
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window); //complete window setup

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) //check if GLAD is working
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback); //activates framebuffer_size_callback when the size of the window's framebuffer changes

	unsigned int vbo; //vertex buffer object
	glGenVertexArrays(1, &vbo); //setup vertex array object
	glBindVertexArray(vbo); //activates vertex array object

	glGenBuffers(1, &vbo); //setup buffer object
	glBindBuffer(GL_ARRAY_BUFFER, vbo); //activates buffer object
	glBufferData(GL_ARRAY_BUFFER, 16 * sizeof(float), data, GL_STATIC_DRAW); //initializes buffer object data store

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0); //define how the vertex data is processed
	glEnableVertexAttribArray(0); //disable a generic vertex attribute array

	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float))); //define how the vertex data is processed
	glEnableVertexAttribArray(1); //disable a generic vertex attribute array


	unsigned int shader = CreateShader(vertexShader, fragmentShader); //create shader program
	
	glUseProgram(shader); //execute shader

	if (tex == -1) { std::cout << "kkr" << std::endl; }


	unsigned __int64 start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	render<<<blocks, threads>>>(pixels, camera);

	cudaDeviceSynchronize();

	unsigned __int64 end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	std::cout << end - start << std::endl;

	//for (unsigned int k = 0; k < W*H; k++)
	//{
		//std::cout << (int)pixels[k];
	//}



	unsigned int texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, W, H, 0, GL_RED, GL_UNSIGNED_BYTE, pixels);
	glGenerateMipmap(GL_TEXTURE_2D);



	while (!glfwWindowShouldClose(window)) //main window loop, closes when window should close
	{
		unsigned __int64 start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		camera.p.z -= 0.01f;
		render<<<blocks, threads>>>(pixels, camera);
		
		cudaDeviceSynchronize();
		unsigned __int64 end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		std::cout << end - start << std::endl;




		glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, W, H, 0, GL_RED, GL_UNSIGNED_BYTE, pixels);

		//input

		//render
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f); //standard background color (the remaining when cleared)
		glClear(GL_COLOR_BUFFER_BIT); //clears the color buffer (so there is no overlap with previous frames)

		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); //draw array data
		

		//check events and swap the buffers
		glfwPollEvents(); //checks for any events like button presses
		glfwSwapBuffers(window); //swaps the color buffer of the window
	}

	glfwTerminate();
	cudaFree(pixels);
	return 0;
}
