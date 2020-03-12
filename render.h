#pragma once
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const unsigned int W = 1000;
const unsigned int H = 1000;

__constant__
const float epsilon = 0.01;


struct vec3
{
	float x;
	float y;
	float z;

	__host__ __device__
	vec3(float x, float y, float z) : x(x), y(y), z(z) {}
	__host__ __device__
	vec3() {}

	__host__ __device__
	vec3 operator*(float scalar) { return vec3(x*scalar, y*scalar, z*scalar); }
	__host__ __device__
	vec3 operator/(float scalar) { return vec3(x/scalar, y/scalar, z/scalar); }
	__host__ __device__
	vec3 operator+(vec3 v) { return vec3(x + v.x, y + v.y, z + v.z); }
	__host__ __device__
	vec3 operator-(vec3 v) { return vec3(x - v.x, y - v.y, z - v.z); }
};

__host__ __device__
float length(vec3 v)
{
	return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

__host__ __device__
vec3 norm(vec3 v)
{
	return v / length(v);
}

__host__ __device__
float dot(vec3 v1, vec3 v2)
{
	return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}


struct Camera
{
	vec3 p; //position of camera
	vec3 dir; //theta, phi, fov

	float width;
	float height;

	float cos_theta;
	float sin_theta;

	float cos_phi;
	float sin_phi;

	__host__ __device__
	Camera() {}

	__host__ __device__
	Camera(vec3 p, vec3 dir) : p(p), dir(dir) 
	{
		cos_theta = cos(dir.x);
		sin_theta = sin(dir.x);

		cos_phi = cos(dir.y);
		sin_phi = sin(dir.y);
	}

	__host__ __device__
	vec3 dir_vec(float i, float j)
	{
		float r = width / (2.0f*tan(dir.z / 2.0f));

		vec3 x(cos_theta * cos_phi, cos_theta*sin_phi, -sin_theta);
		vec3 y(-cos_theta * sin_phi, cos_theta*cos_phi, -sin_theta);
		vec3 z(sin_theta*cos_phi, sin_theta*sin_phi, cos_theta);

		return norm(z * r + x * i + y * j);
	}

	__host__ __device__
	vec3 p0(float i, float j)
	{
		vec3 x(cos_theta * cos_phi, cos_theta*sin_phi, -sin_theta);
		vec3 y(-cos_theta * sin_phi, cos_theta*cos_phi, -sin_theta);

		return (p + x * i + y * j);
	}
};


__host__ __device__
float dist(vec3 p)
{
	return length(p) - 1.0f;
}


__host__ __device__
vec3 gradient(vec3 pos)
{
	vec3 dx( 0.0f, 0.0f, epsilon );
	vec3 dy( 0.0f, epsilon, 0.0f );
	vec3 dz( epsilon, 0.0f, 0.0f );
	return vec3(dist(pos + dx) - dist(pos - dx), dist(pos + dy) - dist(pos - dy), dist(pos + dz) - dist(pos - dz));
}


__global__
void render(unsigned char pixels[W*H], Camera camera)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	camera.width = 1.0f;

	for (unsigned int k = index; k < W*H; k += stride)
	{
		unsigned int j = k / W;
		unsigned int i = k - j * W;

		float x;
		float y;


		x = (float)i / W - 0.5f;
		y = (float)j / H - 0.5f;
		vec3 dir = camera.dir_vec(x, y);
		vec3 p = camera.p0(x, y);

		pixels[k] = 0x00;
		float d;
		for (unsigned int t = 0; t < 150; t++)
		{	
			d = dist(p);
			p = p + dir * d;

			if (d < 0.01)
			{
				float light = dot(norm(gradient(p)), norm(vec3(-1.0f, 1.0f, 1.0f)));
				pixels[k] = 0xff * (0.5f + light * 0.5f) * (0.5f + light * 0.5f) * (0.5f + light * 0.5f);
				break;
			}
		}
	}
}
