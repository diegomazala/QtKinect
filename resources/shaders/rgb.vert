#version 430 core

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec4 in_color;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

uniform vec3 color;

out vec4 v_color;

void main()
{
	mat4 mvp_matrix = projectionMatrix * viewMatrix * modelMatrix;
	gl_Position = mvp_matrix * in_position;

	v_color = vec4(color, 1);
}

