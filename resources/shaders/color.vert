#version 430 core

in vec4 in_position;
in vec4 in_color;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

out vec4 v_color;

void main()
{
	mat4 mvp_matrix = projectionMatrix * viewMatrix * modelMatrix;
	gl_Position = mvp_matrix * in_position;

	v_color = in_color;
}