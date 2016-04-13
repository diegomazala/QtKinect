#version 430 core

in vec4 v_color;

out vec3 finalColor;

void main()
{
	finalColor = v_color.xyz;
}
