#version 430

#ifdef GL_ES
// Set default precision to medium
precision mediump int;
precision mediump float;
#endif


layout(location = 0) in vec4 in_position;
layout(location = 1) in vec3 in_color;

uniform mat4 mvp_matrix;

out vec3 v_color;

void main()
{
    // Calculate vertex position in screen space
    gl_Position = mvp_matrix * in_position;

	v_color = in_color;
}

