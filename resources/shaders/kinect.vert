#ifdef GL_ES
// Set default precision to medium
precision mediump int;
precision mediump float;
#endif

uniform mat4 mvp_matrix;

attribute vec3 in_position;
attribute vec3 in_color;
attribute vec3 in_normal;

varying vec3 v_color;


void main()
{
    // Calculate vertex position in screen space
    gl_Position = mvp_matrix * vec4(in_position, 1);

    // Pass color to fragment shader
    //v_color = in_color;
	v_color = in_normal;
	//v_color = normalize(in_position);
}

