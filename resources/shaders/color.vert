#ifdef GL_ES
// Set default precision to medium
precision mediump int;
precision mediump float;
#endif

uniform mat4 mvp_matrix;

attribute vec4 in_position;



void main()
{
	gl_PointSize = 5;
	// Calculate vertex position in screen space
    //gl_Position = mvp_matrix * vec4(in_position, 1);
	gl_Position = mvp_matrix * in_position;
}

