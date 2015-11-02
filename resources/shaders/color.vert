#ifdef GL_ES
// Set default precision to medium
precision mediump int;
precision mediump float;
precision mediump short;
#endif


attribute vec3 a_position;
attribute vec2 a_texcoord;

uniform mat4 mvp_matrix;
uniform float farPlane;

varying vec2 v_texcoord;
varying float depth;

void main()
{
    // Calculate vertex position in screen space
	vec4 pos = vec4(a_position, 1);
    gl_Position = mvp_matrix * pos;

    // Pass texture coordinate to fragment shader
    // Value will be automatically interpolated to fragments inside polygon faces
    v_texcoord = a_texcoord;

	depth = gl_Position.z / farPlane; // do not divide by w
}
