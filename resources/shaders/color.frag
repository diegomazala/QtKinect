#ifdef GL_ES
// Set default precision to medium
precision mediump int;
precision mediump float;
#endif

uniform vec3 color;

void main()
{
	gl_FragColor = vec4(color, 1);
	//gl_FragColor = vec4(0, 1, 1, 1);
}

