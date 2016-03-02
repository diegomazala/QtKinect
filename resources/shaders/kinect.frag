#ifdef GL_ES
// Set default precision to medium
precision mediump int;
precision mediump float;
#endif

//uniform sampler2D texture;

varying vec3 v_color;

//! [0]
void main()
{
    // Set fragment color from texture
    //gl_FragColor = texture2D(texture, v_texcoord);
	
	gl_FragColor = vec4(v_color, 1);
}
//! [0]

