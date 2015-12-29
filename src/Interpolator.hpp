#ifndef __INTERPOLATOR_HPP__
#define __INTERPOLATOR_HPP__



template<class T> class Interpolator
{
public:

	////////////////////////////////////////////////////////////////
	//	Linear interpolation
	//		target  - the target point, 0.0 - 1.0
	//		v       - a pointer to an array of size 2 containg the two values
	//
	inline static T Linear(float target, T *v)
	{
		return (T)(target*(v[1]) + (T(1.0f) - target)*(v[0]));
	}
	

	////////////////////////////////////////////////////////////////
	//	BiLinear interpolation, linear interpolation in 2D
	//		  target  - a 2D point (X,Y)
	//		  v       - an array of size 4 containg values cockwise around the square starting from bottom left
	//	  cost: performs 3 linear interpolations
	//
	inline static T Bilinear(float *target, T *v)
	{
		T v_prime[2] = 
		{
			Linear(target[1], &(v[0])),
			Linear(target[1], &(v[2]))
		};
		
		return Linear(target[0], v_prime);
	}
	

	////////////////////////////////////////////////////////////////
	//	TriLinear interpolation, linear interpolation in 2D
	//		target  - a 3D point (X,Y)
	//		v       - an array of size 8 containg the values of the 8 corners
	//				of a cube defined as two faces: 0-3 face one (front face)
	//				4-7 face two (back face)
	//	cost: 7 linear interpolations
	//
	inline static T Trilinear(float *target, T *v)
	{
		T v_prime[2] = 
		{
			Bilinear(&(target[0]), &(v[0])),
			Bilinear(&(target[1]), &(v[4]))
		};
		
		return Linear(target[2], v_prime);
	}


	inline static void test_usage()
	{
		// Test Linear interpolation
		float fVarsLin[2] = { 1.0, 2.0 };
		int iVarsLin[2] = { 100, 200 };
		
		float targetLin = 0.5;

		std::cout << "Linear (f): " << Interpolator<float>::Linear(targetLin, fVarsLin) << std::endl;
		std::cout << "Linear (i): " << Interpolator<int>::Linear(targetLin, iVarsLin) << std::endl;

		// Test Bilinear interpolation
		float fVarsBilin[4] = { 1.0, 2.0, 3.0, 4.0 };
		int iVarsBilin[4] = { 100, 200, 300, 400 };
		
		float targetBilin[2] = { 0.5, 1.0 };
		
		std::cout << "Bilinear (f): " << Interpolator<float>::Bilinear(targetBilin, fVarsBilin) << std::endl;
		std::cout << "Bilinear (i): " << Interpolator<int>::Bilinear(targetBilin, iVarsBilin) << std::endl;
		
		// Test Trilinear interpolation
		float fVarsTrilin[8] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
		int iVarsTrilin[8] = { 100, 200, 300, 400, 500, 600, 700, 800 };
		
		float targetTrilin[3] = { 0.5, 0.5, 0.5 };
		
		std::cout << "Trilinear (f): " << Interpolator<float>::Trilinear(targetTrilin, fVarsTrilin) << std::endl;
		std::cout << "Trilinear (i): " << Interpolator<int>::Trilinear(targetTrilin, iVarsTrilin) << std::endl;
	}

};


#endif // __INTERPOLATOR_HPP__



