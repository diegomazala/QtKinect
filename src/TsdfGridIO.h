

#ifndef __TSDF_GRID_IO_H__
#define __TSDF_GRID_IO_H__

#include <string>
#include <fstream>
#include <vector>
#include <Eigen/Dense>


static void save_raw_file(const std::string& filename, const std::size_t size, const void* data)
{
	std::ofstream out_file(filename, std::ios::binary);
	if (out_file.is_open())
	{
		out_file.write(reinterpret_cast<const char*>(data), size);
		out_file.close();
	}
}

static void* load_raw_file(const std::string& filename, const std::size_t size)
{
	char* data = nullptr;
	std::ifstream in_file(filename, std::ifstream::binary);
	if (in_file.is_open())
	{
		data = new char[size];
		in_file.read(data, size);
		in_file.close();
	}
	return reinterpret_cast<void*>(data);
}

template<typename Type>
static void save_array(std::ofstream& out_file, const Type* array, std::size_t count)
{
	out_file.write(reinterpret_cast<const char*>(&count), sizeof(std::size_t));
	out_file.write(reinterpret_cast<const char*>(array), sizeof(Type) * count);
}

template<typename Type>
static void load_array(std::ifstream& in_file, Type* array, std::size_t& count)
{
	in_file.read(reinterpret_cast<char*>(&count), sizeof(std::size_t));
	in_file.read(reinterpret_cast<char*>(array), sizeof(Type) * count);
}

template<typename MatrixType>
static void eigen_binary_save(std::ofstream& out_file, const MatrixType& m)
{
	const MatrixType::Index rows = m.rows();
	const MatrixType::Index cols = m.cols();
	out_file.write(reinterpret_cast<const char*>(&rows), sizeof(MatrixType::Index));
	out_file.write(reinterpret_cast<const char*>(&cols), sizeof(MatrixType::Index));
	out_file.write(reinterpret_cast<const char*>(m.data()), sizeof(MatrixType::Scalar) * m.rows() * m.cols());
}

template<typename MatrixType>
void eigen_binary_load(std::ifstream& in_file, MatrixType& m)
{
	MatrixType::Index rows, cols;
	in_file.read(reinterpret_cast<char*>(&rows), sizeof(MatrixType::Index));
	in_file.read(reinterpret_cast<char*>(&cols), sizeof(MatrixType::Index));
	m.resize(rows, cols);
	in_file.read(reinterpret_cast<char*>(m.data()), sizeof(MatrixType::Scalar) * rows * cols);
	if (in_file.bad())
		throw std::exception("Error reading matrix");
}


template<typename MatrixType>
static void vector_eigen_binary_save(std::ofstream& out_file, const std::vector<MatrixType>& vec)
{
	const std::size_t count = vec.size();
	out_file.write(reinterpret_cast<const char*>(&count), sizeof(std::size_t));
	for (const MatrixType& m : vec)
		eigen_binary_save(out_file, m);
}


template<typename MatrixType>
static void vector_eigen_binary_load(std::ifstream& in_file, std::vector<MatrixType>& vec)
{
	std::size_t count = 0;
	in_file.read(reinterpret_cast<char*>(&count), sizeof(std::size_t));
	vec.resize(count);
	for (MatrixType& m : vec)
		eigen_binary_load(in_file, m);
}


template <typename Type>
static void tsdf_grid_load(
	const std::string& filename,
	Eigen::Matrix<Type, 3, 1>& volume_size,
	Eigen::Matrix<Type, 3, 1>& voxel_size,
	std::vector<Eigen::Matrix<Type, 2, 1>>& params)
{
	std::ifstream in_file(filename, std::ifstream::binary);
	if (in_file.is_open())
	{
		eigen_binary_load(in_file, volume_size);
		eigen_binary_load(in_file, voxel_size);
		vector_eigen_binary_load(in_file, params);
		in_file.close();
	}
	else
	{
		throw(const std::exception("could not open file for loading"));
	}
}


template <typename Type>
static void tsdf_grid_save(
	const std::string& filename, 
	const Eigen::Matrix<Type, 3, 1>& volume_size,
	const Eigen::Matrix<Type, 3, 1>& voxel_size,
	const std::vector<Eigen::Matrix<Type, 2, 1>>& params)
{
	std::ofstream out_file(filename, std::ofstream::binary);
	if (out_file.is_open())
	{
		eigen_binary_save(out_file, volume_size);
		eigen_binary_save(out_file, voxel_size);
		vector_eigen_binary_save(out_file, params);
		out_file.close();
	}
	else
	{
		throw(const std::exception("could not open file for saving"));
	}
}





#endif	//__TSDF_GRID_IO_H__
