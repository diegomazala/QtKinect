#ifndef _OBJ_FILE_H_
#define _OBJ_FILE_H_

#include <iostream>
#include <chrono>
#include <vector>
#include <sstream>
#include <fstream>
#include <Eigen/Dense>



static bool import_obj(const std::string& filename, std::vector<Eigen::Vector3d>& points3D, int max_point_count = INT_MAX)
{
	std::ifstream inFile;
	inFile.open(filename);

	if (!inFile.is_open())
	{
		std::cerr << "Error: Could not open obj input file: " << filename << std::endl;
		return false;
	}

	points3D.clear();

	int i = 0;
	while (inFile)
	{
		std::string str;

		if (!std::getline(inFile, str))
		{
			if (inFile.eof())
				return true;

			std::cerr << "Error: Problems when reading obj file: " << filename << std::endl;
			return false;
		}

		if (str[0] == 'v')
		{
			std::stringstream ss(str);
			std::vector <std::string> record;

			char c;
			double x, y, z;
			ss >> c >> x >> y >> z;

			Eigen::Vector3d p(x, y, z);
			points3D.push_back(p);
		}

		if (i++ > max_point_count)
			break;
	}

	inFile.close();
	return true;
}

static bool import_obj(const std::string& filename, std::vector<Eigen::Vector3f>& points3D, int max_point_count = INT_MAX)
{
	std::ifstream inFile;
	inFile.open(filename);

	if (!inFile.is_open())
	{
		std::cerr << "Error: Could not open obj input file: " << filename << std::endl;
		return false;
	}

	points3D.clear();

	int i = 0;
	while (inFile)
	{
		std::string str;

		if (!std::getline(inFile, str))
		{
			if (inFile.eof())
				return true;

			std::cerr << "Error: Problems when reading obj file: " << filename << std::endl;
			return false;
		}

		if (str[0] == 'v')
		{
			std::stringstream ss(str);
			std::vector <std::string> record;

			char c;
			float x, y, z;
			ss >> c >> x >> y >> z;

			Eigen::Vector3f p(x, y, z);
			points3D.push_back(p);
		}

		if (i++ > max_point_count)
			break;
	}

	inFile.close();
	return true;
}

static bool import_obj(const std::string& filename, std::vector<Eigen::Vector4f>& points3D, int max_point_count = INT_MAX)
{
	std::ifstream inFile;
	inFile.open(filename);

	if (!inFile.is_open())
	{
		std::cerr << "Error: Could not open obj input file: " << filename << std::endl;
		return false;
	}

	points3D.clear();

	int i = 0;
	while (inFile)
	{
		std::string str;

		if (!std::getline(inFile, str))
		{
			if (inFile.eof())
				return true;

			std::cerr << "Error: Problems when reading obj file: " << filename << std::endl;
			return false;
		}

		if (str[0] == 'v')
		{
			std::stringstream ss(str);
			std::vector <std::string> record;

			char c;
			double x, y, z;
			ss >> c >> x >> y >> z;

			Eigen::Vector4f p(x, y, z, 1.0f);
			points3D.push_back(p);
		}

		if (i++ > max_point_count)
			break;
	}

	inFile.close();
	return true;
}


static bool import_obj(const std::string& filename, std::vector<Eigen::Vector4f>& vertices, std::vector<Eigen::Vector3f>& normals)
{
	std::ifstream inFile;
	inFile.open(filename);

	if (!inFile.is_open())
	{
		std::cerr << "Error: Could not open obj input file: " << filename << std::endl;
		return false;
	}

	vertices.clear();
	normals.clear();

	int i = 0;
	while (inFile)
	{
		std::string str;

		if (!std::getline(inFile, str))
		{
			if (inFile.eof())
				return true;

			std::cerr << "Error: Problems when reading obj file: " << filename << std::endl;
			return false;
		}

		if (str[0] == 'v')
		{
			std::stringstream ss(str);

			char c[3];
			float x, y, z, w;
			
			ss >> c >> x >> y >> z >> w;
			

			if (str[1] == 'n')			// read normal
			{
				normals.push_back(Eigen::Vector3f(x, y, z));
			}
			else						// read vertex
			{
				Eigen::Vector4f p(x, y, z, 1.0f);
				if ((int)w != 0) p.w() = w;
				vertices.push_back(p);
			}
		}
	}

	inFile.close();
	return true;
}




template<typename Type, int Rows>
static void export_obj_with_normals(const std::string& filename, const std::vector<Eigen::Matrix<Type, Rows, 1>>& vertices, const std::vector<Eigen::Matrix<Type, Rows, 1>>& normals)
{
	std::ofstream file;
	file.open(filename);
	for (const auto v : vertices)
		file << std::fixed << "v " << v.transpose() << std::endl;
	for (const auto n : normals)
		file << std::fixed << "vn " << n.transpose() << std::endl;
	file.close();
}

template<typename Type, int Rows>
static void export_obj_with_colors(const std::string& filename, const std::vector<Eigen::Matrix<Type, Rows, 1>>& vertices, const std::vector<Eigen::Vector3f>& rgb)
{
	std::ofstream file;
	file.open(filename);
	for (int i = 0; i < vertices.size(); ++i)
	{
		const auto& v = vertices[i];
		const auto& c = rgb[i];
		file << std::fixed << "v " << v.transpose() << '\t' << c.transpose() << std::endl;
	}
	file.close();
}



template<typename Type, int Rows>
static void export_obj(const std::string& filename, const std::vector<Eigen::Matrix<Type, Rows, 1>>& vertices)
{
	std::ofstream file;
	file.open(filename);
	for (const auto v : vertices)
	{
		file << std::fixed << "v " << v.transpose() << std::endl;
	}
	file.close();
}

template<typename Type>
static void export_obj(const std::string& filename, const Type* vertex_array, const size_t vertex_count, const size_t tuple_size)
{
	std::ofstream file;
	file.open(filename);
	for (size_t i = 0; i < vertex_count * tuple_size; i += tuple_size)
	{
		file << std::fixed << "v ";
		for (size_t j = 0; j < tuple_size; ++j)
			file << vertex_array[i+j] << ' ';
		file << std::endl;
	}
	file.close();
}



static void export_obj(const std::string& filename, const std::vector<Eigen::Vector3d>& points3D, const Eigen::Matrix4d& proj, const Eigen::Matrix4d& view, int window_width, int window_height)
{
	std::ofstream file;
	file.open(filename);
	for (const auto X : points3D)
	{
		Eigen::Vector4d clip = proj * view * X.homogeneous();
		const Eigen::Vector3d ndc = (clip / clip.w()).head<3>();
		Eigen::Vector3f pixel;
		pixel.x() = window_width / 2.0f * ndc.x() + window_width / 2.0f;
		pixel.y() = window_height / 2.0f * ndc.y() + window_height / 2.0f;
		pixel.z() = 0.0; // (far_plane - near_plane) / 2.0f * ndc.z() + (far_plane + near_plane) / 2.0f;

		file << std::fixed << "v " << pixel.transpose() << std::endl;
	}
	file.close();
}

#endif // _OBJ_FILE_H_
