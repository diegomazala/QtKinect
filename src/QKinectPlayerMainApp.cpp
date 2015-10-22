#if 0

#include <iostream>
#include <vector>
#include <fstream>

template<typename T>
class VectorFile
{
public:
	VectorFile(){}

	void save(const std::string& filename)
	{
		out_file.open(filename, std::ios::out | std::ios::binary);
		//vector_write(out_file, v1);
		out_file.close();
	}

	void load(const std::string& filename)
	{
		in_file.open(filename, std::ios::in | std::ios::binary);
		//vector_read(in_file, v2);
		in_file.close();
	}

	void write()
	{
		// writing header
		const std::size_t count = header.size();
		out_file.write(reinterpret_cast<const char*>(&count), sizeof(std::size_t));
		out_file.write(reinterpret_cast<const char*>(&header[0]), count * sizeof(T));
	}

	void read()
	{

	}

	void append(const std::vector<T>& append_data)
	{
		data.insert(data.end(), append_data);
	}

private:

	std::ofstream out_file;
	std::ifstream in_file;

	std::vector<T> header;
	std::vector<T> data;
};


template<typename T>
void vector_write(std::ofstream& out_file, const std::vector<T>& data)
{
	const std::size_t count = data.size();
	out_file.write(reinterpret_cast<const char*>(&count), sizeof(std::size_t));
	out_file.write(reinterpret_cast<const char*>(&data[0]), count * sizeof(T));
}


template<typename T>
void vector_read(std::ifstream& in_file, std::vector<T>& data)
{
	std::size_t count;
	in_file.read(reinterpret_cast<char*>(&count), sizeof(std::size_t));
	data.resize(count);
	in_file.read(reinterpret_cast<char*>(&data[0]), count * sizeof(T));
}


bool test_read_write_int()
{
	std::vector<int> v1 = { 1, 2, 3 };
	std::vector<int> v2;

	std::ofstream out_file;
	out_file.open("c:/temp/test_write_int.bin", std::ios::out | std::ios::binary);
	vector_write(out_file, v1);
	out_file.close();

	std::ifstream in_file;
	in_file.open("c:/temp/test_write_int.bin", std::ios::in | std::ios::binary);
	vector_read(in_file, v2);
	in_file.close();

	for (const auto v : v1)
		std::cout << v << ' ';
	std::cout << std::endl;

	for (const auto v : v2)
		std::cout << v << ' ';
	std::cout << std::endl;

	return v1 == v2;
}


bool test_read_write_double()
{
	std::vector<double> v1 = { 1.1, 2.2, 3.3 };
	std::vector<double> v2;

	std::ofstream out_file;
	out_file.open("c:/temp/test_write_double.bin", std::ios::out | std::ios::binary);
	vector_write(out_file, v1);
	out_file.close();

	std::ifstream in_file;
	in_file.open("c:/temp/test_write_double.bin", std::ios::in | std::ios::binary);
	vector_read(in_file, v2);
	in_file.close();

	for (const auto v : v1)
		std::cout << v << ' ';
	std::cout << std::endl;

	for (const auto v : v2)
		std::cout << v << ' ';
	std::cout << std::endl;

	return v1 == v2;
}


int main(int argc, char **argv)
{
	std::cout << "test int   : " << (test_read_write_int() ? "ok" : "fail") << std::endl;
	std::cout << "test double: " << (test_read_write_double() ? "ok" : "fail") << std::endl;

	return EXIT_SUCCESS;
}


#else

#include <QApplication>
#include "MainWindow.h"
#include "QKinectPlayerCtrl.h"
#include <iostream>



int main(int argc, char **argv)
{
	QApplication app(argc, argv);

	MainWindow w;
	w.show();

	QKinectPlayerCtrl controller;
	controller.setView(&w);
	controller.setupConnections();
	controller.startKinect();

	return app.exec();
}
#endif
