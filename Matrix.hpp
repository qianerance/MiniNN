#pragma once
#include <vector>
#include <functional>

class Matrix
{
private:
	int rows;
	int cols;
	std::vector<double> data;
public:
	Matrix();
	Matrix(int rows, int cols);
	explicit Matrix(int size);
	Matrix(int rows, int cols, double randmin, double randmax);
	double& operator()(int row, int col);
	double operator()(int row, int col) const;

	Matrix operator+(const Matrix& other) const;
	Matrix operator-(const Matrix& other) const;
	Matrix operator*(const Matrix& other) const;

	Matrix operator*(double scalar) const;
	friend Matrix operator*(double scalar, const Matrix& m);
	Matrix operator/(double scalar) const;

	static Matrix pow(const Matrix& m, double exponent);

	Matrix hadamard(const Matrix& other) const;

	Matrix transpose() const;
	Matrix apply(std::function<double(double)> func) const;
	Matrix rowsum() const;
	Matrix broadcast_down(int rows) const;
	Matrix sliceRows(size_t start, size_t count);

	Matrix shuffleRows(std::vector<size_t> indices) const;
	size_t argmax(size_t row) const;

	int getRows() const { return rows; }
	int getCols() const { return cols; }
};