#include <vector>
#include <stdexcept>
#include <functional>
#include <random>
#include "Matrix.hpp"

double randomDouble(double randmin, double randmax)
{
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(randmin, randmax);
	return dist(gen);
}

Matrix::Matrix() : rows(0), cols(0)
{}

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows * cols)
{}

Matrix::Matrix(int size) : rows(size), cols(size), data(size * size)
{}

Matrix::Matrix(int rows, int cols, double randmin, double randmax) : rows(rows), cols(cols), data(rows * cols)
{
	for (int i = 0; i < rows * cols; ++i)
		data[i] = randomDouble(randmin, randmax);
}

double &Matrix::operator()(int row, int col)
{
	return data[row * cols + col];
}

double Matrix::operator()(int row, int col) const
{
	return data[row * cols + col];
}

Matrix Matrix::operator+(const Matrix& other) const
{
	if (this->rows != other.rows || this->cols != other.cols)
		throw std::invalid_argument("Matrices must have the same dimensions for addition.");

	Matrix result(this->rows, this->cols);
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			result(i, j) = (*this)(i, j) + other(i, j);
	return result;
}

Matrix Matrix::operator-(const Matrix& other) const
{
	if (this->rows != other.rows || this->cols != other.cols)
		throw std::invalid_argument("Matrices must have the same dimensions for subtraction.");

	Matrix result(this->rows, this->cols);
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			result(i, j) = (*this)(i, j) - other(i, j);
	return result;
}

Matrix Matrix::operator*(const Matrix& other) const
{
	if (this->cols != other.rows)
		throw std::invalid_argument("Incompatible matrix dimensions for multiplication.");

	Matrix result(this->rows, other.cols);

	double sum = 0.0;

	Matrix otherT = other.transpose();
#pragma omp parallel for
	for(int i = 0; i < this->rows; ++i)
	{
		for (int j = 0; j < other.cols; ++j)
		{
			sum = 0.0;
			for (int k = 0; k < this->cols; ++k)
			{
				sum += (*this)(i, k) * otherT(j, k); // otherT(k, j);
			}
			result(i, j) = sum;
		}
	}

	return result;
}

Matrix Matrix::operator*(double scalar) const
{
	Matrix result(this->rows, this->cols);

	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			result(i, j) = (*this)(i, j) * scalar;

	return result;
}

Matrix Matrix::operator/(double scalar) const
{
	Matrix result(this->rows, this->cols);
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			result(i, j) = (*this)(i, j) / scalar;
	return result;
}

Matrix Matrix::pow(const Matrix& m, double exponent)
{
	size_t rows = m.getRows();
	size_t cols = m.getCols();
	Matrix result(rows, cols);

	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < cols; ++j)
			result(i, j) = std::pow(m(i, j), exponent);

	return result;
}

Matrix Matrix::hadamard(const Matrix& other) const
{
	if (this->rows != other.rows || this->cols != other.cols)
		throw std::invalid_argument("Matrices must have the same dimensions for element-wise multiplication.");

	Matrix result(this->rows, this->cols);
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			result(i, j) = (*this)(i, j) * other(i, j);

	return result;
}

Matrix operator*(double scalar, const Matrix& m)
{
	return m * scalar;
}

Matrix Matrix::transpose() const
{
	Matrix result(this->cols, this->rows);
	
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			result(j, i) = (*this)(i, j);

	return result;
}

Matrix Matrix::apply(std::function<double(double)> func) const
{
	Matrix result(this->rows, this->cols);

	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			result(i, j) = func((*this)(i, j));

	return result;
}

Matrix Matrix::rowsum() const
{
	Matrix result(1, this->cols);

	for (int j = 0; j < cols; ++j)
	{
		double sum = 0.0;
		for (int i = 0; i < rows; ++i)
			sum += (*this)(i, j);
		result(0, j) = sum;
	}
	return result;
}

Matrix Matrix::broadcast_down(int rows) const
{
	Matrix result(rows, this->cols);
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			result(i, j) = (*this)(0, j);
	return result;
}

Matrix Matrix::sliceRows(size_t start, size_t count)
{
	if (start + count > rows)
        count = rows - start;

	Matrix m = Matrix(count, this->getCols());
	for (size_t i = 0; i < count; ++i)
		for (size_t j = 0; j < this->getCols(); ++j)
			m(i, j) = (*this)(start + i, j);
	return m;
}

Matrix Matrix::shuffleRows(std::vector<size_t> indices) const
{
	if (indices.size() != this->getRows())
		throw std::invalid_argument("Indices size must match the number of rows in the matrix.");

	Matrix result(this->rows, this->cols);
	for (size_t i = 0; i < indices.size(); ++i)
		for (size_t j = 0; j < this->getCols(); ++j)
			result(i, j) = (*this)(indices[i], j);

	return result;
	
}

size_t Matrix::argmax(size_t row) const
{
	double maxVal = (*this)(row, 0);
	size_t index = 0;
	for (size_t j = 1; j < cols; ++j)
		if ((*this)(row, j) > maxVal)
		{
			maxVal = (*this)(row, j);
			index = j;
		}

	return index;
}
