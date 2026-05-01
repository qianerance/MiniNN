#include <stdexcept>
#include <fstream>
#include <cmath>
#include <cstdint>
#include "Dense.hpp"
#include "Matrix.hpp"

Dense::Dense(int inputSize, int outputSize) : W(inputSize, outputSize, -0.1, 0.1), b(1, outputSize)
{ }

Dense::Dense(int inputSize, int outputSize, double weightmin, double weightmax)
	: W(inputSize, outputSize, weightmin, weightmax), b(1, outputSize)
{ }

Dense::Dense(const Matrix& W, const Matrix& b) 
{
	if (W.getCols() != b.getCols() || b.getRows() != 1)
		throw std::invalid_argument("W and b dimensions are incompatible. W should be (m * n) and b should be (1 * n).");

	this->W = W;
	this->b = b;
}

Matrix Dense::forward(const Matrix& X)
{
	if (X.getCols() != W.getRows())
		throw std::invalid_argument("Batch input size does not match the expected input size of the layer.");
	
	lastInput = X;

	Matrix Z = X * W;
	Matrix b = this->b.broadcast_down(Z.getRows());

	return Z + b;
}

Matrix Dense::backward(const Matrix& dZ)
{
	if (dZ.getRows() != lastInput.getRows() || dZ.getCols() != W.getCols())
		throw std::invalid_argument("dZ dimensions do not match the expected output dimensions of the layer.");

	dW = lastInput.transpose() * dZ;
	db = dZ.rowsum();

	Matrix dX = dZ * W.transpose();
	return dX;
}

void Dense::update(double lr)
{
	W = W - lr * dW;
	b = b - lr * db;
}

void Dense::save(std::ofstream& out) const
{
	uint32_t r = static_cast<uint32_t>(W.getRows());
	uint32_t c = static_cast<uint32_t>(W.getCols());

	out.write((char*)&r, sizeof(uint32_t));
	out.write((char*)&c, sizeof(uint32_t));

	double val;
	for (int i = 0; i < W.getRows(); ++i)
		for (int j = 0; j < W.getCols(); ++j)
		{
			val = W(i, j);
			out.write((char*)&val, sizeof(double));
		}

	uint32_t br = static_cast<uint32_t>(b.getRows());
	uint32_t bc= static_cast<uint32_t>(b.getCols());

	out.write((char*)&br, sizeof(uint32_t));
	out.write((char*)&bc, sizeof(uint32_t));

	for (int i = 0; i < b.getRows(); ++i)
		for (int j = 0; j < b.getCols(); ++j)
		{
			val = b(i, j);
			out.write((char*)&val, sizeof(double));
		}
}

void Dense::load(std::ifstream& in)
{
	uint32_t r, c;
	in.read((char*)&r, sizeof(uint32_t));
	in.read((char*)&c, sizeof(uint32_t));
	
	W = Matrix(r, c);
	for (uint32_t i = 0; i < r; ++i)
		for (uint32_t j = 0; j < c; ++j)
		{
			in.read((char*)&W(i, j), sizeof(double));
		}

	in.read((char*)&r, sizeof(uint32_t));
	in.read((char*)&c, sizeof(uint32_t));

	b = Matrix(r, c);
	for (uint32_t i = 0; i < r; ++i)
		for (uint32_t j = 0; j < c; ++j)
		{
			in.read((char*)&b(i, j), sizeof(double));
		}
}

size_t Dense::getInputSize() const { return W.getRows(); }
size_t Dense::getOutputSize() const { return W.getCols(); }