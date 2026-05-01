#include <stdexcept>
#include <cmath>
#include "Elementwise.hpp"

Elementwise::Elementwise(std::function<double(double)> func, 
	std::function<double(double)> derivative) : func(func), derivative(derivative)
{ }

Matrix Elementwise::forward(const Matrix& X)
{
	lastInput = X;
	return X.apply(func);
}

Matrix Elementwise::backward(const Matrix& dA)
{
	if (dA.getRows() != lastInput.getRows() || dA.getCols() != lastInput.getCols())
		throw std::invalid_argument("dA dimensions do not match activation input dimensions.");

	Matrix dF = lastInput.apply(derivative);
	return dA.hadamard(dF);
}

Elementwise Elementwise::Sigmoid()
{
	std::function<double(double)> sigmoidFunc = [](double x) 
		{
			return 1.0 / (1.0 + std::exp(-x));
		};
	std::function<double(double)> sigmoidDeriv = [](double x) 
		{
			return (std::exp(-x)) / (std::pow((1 + std::exp(-x)), 2)); // sigma(x) * (1 - sigma(x))
		};
	return Elementwise(sigmoidFunc, sigmoidDeriv);
}

Elementwise Elementwise::ReLU()
{
	std::function<double(double)> reluFunc = [](double x)
		{
			return x > 0 ? x : 0;
		};
	std::function<double(double)> reluDeriv = [](double x)
		{
			return x > 0 ? 1 : 0;
		};
	return Elementwise(reluFunc, reluDeriv);
}

Elementwise Elementwise::Tanh()
{
	std::function<double(double)> tanhFunc = [](double x)
		{
			return std::tanh(x);
		};
	std::function<double(double)> tanhDeriv = [](double x)
		{
			double t = std::tanh(x);
			return 1 - t * t;
		};
	return Elementwise(tanhFunc, tanhDeriv);
}

Elementwise Elementwise::None()
{
	std::function<double(double)> f = [](double x)
		{
			return x;
		};
	return Elementwise(f, f);
}
