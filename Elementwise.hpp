#pragma once
#include <functional>
#include "Matrix.hpp"
#include "Activation.hpp"

class Elementwise final : public Activation
{
private:
	Matrix lastInput;
	std::function<double(double)> func;
	std::function<double(double)> derivative;
public:
	Elementwise(std::function<double(double)> func, std::function<double(double)> derivative);

	Matrix forward(const Matrix& X) override;
	Matrix backward(const Matrix& dA) override;

	static Elementwise Sigmoid();
	static Elementwise ReLU();
	static Elementwise Tanh();
	static Elementwise None();
};