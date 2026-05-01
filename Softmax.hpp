#pragma once
#include "Activation.hpp"

class Softmax final : public Activation
{
private:
	Matrix lastOutput;
	//std::function<double(double)> func;
	//std::function<double(double)> derivative;
public:
	Softmax();

	Matrix forward(const Matrix& X) override;
	Matrix backward(const Matrix& dA) override;
};