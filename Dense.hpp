#pragma once
#include "Matrix.hpp"
#include "Layer.hpp"

class Dense final : public Layer
{
private:
	Matrix W; // Weights
	Matrix b; // Biases
	
	Matrix lastInput; // Store the last input for backpropagation
	Matrix dW; // Weights grad
	Matrix db; // Biases grad
public:
	Dense(int inputSize, int outputSize);
	Dense(int inputSize, int outputSize, double weightmin, double weightmax);
	Dense(const Matrix& W, const Matrix& b); // if W (m * n) then b MUST BE (1 * n)

	Matrix forward(const Matrix& X) override;
	Matrix backward(const Matrix& dZ) override;
	void update(double lr) override;

	void save(std::ofstream& out) const;
	void load(std::ifstream& in);

	size_t getInputSize() const;
	size_t getOutputSize() const;
};