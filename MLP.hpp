#pragma once
#include <vector>
#include <memory>
#include <string>

#include "Matrix.hpp"
#include "Dense.hpp"
#include "Activation.hpp"
#include "MSE.hpp"
#include "Layer.hpp"

class MLP
{
private:
	size_t inputSize;
	size_t currentOutputSize;

	double loss;
	std::vector<std::unique_ptr<Layer>> layers;
	std::unique_ptr<Loss> lossFunc;

	Matrix forward(const Matrix& X);
public:
	MLP(size_t inputSize, std::unique_ptr<Loss> lossFunc);
	void addLayer(size_t neurons, std::unique_ptr<Activation> f, double weightmin, double weightmax);
	void trainOnce(const Matrix& X, const Matrix& Y, double learningRate);
	void train(const Matrix& X, const Matrix& Y, size_t epochs, double learningRate);
	Matrix predict(const Matrix& X);
	double getLastLoss() const;

	void save(const std::string& filename) const;
	void load(const std::string& filename);
};