#pragma once
#include <memory>

#include "Dense.hpp"
#include "Activation.hpp"
#include "MSE.hpp"

class Perceptron
{
private:
	size_t inputSize;
	size_t outputSize;

	double loss;
	Dense dense;
	std::unique_ptr<Activation> activation;
	std::unique_ptr<Loss> lossFunc;
public:
	Perceptron(size_t inputSize, size_t outputSize);
	Perceptron(size_t inputSize, size_t outputSize, std::unique_ptr<Activation> activation, std::unique_ptr<Loss> lossFunc);

	void trainOnce(const Matrix& X, const Matrix& Y, double learningRate);
	void train(const Matrix& X, const Matrix& Y, size_t epochs, double learningRate);
	double getLastLoss();
	Matrix predict(const Matrix& X);
};