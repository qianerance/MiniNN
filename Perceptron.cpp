#include "Perceptron.hpp"
#include "Activation.hpp"
#include "Elementwise.hpp"


Perceptron::Perceptron(size_t inputSize, size_t outputSize) : 
	dense(Dense(inputSize, outputSize)),
	activation(std::make_unique<Elementwise>(Elementwise::Sigmoid())), 
	lossFunc(std::make_unique<MSE>()), 
	inputSize(inputSize), 
	outputSize(outputSize), 
	loss(0)
{}

Perceptron::Perceptron(size_t inputSize, size_t outputSize, std::unique_ptr<Activation> activation, std::unique_ptr<Loss> lossFunc) :
	dense(Dense(inputSize, outputSize)),
	activation(std::move(activation)),
	lossFunc(std::move(lossFunc)),
	inputSize(inputSize),
	outputSize(outputSize),
	loss(0)
{}

void Perceptron::trainOnce(const Matrix& X, const Matrix& Y, double learningRate)
{
	Matrix y_pred = activation->forward(dense.forward(X)); // prediction with applied activation

	loss = lossFunc->forward(y_pred, Y); // loss 

	Matrix dLoss = lossFunc->backward(); // dLoss
	Matrix dA = activation->backward(dLoss); // activation backward with loss
	dense.backward(dA);
	dense.update(learningRate);
}

void Perceptron::train(const Matrix& X, const Matrix& Y, size_t epochs, double learningRate)
{
	// Y = TRUE VALUES
	for (size_t i = 0; i < epochs; ++i)
		trainOnce(X, Y, learningRate);
}

double Perceptron::getLastLoss()
{
	return loss;
}

Matrix Perceptron::predict(const Matrix& X)
{
	return activation->forward(dense.forward(X));
}
