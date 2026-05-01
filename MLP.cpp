#include <iterator>
#include <fstream>
#include "MLP.hpp"


MLP::MLP(size_t inputSize, std::unique_ptr<Loss> lossFunc)
	: inputSize(inputSize), currentOutputSize(inputSize), loss(0), lossFunc(std::move(lossFunc))
{ }

void MLP::addLayer(size_t neurons, std::unique_ptr<Activation> f, double weightmin, double weightmax)
{
	size_t layerInputSize = currentOutputSize;
	layers.push_back(std::make_unique<Dense>(layerInputSize, neurons, weightmin, weightmax));
	layers.push_back(std::move(f));
	currentOutputSize = neurons;
}

Matrix MLP::forward(const Matrix& X)
{
	Matrix out = X;
	for (auto& layer : layers)
		out = layer->forward(out);
	return out;
}

void MLP::trainOnce(const Matrix& X, const Matrix& Y, double learningRate)
{
	Matrix y_pred = forward(X);	
	this->loss = lossFunc->forward(y_pred, Y);
	Matrix grad = lossFunc->backward();
	for (auto it = layers.rbegin(); it != layers.rend(); ++it)
		grad = (*it)->backward(grad);
	
	for (auto& layer : layers)
		layer->update(learningRate);
}

void MLP::train(const Matrix& X, const Matrix& Y, size_t epochs, double learningRate)
{
	for (size_t i = 0; i < epochs; ++i)
		trainOnce(X, Y, learningRate);
}

Matrix MLP::predict(const Matrix& X)
{
	return forward(X);
}

double MLP::getLastLoss() const { return loss; }

void MLP::save(const std::string& filename) const
{
	std::ofstream out(filename, std::ios::binary);

	for (auto& layer : layers)
	{
		Dense* d = dynamic_cast<Dense*>(layer.get());
		if (d)
			d->save(out);
	}
}

void MLP::load(const std::string& filename)
{
	std::ifstream in(filename, std::ios::binary);

	for (auto& layer : layers)
	{
		Dense* d = dynamic_cast<Dense*>(layer.get());
		if (d)
			d->load(in);
	}
}
