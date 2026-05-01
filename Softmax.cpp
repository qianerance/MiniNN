#include <cmath>
#include "Softmax.hpp"

Softmax::Softmax()
{
}

Matrix Softmax::forward(const Matrix& X)
{
	Matrix P(X.getRows(), X.getCols());

	double maxVal, denominatorSum;
	for (int i = 0; i < X.getRows(); ++i)
	{
		maxVal = X(i, 0);
		for (int j = 1; j < X.getCols(); ++j)
			if (X(i, j) > maxVal)
				maxVal = X(i, j);

		denominatorSum = 0.0;
		for (int k = 0; k < X.getCols(); ++k)
		{
			denominatorSum += std::exp(X(i, k) - maxVal);
		}
		
		for (int j = 0; j < X.getCols(); ++j)
			P(i, j) = std::exp(X(i, j) - maxVal) / denominatorSum;
	}
	lastOutput = P;
	return P;
}

Matrix Softmax::backward(const Matrix& dA)
{
	return dA;
}
