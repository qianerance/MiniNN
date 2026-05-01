#include <stdexcept>
#include <cmath>
#include "CrossEntropy.hpp"

double CrossEntropy::forward(const Matrix& y_pred, const Matrix& y_true)
{
    if (y_pred.getRows() != y_true.getRows() || y_pred.getCols() != y_true.getCols())
		throw std::invalid_argument("y_pred and y_true must have the same dimensions.");

    double sum = 0.0;
	this->y_pred = y_pred;
	this->y_true = y_true;

	for (int i = 0; i < y_pred.getRows(); ++i)
        for (int j = 0; j < y_pred.getCols(); ++j)
            sum += y_true(i, j) * std::log(y_pred(i, j) + 1e-15); // 1e-15 to prevent log(0) = ?-inf
    return -sum / y_pred.getRows();
}

Matrix CrossEntropy::backward()
{
    return (y_pred - y_true) / y_pred.getRows();
}
