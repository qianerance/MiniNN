#include <cmath>
#include <stdexcept>

#include "MSE.hpp"
#include "Matrix.hpp"

double MSE::forward(const Matrix& y_pred, const Matrix& y_true)
{
    if(y_pred.getCols () != y_true.getCols() || y_pred.getRows() != y_true.getRows())
		throw std::invalid_argument("y_pred and y_true must have the same dimensions.");


    double sum = 0.0;
    double diff = 0.0;
    for (int i = 0; i < y_pred.getRows(); ++i)
        for (int j = 0; j < y_pred.getCols(); ++j)
        {
            diff = y_pred(i, j) - y_true(i, j);
            sum += diff * diff;
        }   

    this->y_pred = y_pred;
	this->y_true = y_true;

    return sum / (y_pred.getCols() * y_pred.getRows());
}

Matrix MSE::backward()
{
    Matrix dY = (y_pred - y_true) * (2.0 / (y_pred.getRows() * y_pred.getCols()));
    return dY;
}