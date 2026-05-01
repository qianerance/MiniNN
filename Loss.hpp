#pragma once
#include "Matrix.hpp"

class Loss 
{
protected:
    Matrix y_pred;
    Matrix y_true;
public:
    virtual double forward(const Matrix&, const Matrix&) = 0;
    virtual Matrix backward() = 0;
};