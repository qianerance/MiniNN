#pragma once
#include "Matrix.hpp"
#include "Loss.hpp"

class MSE final : public Loss
{
public:
    double forward(const Matrix& y_pred, const Matrix& y_true) override;
    Matrix backward() override;
};