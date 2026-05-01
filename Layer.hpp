#pragma once
#include "Matrix.hpp"
class Layer
{
public:
	virtual Matrix forward(const Matrix&) = 0;
	virtual Matrix backward(const Matrix&) = 0;
	virtual ~Layer() = default;
	virtual void update(double lr) {};
};