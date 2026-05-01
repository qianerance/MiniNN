#include <iostream>
#include "Matrix.hpp"
#include "MLP.hpp"
#include "Elementwise.hpp"

void matrixprinter(const Matrix& m);

int main()
{
    // XOR problem solution
    Matrix X = Matrix(4, 2);
    X(0, 0) = 0; X(0, 1) = 0;
    X(1, 0) = 0; X(1, 1) = 1;
    X(2, 0) = 1; X(2, 1) = 0;
    X(3, 0) = 1; X(3, 1) = 1;

    Matrix Y = Matrix(4, 1);
    Y(0, 0) = 0;
    Y(1, 0) = 1;
    Y(2, 0) = 1;
    Y(3, 0) = 0;

    MLP mlp(2, std::make_unique<MSE>());
    mlp.addLayer(2, std::make_unique<Elementwise>(Elementwise::Sigmoid()), -2, 2);
    mlp.addLayer(1, std::make_unique<Elementwise>(Elementwise::Sigmoid()), -2, 2);

    std::cout << "XOR MLP\n";

    for (int epoch = 0; epoch < 10000; ++epoch)
        mlp.trainOnce(X, Y, 0.1);

    Matrix X_test(4, 2);
    X_test(0, 0) = 1; X_test(0, 1) = 1; // 0
    X_test(1, 0) = 0; X_test(1, 1) = 0; // 0
    X_test(2, 0) = 1; X_test(2, 1) = 0; // 1
    X_test(3, 0) = 0; X_test(3, 1) = 1; // 1
    
    Matrix Y_pred = mlp.predict(X_test);

    std::cout << "Prediction for:\n";
    matrixprinter(X_test);
    matrixprinter(Y_pred);
    return 0;
}

void matrixprinter(const Matrix& m)
{
	for (int i = 0; i < m.getRows(); ++i)
	{
		for (int j = 0; j < m.getCols(); ++j)
			std::cout << m(i, j) << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;
}