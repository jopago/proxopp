#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

#include "proximalGradient.hpp"

int main()
{
	const int N = 200;

	VectorXf b = VectorXf::Random(N);
	VectorXf x = VectorXf::Random(N);
	MatrixXf A = MatrixXf::Random(N,N);
	
	proximalGradientSolver solver(N, 1e-3, 1e-3, 2, 150);
	fistaSolver fista(N, 1e-3, 0.0f, 2, 150);

	VectorXf ista_x 	= solver.solve(A,b);
	VectorXf fista_x 	= fista.solve(A,b);

	std::cout << (ista_x - fista_x).squaredNorm() << std::endl; 
}