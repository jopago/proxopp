#ifndef __PROXIMAL_GRADIENT
#define __PROXIMAL_GRADIENT

#include "Solver.hpp"
#include "proximals.hpp"

using namespace Eigen;

// Simple proximal gradient descent without backtracking (for now)

class proximalGradientSolver : public Solver 
{
public:
	proximalGradientSolver(int n, float gamma = 0.1f, 
		float step_size = 0.01f, int verbose=1, int max_steps=100, 
		std::string name="ISTA") : Solver(n, verbose, max_steps, name), gamma(gamma), step_size(step_size)
	{

	}
	~proximalGradientSolver() {}

	void initParameters(MatrixXf& A, VectorXf& b) override
	{
		_A = A; 
		_b = b;
	}

	void iterate() override
	{	
		_x = softThresholding(_x - step_size*gradient(_x), gamma);
	}

	float currentObjective()
	{
		return 0.5f*(_A*_x - _b).squaredNorm() + gamma*_x.lpNorm<1>();
	}

protected:
	VectorXf gradient(VectorXf x)
	{
		return _A.transpose()*(_A*x - _b);
	}

	MatrixXf _A;
	VectorXf _b;

	float step_size; 
	float gamma; 
};

/* 	FISTA
	A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems 
	Beck. A, and Teboulle, M.
	SIAM Journal of Imaging Sciences, 2(1), pp. 183-202 

	This algorithm has faster convergence rate than ISTA thanks to an
	acceleration à la Nesterov */

class fistaSolver : public proximalGradientSolver 
{
public:
	fistaSolver(int n, float gamma = 0.1f, float momentum = 0.0f, 
		int verbose=1, int max_steps=100, 
		std::string name="FISTA") : proximalGradientSolver(n, gamma, step_size, verbose, max_steps,
		name),
	momentum(momentum)
	{
		_z 	= VectorXf::Zero(n);
		_Ab = VectorXf::Zero(n);
		_x_new = _x;
	}
	~fistaSolver() {}

	void initParameters(MatrixXf& A, VectorXf& b) override
	{
		_A = A; 
		_b = b;
		_Ab = _A.transpose()*_b;

		_Q = _A.transpose()*A;
		
		float spectralRadius = _Q.operatorNorm(); 
		step_size = 1.0 / ( spectralRadius); 
	}

	void iterate() override
	{	
		_x_new = softThresholding(_z - step_size*(_Q*_z - _Ab), gamma * step_size);
		momentum_new =  0.5f*(1.0f + sqrt(1.0f + 4.0f*momentum*momentum)); 
		_z = _x_new + ((momentum-1) / momentum_new) * (_x_new - _x);

		_x = _x_new;
		momentum = momentum_new;
	}

private:
	MatrixXf _Q;
	VectorXf _Ab, _z, _x_new; 

	float momentum, momentum_new; // Nesterov momentum
};

#endif 
