#ifndef __DOUGLAS_RACHFORD
#define __DOUGLAS_RACHFORD

#include "Solver.hpp"
#include "proximals.hpp"

using namespace Eigen; 

class douglasRachfordSolver : public Solver 
{
public:
	douglasRachfordSolver(int n, int verbose=1, int max_steps=100, std::string name="Douglas Rachford") : Solver(n, verbose, max_steps, name)
	{
		_y = VectorXf::Zero(n);
	}
	~douglasRachfordSolver() {}

	void initParameters(MatrixXf& A, VectorXf& b) override
	{
		_Q = A.transpose()*((A*A.transpose()).inverse());
		_b = b;
		_A = A;
	}

	void iterate() override 
	{
		_x = proximal(_y);
		_y = _y + softThresholding(2*_x - _y, 1.0f);
	}

	float currentObjective() override
	{
		return _x.lpNorm<1>(); // L1 norm 
	}
private:
	VectorXf proximal(VectorXf& x)
	{
		// Proximal operator of indicator function of {Ax = b} 
		return (x - _Q*(_A*x - _b));
	}

	MatrixXf _Q;
	MatrixXf _A;
	VectorXf _b;

	VectorXf _y;
};

#endif