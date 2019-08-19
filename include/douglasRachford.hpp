#ifndef __DOUGLAS_RACHFORD
#define __DOUGLAS_RACHFORD

#include "Solver.hpp"
#include "proximals.hpp"
#include <limits>
#include <memory>

namespace proxopp {
//	This class solves min f(x) + g(x) where f,g are convex
//	and proximal, the proximal operator is given in a
//	class that can evaluate prox_{\lambda f}(x) and f(x)

// 	Note that it is not symmetric in f and g
//	By swapping f and g you get two different algorithms
//	Depending on the context you might prefer one or the other 

class douglasRachfordSolver : public Solver
{
public:
	douglasRachfordSolver(int n, int verbose=1, int max_steps=100,std::string name="Douglas Rachford") :
	Solver(n, verbose, max_steps, name)
	{
		_y = _x;
	}
	
	~douglasRachfordSolver() {}

	void iterate()
	{
		Eigen::VectorXf prox_step;

		_x = (*proxF)(_y, 1.0f);
		prox_step = 2*_x - _y;
		_y = _y + (*proxG)(prox_step, 1.0f) - _x;
	}

	float currentObjective() override
	{
		return proxF->f(_x) + proxG->f(_x);
	}

	void setProxF(const std::shared_ptr<proxOperator>& proxF) 
	{
		this->proxF = proxF;
	}

	void setProxG(const std::shared_ptr<proxOperator>& proxG)
	{
		this->proxG = proxG;
	}

	void swapProx()
	{
		swap = true;		
	}
protected:
	Eigen::VectorXf _y;

	std::shared_ptr<proxOperator> proxF, proxG;
	int swap = false;
};


class basisPursuitSolver : public douglasRachfordSolver 
{
public:
	basisPursuitSolver(int n, int verbose=1, int max_steps=100, std::string name="Basis Pursuit (DR)") : 
	douglasRachfordSolver(n, verbose, max_steps, name)
	{
		proxF = std::make_shared<softThresholdingOperator>(); 
	}
	~basisPursuitSolver() {}

	void initParameters(Eigen::MatrixXf& A, Eigen::VectorXf& b) override
	{
		_A = A;
		_b = b;

		proxG = std::make_shared<proxLinearEquality>(_A, _b); 
		
		if(swap)
		{
			proxF.swap(proxG);
		}
	}

	float currentObjective() override
	{
		return _x.lpNorm<1>();
	} 
private:
	Eigen::MatrixXf _A;
	Eigen::VectorXf _b;
};
}

#endif
