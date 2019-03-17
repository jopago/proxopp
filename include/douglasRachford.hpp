#ifndef __DOUGLAS_RACHFORD
#define __DOUGLAS_RACHFORD

#include "Solver.hpp"
#include "proximals.hpp"

using namespace Eigen; 

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
	~douglasRachfordSolver() 
	{
		delete proxF;
		delete proxG;
	}

	void iterate()
	{
		VectorXf prox_step;

		_x = (*proxF)(_y, 1.0f);
		prox_step = 2*_x - _y;
		_y = _y + (*proxG)(prox_step, 1.0f) - _x;
	}

	float currentObjective() override
	{
		return proxF->f(_x) + proxG->f(_x);
	}

	void setProxF(proxOperator* proxF) 
	{
		this->proxF = proxF;
	}

	void setProxG(proxOperator* proxG)
	{
		this->proxG = proxG;
	}

	void swapProx()
	{
		swap = true;		
	}
protected:
	VectorXf _y;

	proxOperator *proxF, *proxG;
	int swap = false;
};


class basisPursuitSolver : public douglasRachfordSolver 
{
public:
	basisPursuitSolver(int n, int verbose=1, int max_steps=100, std::string name="Basis Pursuit (DR)") : 
	douglasRachfordSolver(n, verbose, max_steps, name)
	{
		proxF = new softThresholdingOperator(); 
	}
	~basisPursuitSolver() {}

	void initParameters(MatrixXf& A, VectorXf& b) override
	{
		_A = A;
		_b = b;

		proxG = new proxLinearEquality(_A,_b); 

		if(swap)
		{
			proxOperator *tmp;
			tmp = proxF;
			proxF = proxG;
			proxG = tmp;
		}
	}

	/*
	float currentObjective() override
	{
		return _x.lpNorm<1>(); 
	} */
private:
	MatrixXf _A;
	VectorXf _b;
};

#endif