#ifndef __SOLVER__
#define __SOLVER__

#include <iostream>
#include <fstream>
#include <Eigen/Dense>

using namespace Eigen;

class Solver
{
public:
	Solver(int n, int verbose=1, int max_steps=100, std::string name="solver") : step(0), verbose(verbose), 
			max_steps(max_steps), name(name)
	{
		this->n = n;
		_x = VectorXf::Zero(n);

		if(verbose > 1)
		{
			output.open("output/" + name + ".csv");

			if(!output.is_open())
			{
				std::cout << "(" << name << ")" << " cannot open output file\n";
			} else 
			{
				output << "step,objective\n";
			}
		}

		stop = 0;
	}

	virtual ~Solver() 
	{
		if(output.is_open()) output.close();
	}

	virtual void callback() 
	{
		if(verbose > 0) 
		{
			std::cout << "(" << name << ") " << step << "/" << max_steps << " objective: " << currentObjective() << " \n";

			if(verbose > 1 && output.is_open())
			{
				output << step << "," << currentObjective() << std::endl;
			}
		}
	}

	virtual void iterate() {}

	virtual float currentObjective() 
	{
		return 0.0f; 
	}

	virtual void initParameters(MatrixXf& A, VectorXf& b) {}

	virtual VectorXf solve(MatrixXf& A, VectorXf& b) 
	{
		initParameters(A,b);

		for(step = 0; step < max_steps; step++)
		{
			iterate();
			callback();
			if(stop) break; 
		}

		return _x;
	}

	void setMaxSteps(int max_steps)
	{
		this->max_steps = max_steps;
	}

	void setVerbose(int verbose)
	{
		this->verbose = verbose;
	}

	int getMaxSteps() { return max_steps; }

	std::string getName() { return name; }
	
protected:
	int verbose;
	int n; 
	int step; 
	int stop; 
	int max_steps; 
	std::string name;
	std::ofstream output;

	VectorXf _x; // current state 
};

#endif