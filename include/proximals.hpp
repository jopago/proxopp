#ifndef __PROXIMALS__
#define __PROXIMALS__

#include <Eigen/Dense>

using namespace Eigen; 

class proxOperator
{
public:
	proxOperator() {}
	virtual ~proxOperator() {}

	virtual float f(VectorXf& x) 
	{
		// f(x)
		return 0.0f;
	}

	virtual VectorXf operator()(VectorXf& x, float lambda)
	{
		// prox_{\lambda f}(x)
		return x;
	}
};

//	The proximal operator of the L1 norm
//	soft_thresholding(x,lambda) = sign(x)*max(|x| - lambda, 0) (element-wise)

class softThresholdingOperator : public proxOperator
{
public:
	float f(VectorXf& x) override
	{
		return x.lpNorm<1>();
	}

	VectorXf operator()(VectorXf& x, float lambda) override
	{
		return x.array().sign()*((x.array().abs()-lambda).max(ArrayXf::Zero(x.rows())));
	}
};

//	The proximal operator of the (convex) indicator
//	function of the set {x | Ax = b} is
//	x - At(AtA)^(-1)(Ax - b) 

class proxLinearEquality : public proxOperator
{
public:
	proxLinearEquality(MatrixXf& A, VectorXf& b) : _A(A), _b(b) 
	{
		_Q = A.transpose()*((A*A.transpose()).inverse());
	}

	float f(VectorXf& x) override
	{
		if((_A*x - _b).norm() < tol) 
		{
			return  (_A*x - _b).norm();
		} else {
			return Infinity; 
		}
	}

	VectorXf operator()(VectorXf& x, float lambda) override
	{
		return (x - _Q*(_A*x - _b));
	}
private:
	MatrixXf _A;
	MatrixXf _b;
	MatrixXf _Q; 

	const double tol = 1e-6;
};

#endif