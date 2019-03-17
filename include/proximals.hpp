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
			return 0;
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

//	Proximal operator of the squared L2 norm 

class proximalL2Square : public proxOperator
{
public:
	float f(VectorXf& x) override
	{
		return x.squaredNorm();
	}

	VectorXf operator()(VectorXf& x, float lambda) override
	{
		return x/(1.0f+lambda);
	}
}; 

//	Proximal operator of the indicator function of 
//	the set {x | max abs(x_i) <= lambda} (Linf ball)

class proximalLinfBall : public proxOperator
{
public:
	proximalLinfBall(float lambda) : radius(lambda) {}

	float f(VectorXf& x) override
	{
		return (x.lpNorm<Infinity>() < radius) ? 0 : Infinity;
	}

	VectorXf operator()(VectorXf& x, float lambda = 0.0f) override
	{
		for(int i = 0; i < x.rows(); i++)
		{
			if(x[i] > radius) 	x[i] = radius;
			if(x[i] < -radius) 	x[i] = -radius; 
		}
	}
private:
	float radius;
};

//	Fast Projection onto the L1 Ball using sorting
//	Using algorithm from
//	Held, M., Wolfe, P., Crowder, H.: Validation of subgradient optimization.
//	Mathematical Programming6, 62–88 (1974)

//	Alternatively, see Condat, L. 
// 	Fast Projection onto the Simplex and the L1-Ball
//	https://www.gipsa-lab.grenoble-inp.fr/~laurent.condat/publis/Condat_simplexproj.pdf
//	It corresponds to Algorithm 1

class proximalL1Ball : public proxOperator
{
public:
	proximalL1Ball(float lambda) : radius(lambda) {}

	float f(VectorXf& x) 
	{
		return (x.lpNorm<1>() < radius) ? 0 : Infinity;
	}

	VectorXf operator()(VectorXf& x, float lambda = 0.0f)
	{
		if(x.lpNorm<1>() < radius) return x;
		
		ArrayXf u = ArrayXf(x.array().abs());
		std::sort(u.data(), u.data()+u.size(), std::greater<float>());	
		
		int k=1, K;
		float tau = 0.0f;

		for(k=1;k<=x.rows();k++)
		{
			float mean = 0.0f;

			for(int i = 0; i < k; i++)
			{
				mean += u[i];
			} 
			mean = (mean-radius)/k; 

			if(mean < u[k-1]) 
			{
				K 	= k-1;
				tau = mean;
			} 
		}
		std::cout << tau << std::endl; 

		VectorXf xtau = x.array().abs()-tau*ArrayXf::Ones(x.rows());
		return x.array().sign()*(xtau).array().max(VectorXf::Zero(x.rows()).array());
	}
private:
	float radius;
};

//	Projection on the intersection of L1 and Linf balls
//	with binary search 

class proximalL1LinfBall : public proxOperator
{
public:
	proximalL1LinfBall(float radius_l1, float radius_linf) : r_l1(radius_l1),
	r_linf(radius_linf)
	{}

	float f(VectorXf& x)
	{
		if(x.lpNorm<1>()<r_l1 && x.lpNorm<Infinity>() < r_linf) return 0.0f;
		return Infinity; 
	}

	VectorXf operator()(VectorXf& x)
	{
		int n = x.rows();
		proximalLinfBall projLinf(r_linf);
		VectorXf y = projLinf(x);

		if(y.lpNorm<1>() < r_l1) return y;

		const double eps = 1e-6;
		const int itermax = 50; 
		int k = 0;
		float nu_l = 0.0f;
		float nu_r = x.array().abs().maxCoeff();
		ArrayXf z1,z2;

		while((k < itermax) && (nu_l - nu_r > eps))
		{
			float nu_m = 0.5*(nu_l+nu_r);

			z1 = y.array().abs()-nu_m*ArrayXf::Ones(n);
			z1 = z1.max(ArrayXf::Zero(n));
			z2 = z1.min(r_linf*ArrayXf::Ones(n));

			if(z2.lpNorm<1>() < r_l1)
			{
				nu_r = nu_m;
			} else {
				nu_l = nu_m;
			}

			k++;
		}

		return z2*x.array().sign();
	}
private:
	float r_l1; // radius of L1 ball
	float r_linf;  // radius of Linf ball (max coeff of abs values)
};

#endif