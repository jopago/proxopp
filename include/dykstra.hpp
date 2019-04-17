#ifndef __DYSKTRA__
#define __DYSKTRA__

#include <Eigen/Dense>
#include "proximals.hpp"

namespace proxopp {
	
//  Dykstra's algorithm to compute projections on intersection of convex sets
//  P. L. Combettes and J.-C. Pesquet, "Proximal splitting methods in signal processing"

class DykstraProjection 
{
	public:
		DykstraProjection(proximalOperator *pF, proximalOperator *pG, 
				int n_iterations=10) :
			pF(pF), pG(pG), n_iterations(n_iterations) {}
		// Provided the convex F and G are "simple", convergence 
		// should be fast. 

		~DykstraProjection() {}

		Eigen::VectorXf project(Eigen::VectorXf& x0)
		{
			Eigen::VectorXf x,y,p,q; 

			x = x0;
			p = q = Eigen::VectorXf::Zeros(x0.rows()); 	

			for(int k=0; k < n_iterations;k++) 
			{
				y = (*pF)(x+p); 
				p = x + p - y;
				x = (*pG)(y+q);
				q = y + q - x; 
			}

			return x;
		}
	private:
		proximalOperator *pF, *pG; 
		int n_itierations;
};
} // namespace proxopp 

#endif 
