#ifndef __PROXIMALS__
#define __PROXIMALS__

#include <Eigen/Dense>

using namespace Eigen; 

// proximal operator of lambda * ||.||_1

VectorXf softThresholding(VectorXf x, float lambda)
{
	return x.array().sign()*((x.array().abs()-lambda).max(ArrayXf::Zero(x.rows())));
}

#endif
