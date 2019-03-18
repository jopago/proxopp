#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

#include "admmSolver.hpp"

int main(int argc, char **argv)
{
    VectorXf mu = VectorXf(4);
    const float regularization = 0.1f;
    const float volatility_target = 0.05; // 3% max vol

    // Vector of asset expected returns
    mu << 0.1, 0.3, -0.5, 0.6;

    MatrixXf Sigma(4,4);
    // Sample covariance matrix (positive-definite)
    Sigma << 4.2,   0.3,    0.1,    1.25,
             0.3,   5.95,   0.56,   1.56,
             0.1,   0.56,   4.45,   0.25,
             1.25,  1.56,   0.25,   4.56;

    VectorXf x0 = VectorXf::Zero(4);

    // Regularized markowitz optimization

    ptfVolConstrainedL2 optim(mu, Sigma, x0, volatility_target, regularization);
    optim.setVerbose(1);

    VectorXf ptf = optim.solve();
    std::cout << "Solution: \n";
    std::cout << ptf << std::endl;
    std::cout << "(ex-ante) Volatility :" << ptf.dot(Sigma*ptf) << std::endl;
    return 0;
}
