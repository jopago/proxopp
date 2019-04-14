#ifndef __ADMM_SOLVER
#define __ADMM_SOLVER

#include "Solver.hpp"
#include "proximals.hpp"

// ADMM Algorithm to solve  arg min f(x) + g(z)
//                          s.t Ax + z = 0

//  https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf

namespace proxopp 
{
class admmSolver : public Solver
{
public:
    admmSolver(int n) : Solver(n)
    {
        mu = 1e6;
        rho = 1.0f;

        dual_tol = primal_tol = 1e-8;
        dual_err = primal_err = 1.0f;
        multiplier = 2.0f;

        z = _x;
        u = z;
        z_old = z;
        A = Eigen::MatrixXf::Identity(n,n);

        max_steps = 300;
    }
    ~admmSolver() {}

    Eigen::VectorXf solve()
    {
        step = 0;

        while((step < max_steps) && (primal_err > primal_tol) || (dual_err > dual_tol))
        {
            x_step(); // updates _x
            z_step(); // updates z

            Eigen::VectorXf r = A*_x + z;
            Eigen::VectorXf s = rho*A*(z - z_old);
            u += r;

            primal_err  = r.squaredNorm();
            dual_err    = s.squaredNorm();

            if(primal_err > mu*dual_err)
            {
                rho *= multiplier;
                u /= multiplier;
                rho_callback();
            } else if(dual_err > mu*primal_err)
            {
                rho /= multiplier;
                u *= multiplier;
                rho_callback();
            }

            z_old = z;
            step++;
            callback();
        }

        return _x;
    }
    //  Called when rho changed
    virtual void rho_callback() {}

    virtual void x_step() {}

    virtual void z_step() {}
protected:
    Eigen::VectorXf z, z_old;
    Eigen::VectorXf u;
    Eigen::MatrixXf A;

    float mu, rho;
    float dual_tol, primal_tol;
    float dual_err, primal_err;
    float multiplier;
};

//  Application of ADMM to solve a portfolio optimization Problem
//  arg min - <mu,x> + lambda*||x-x0||^_2
//  s.t <x,Sigma*x> <= vol_target^2

//  Ridge regularization makes the problem well-defined

class ptfVolConstrainedL2 : public admmSolver
{
public:
    ptfVolConstrainedL2(Eigen::VectorXf returns, Eigen::MatrixXf covariance, Eigen::VectorXf x0, float vol_target = 0.03f,
    float lambda = 0.01f) : admmSolver(returns.rows()), vol_target(vol_target),
    lambda(lambda), covariance(covariance), returns(returns), x0(x0)
    {
        projL2 = new proximalL2Ball(vol_target);
        xStepSystem = lambda*Eigen::MatrixXf::Identity(returns.rows(), returns.rows()) + rho*covariance;

        //  The quadratic form constraint <x,Sigma*x> <= vol_target^2
        //  Can be written as ||Lx|| <= vol_target^2 with L the
        //  Cholesky factor of Sigma, thus allowing a faster and easier ADMM

        L = covariance.llt().matrixL().transpose();
        A = -L;
    }
    ~ptfVolConstrainedL2()
    {
        delete projL2;
    }

    void rho_callback()
    {
        // Update system matrix if rho has changed
        xStepSystem = lambda*Eigen::MatrixXf::Identity(n,n) + rho*covariance;
    }

    void x_step()
    {
        Eigen::VectorXf b = returns + lambda*x0 + rho*L.transpose()*(z+u);
        _x = xStepSystem.llt().solve(b);
    }

    void z_step()
    {
        Eigen::VectorXf w = L*_x - u;
        z = (*projL2)(w, 0);
    }

    float currentObjective()
    {
        // Regularized Markowtiz objective
        Eigen::VectorXf Lx = L*_x;
        return -returns.dot(_x) + lambda*(_x-x0).squaredNorm() + projL2->f(Lx);
    }
private:
    proxOperator *projL2;

    float lambda, vol_target;
    Eigen::VectorXf returns, x0;
    Eigen::MatrixXf covariance, L;
    Eigen::MatrixXf xStepSystem;
};
} // namespace proxopp 
#endif
