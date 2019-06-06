
# proxopp
> Proximal optimization in C++ with Eigen for convex optimization

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](LICENSE)

![screenshot](https://raw.githubusercontent.com/jopago/proxopp/master/output/screenshot.png)

# Proximal operators

The proximal operator of a convex function is defined as follows:

![](https://latex.codecogs.com/gif.latex?%5Ctext%7Bprox%7D_%7Bf%7D%5Cleft%28x%20%5Cright%20%29%20%3D%20%5Carg%5Cmin_%7By%20%5Cin%20%5Cmathbb%7BR%7D%5En%7D%20f%28y%29%20&plus;%20%5Cfrac12%20%5Cleft%5CVert%20y-x%20%5Cright%5CVert_2%5E2)

It plays an important role in several optimization algorithms that can be used to solve Basis Pursuit and LASSO problems 
that appear frequently in signal processing (compressed sensing for instance), quantitative finance and machine learning settings In this repo I have currently implemented:

- Douglas Rachford (or proximal splitting) for basis pursuit
- Proximal gradient descent for LASSO (ISTA)
- FISTA
- ADMM for portfolio optimization

![convergence](https://raw.githubusercontent.com/jopago/proxopp/master/output/convergence.png)

# Todo

- Constraints proxial operator (Dykstra's algorithm)
- Testing 
- ~~ADMM~~

# References 

[Proximal Algorithms lecture notes](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf)

[FISTA](https://people.rennes.inria.fr/Cedric.Herzet/Cedric.Herzet/Sparse_Seminar/Entrees/2012/11/12_A_Fast_Iterative_Shrinkage-Thresholding_Algorithmfor_Linear_Inverse_Problems_(A._Beck,_M._Teboulle)_files/Breck_2009.pdf)

