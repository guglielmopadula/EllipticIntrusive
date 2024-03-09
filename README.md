A repo of benchmark tests using RBnics of an [elliptic pde with a non affine parametric dependency](https://github.com/RBniCS/RBniCS/blob/master/tutorials/05_gaussian/tutorial_gaussian_exact.ipynb).

|Method                                     |Train error|Test Error|Time   |
|-------------------------------------------|-----------|----------|-------|
|PODGalerkin(Nmax=20)                       |3.1e-03    |5.5e-03   |9.2e-01|
|PODGalerkinDEIMPOD(Nmax=20,DEIM=21)        |7.4e-03    |4.7e-01   |4.6e-01|
|PODGalerkinDEIMGreedy(Nmax=20,DEIM=21)     |3.3e-03    |3.4e-03   |3.6e-01|
|PODGalerkinEIMPOD(Nmax=20,EIM=21)          |9.2e-03    |8.1e-03   |5.1e-01|      
|PODGalerkinEIMGreedy(Nmax=20,EIM=21)       |3.6e-03    |4.6e-03   |6.3e-01| 
|ReducedBasis(Nmax=20)                      |4.1e-04    |7.8e-04   |5.6e-01| 
|ReducedBasisDEIMPOD(Nmax=20,DEIM=21)       |6.2e-03    |7.5e-03   |4.8e-01|    
|ReducedBasisDEIMGreedy(Nmax=20,DEIM=21)    |6.0e-03    |4.6e-03   |6.1e-01|
|ReducedBasisEIMPOD(Nmax=20,EIM=21)         |1.3e-02    |1.3e-02   |4.7e-01|      
|ReducedBasisEIMGreedy(Nmax=20,EIM=21)      |3.8e-03    |3.8e-03   |4.7e-01|       
