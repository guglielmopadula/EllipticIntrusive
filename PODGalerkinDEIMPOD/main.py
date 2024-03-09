from dolfin import *
from rbnics import *
from time import time
import numpy as np
@DEIM("online",basis_generation="POD")
@ExactParametrizedFunctions("offline")
class Gaussian(EllipticCoerciveProblem):

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        EllipticCoerciveProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        self.u = TrialFunction(V)
        self.v = TestFunction(V)
        self.dx = Measure("dx")(subdomain_data=subdomains)
        self.f = ParametrizedExpression(
            self, "exp(- 2 * pow(x[0] - mu[0], 2) - 2 * pow(x[1] - mu[1], 2))", mu=(0., 0.),
            element=V.ufl_element())
        # note that we cannot use self.mu in the initialization of self.f, because self.mu has not been initialized yet

    # Return custom problem name
    def name(self):
        return "Elliptic"

    # Return the alpha_lower bound.
    def get_stability_factor_lower_bound(self):
        return 1.

    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        if term == "a":
            return (1.,)
        elif term == "f":
            return (1.,)
        else:
            raise ValueError("Invalid term for compute_theta().")

    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "a":
            u = self.u
            a0 = inner(grad(u), grad(v)) * dx
            return (a0,)
        elif term == "f":
            f = self.f
            f0 = f * v * dx
            return (f0,)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, Constant(0.0), self.boundaries, 1),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 2),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 3)]
            return (bc0,)
        elif term == "inner_product":
            u = self.u
            x0 = inner(grad(u), grad(v)) * dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")

mesh = Mesh("../data/gaussian.xml")
subdomains = MeshFunction("size_t", mesh, "../data/gaussian_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "../data/gaussian_facet_region.xml")

V = FunctionSpace(mesh, "Lagrange", 1)

problem = Gaussian(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(-1.0, 1.0), (-1.0, 1.0)]
problem.set_mu_range(mu_range)

reduction_method = PODGalerkin(problem)
reduction_method.set_Nmax(20,DEIM=21)
reduction_method.set_tolerance(1e-8,DEIM=1e-4)

reduction_method.initialize_training_set(50, DEIM=60)
reduced_problem = reduction_method.offline()

online_mu = (0.3, -1.0)
reduced_problem.set_mu(online_mu)
reduced_solution = reduced_problem.solve()

training_param=np.load('../data/train_params.npy')
testing_param=np.load('../data/test_params.npy')


start=time()
for i in range(50):
    reduced_problem.set_mu(tuple(training_param[i]))
    reduced_problem.solve()
    reduced_problem.export_solution(filename="online_solution_train_{}".format(i))

for i in range(50):
    reduced_problem.set_mu(tuple(testing_param[i]))
    reduced_problem.solve()
    reduced_problem.export_solution(filename="online_solution_test_{}".format(i))

end=time()

np.save("time.npy",end-start)