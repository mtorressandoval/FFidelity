# FastFidelity using few PauliMatrices 

This project is based on the algorithm introduced in [Flammia-2011](http://dx.doi.org/10.1103/PhysRevLett.106.230501), in companion with the optimization algorithm cpsa [Gidi-2021]().

## Fidelity using few Pauli Matrices
Let us considere two density states $\rho$ and $\sigma$. Here $\rho$ is a known pure state while $\sigma$ is the a quantum state that has to be reconstruct. The fidelity bewtween these two states is given by
$$F(\rho,\sigma)=\operatorname{tr}(\sigma\rho)=\sum_k \chi_\rho(k) \chi_\sigma(k) $$