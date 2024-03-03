# FastFidelity using few PauliMatrices 

This project is based on the algorithm introduced in [Flammia-2011](http://dx.doi.org/10.1103/PhysRevLett.106.230501), in companion with the optimization algorithm cpsa [Gidi-2021]().

In the following lines I will describe the building block of the code.
## Fidelity using few Pauli Matrices
Let us considere two density states $\rho$ and $\sigma$. Here $\rho$ is a known pure state while $\sigma$ is the a quantum state that has to be reconstruct. The fidelity bewtween these two states is given by
$$F(\rho,\sigma)=\mathrm{tr}(\sigma\rho) $$
we can expanse our states $\rho$ and $\sigma$ in term of Pauli-operators $W_{k}$ with $k=1,..,d^2$. In this basis the Fidelity takes the form
$$F(\rho,\sigma)=\sum_{k=1}^{d^2}\chi_{\rho}(k)\chi_{\sigma} $$.
Is it possible to construct an estimator of this quantity. Is it direct to see that we can construct the following probability distribution 
$$ P(k)=\chi_{\rho}(k)^2 $$
and therefore, we can estimate the fidelity $F$ using a MonteCarlo simulation
$$F(\rho,\sigma)=\sum_{\tilde{k}}\frac{\chi_{\sigma}}{\chi_{\rho}(k)}$$ 
where the values of $\tilde{k}$ are chosen with respect the probability distribution $P(k)$.
