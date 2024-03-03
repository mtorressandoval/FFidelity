# FastFidelity using few PauliMatrices 

This project is based on the algorithm introduced in [Flammia-2011](http://dx.doi.org/10.1103/PhysRevLett.106.230501), in companion with the optimization algorithm cpsa [Gidi-2021]().

In the following lines I will describe each block of the code.
## Fidelity using few Pauli Matrices
Let's consider two density states $\rho$ and $\sigma$. Here $\rho$ is a known pure state while $\sigma$ is the a quantum state that has to be reconstruct. The fidelity between these two states is given by
$$F(\rho,\sigma)=\mathrm{tr}(\sigma\rho). $$
Is it convenient to expand our states $\rho$ and $\sigma$ in term of Pauli-operators $W_{k}$ with $k=1,..,d^2$. In this basis the Fidelity takes the form
$$F(\rho,\sigma)=\sum_{k=1}^{d^2}\chi_{\rho}(k)\chi_{\sigma}(k) $$.
where $\chi_{\rho}(k)=\mathrm{tr}(\rho W_{k}/\sqrt{d})$. From this expresion we can construct an estimator of the Fidelity. Is it direct to see that $ P(k)=\chi_{\rho}(k)^2 $ corresponds to a probability distribution, and therefore, we can estimate the fidelity $F$ using a MonteCarlo simulation, this is
$$F(\rho,\sigma)=\sum_{\tilde{k}}\frac{\chi_{\sigma}}(k){\chi_{\rho}(k)}$$ 
where the values of $\tilde{k}$ are chosen randomly with respect the probability distribution $P(k)$.
