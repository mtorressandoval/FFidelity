# FastFidelity using few PauliMatrices (work-in-progress)

This project is based on the algorithm introduced in [Flammia-2011](http://dx.doi.org/10.1103/PhysRevLett.106.230501), in companion with the optimization algorithm cpsa [Gidi-2021]().

In the following lines I will describe each block of the code.
## Fidelity using few Pauli Matrices
Let's consider two density states $\rho$ and $\sigma$. Here $\rho$ is a known pure state while $\sigma$ is the a quantum state that has to be reconstruct. The fidelity between these two states is given by
$$F(\rho,\sigma)=\mathrm{tr}(\sigma\rho). $$
For our purposes, it is convenient to expand our states $\rho$ and $\sigma$ in terms of Pauli-operators $W_{k}$ with $k=1,..,d^2$. In this basis, the fidelity takes the form
$$F(\rho,\sigma)=\sum_{k=1}^{d^2}\chi_{\rho}(k)\chi_{\sigma}(k), $$
where $\chi_{\rho}(k)=\mathrm{tr}(\rho W_{k}/\sqrt{d})$. Giving that $P(k)=\chi_{\rho}(k)^2$ is a probability distribution, we can estimate the fidelity $F$, using for example, a MonteCarlo method
$$F(\rho,\sigma)\approx\sum_{\tilde{k}}\frac{\chi_{\sigma}(\tilde{k})}{\chi_{\rho}(\tilde{k})},$$ 
where the values of $\tilde{k}$ are chosen randomly with respect the probability distribution $P(k)$. This algorithm show us that we can estimate the fidelity by just performing few measures on the quantum state $\sigma$.
## Fast Tensor Product

In order to obtain the probability distribution $P(k)$ we need to perform $d^2$ measures on the state $\rho$. For a system of $n_{q}$ qubits, the operators $W_{k}$ will be equal to the tensor product of $n_{q}$ matrices of $2\times2$. For large number of qubits, this operation is highly expensive and the naive algorithm will take too time. We can accelerate the action of the tensor product by using the following relation
$$(A\otimes B)X=AXB^{T}$$
where the matrix $S$, $T$ and $X$ are of dimension $m\times m$, $n\times n$ and $m\times n$ respectively. Using the previous relation we can compute the action of chain of operators acting on a vector recursively. This is implmented via the function FastTensorProd

``` python

def FastTensorProd(A,x):
#Given a list of matrices A = [Ar,...,A1] and a vector x,
#return (Ar o ... o A1)x where o is the Kronecker product
    L=len(x)
    x=x.reshape(L//2, 2)
    for a in range(len(A)-1):
        x=x@A[-a-1].T
    x=x.reshape(2,L//2)
    x=A[0]@x
    return x.flatten()
```


