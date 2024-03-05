import numpy as np
from itertools import product
from qiskit_aer.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
import random as rd


class Mean_Direct_Fidelity:

    def __init__(self, 
                NQ) -> None:
        
        self.NQ    = NQ
        self.d     = 2**NQ 

        sigma0 = np.array([[1, 0], [0, 1]])
        sigma1 = np.array([[0, 1], [1, 0]])
        sigma2 = np.array([[0, -1j], [1j, 0]])
        sigma3 = np.array([[1, 0], [0, -1]])
        Sigmamu=[sigma0, sigma1, sigma2, sigma3]
        self.Medida={}
        self.Sigmamu = Sigmamu 

        SigmaL=['I','X','Y','Z']
        ChainSigmaL=list(product(SigmaL, repeat=NQ))
        WW=[]
        for a in range(len(ChainSigmaL)):
            WW.append(''.join(ChainSigmaL[a])) 

        self.WW = WW 
#----------------------------------------------------------------
    def WWPauli(self,j):
        return SparsePauliOp.from_list([(self.WW[j], 1)])    
#----------------------------------------------------------------    
    def NormalizeMeasure(self,j, 
                            QuantumState,
                            estimator = Estimator() ,
                            shots = 1000  ):#Estimator is a class
        job = estimator.run(QuantumState, 
                            self.WWPauli(j), 
                            run_options = { 'shots' : shots } )
        
        return (1/np.sqrt(self.d))*(job.result().values[0])
#----------------------------------------------------------------    
    def FastTensorProd(self,A,x):
#Given a list of matrices A = [A1,...,ANQ] and a vector x,
#return (A1 o ... o ANQ)x where o is the Kronecker product
        L=len(x)
        x=x.reshape(L//2, 2)
        for a in range(len(A)-1):
            x=x@A[-a-1].T
        x=x.reshape(2,L//2)
        x=A[0]@x
        return x.flatten()    
#-----------------------------------------------------------------    
    def ChiRHO(self,RHO0):
        RHO0 = np.array(RHO0)
        ChiRho = []
        for A in product(self.Sigmamu, repeat=self.NQ):
            ChiRho.append( (1/np.sqrt(self.d)) 
                               * np.dot(RHO0.conjugate(), 
                                        self.FastTensorProd(A,RHO0)))
        return np.array(ChiRho)
#-----------------------------------------------------------------    
    def MeanFidelity(self, 
                        Nrepetitions, 
                        Npoints, 
                        RHO0, 
                        OMEGA,
                        estimator = Estimator(),
                        shots     = 1000 ):
        """
        Nrepetitions : montecarlo size for mean fidelity
        Npoints      : montecarlo size for direct fidelity estimation 
        Rho0         : pure state as vector
        Omega        : mixed state as QuantumCircuit
        """
        Chirho = self.ChiRHO(RHO0)
        totalsum = []
        

        for i in range(Nrepetitions):
            kreduce = rd.choices(range( self.d**2), 
                                weights=(Chirho.real)**2, k=Npoints)            
            sum = 0
            for j in set(kreduce):
                if j not in self.Medida:
                    self.Medida[j] = self.NormalizeMeasure(j, OMEGA, estimator, shots)
                repetitions=kreduce.count(j)
                sum += (1 / len(kreduce)) * (self.Medida[j]/ Chirho[j])*repetitions
            totalsum.append(sum)  
        totalsum = np.array(totalsum).real
        return np.sum(totalsum) / len(totalsum)