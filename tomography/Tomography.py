import numpy as np
from scipy.optimize import minimize
from .FastFidelity import Mean_Direct_Fidelity
from .cspsa import CSPSA

def NearSparseTomography( phi, MDF ):
    """
    phi : initial condition for the search
    MDF : FastFidelity class 
    """
    phi = phi / np.linalg.norm(phi)
    phi = np.array( [ phi.real, phi.imag ] )

    if isinstance( MDF, Mean_Direct_Fidelity ):
    
        def CostF( psi, MDF ): 
            psi = psi.reshape(2,-1)
            psi = psi[0] + 1j*psi[1]
            M_d = MDF.Measures 
            M   = MDF.Chi(psi,truncation=False) 
            f   = 0
            for j, _ in enumerate( M ):
                if j in M_d:
                    f += np.sum( np.abs( M_d[j] - M[j] )**2 ) 
                else:
                    f += 0.1*np.sum( np.abs( M[j] )**2 ) 
            return f 

        results = minimize( CostF, phi, args=(MDF) )

    psi_hat = results.x / np.linalg.norm(results.x )
    psi_hat = psi_hat.reshape(2,-1)
    psi_hat = psi_hat[0] + 1j*psi_hat[1]

    return psi_hat



def SelfGuidedTomography( infidelity,
                            guess, 
                            num_iter, 
                            callback = lambda x, i : None,
                            postprocessing = None,
                            ):

    GAINS = {
            "a": 1.0,
            "b": 0.2,
            "A": .0,
            "s": 1.0,
            "t": 1 / 6,
            }

    def update( guess, update ):
        guess = guess + update
        guess = guess / np.linalg.norm( guess )
        if postprocessing is not None:
            guess = postprocessing( guess ) 
        return guess

    optimizer = CSPSA( init_iter=0, 
                        callback=callback,
                        gains = GAINS,
                        apply_update = update,
                        perturbations=(1,-1,1j,-1j,)
                        )

    results = optimizer.run(infidelity, guess, progressbar=False, num_iter=num_iter)

    return results 
    