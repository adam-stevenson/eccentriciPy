# A. T. Stevenson
# code to fit to eccentricity distributions.


def main():

    #---------------------------------------------------------
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import pymultinest
    import corner
    import os
    import sys
    from scipy.special import ndtri
    import chaospy
    from pymultinest.solve import Solver
    from astropy import constants as c
    from astroquery.simbad import Simbad
    import requests
    import pyvo
    import pymc as pm
    import arviz as az

    LN2PI = np.log(2.*np.pi)
    open('eccentriciPy_results.txt', 'w').close()

    #n.b. for mac, pymultinest needs this in ~/.zshrc: export DYLD_LIBRARY_PATH="/Users/a.t.stevenson@bham.ac.uk/adam/Software/eccentriciPy/MultiNest/lib:$DYLD_LIBRARY_PATH"

    #------------------------------------------------------------
    # OPTIONS
    # Define the cuts on the sample of RV planets you want to probe:

    #periods in days (for all, set 0--100000000000000 or something silly)
    p_upp = 100
    p_low = 0 

    #can have both as true 
    multi = True
    single = True

    #masses in Earth-masses.
    m_upp = 10
    m_low = 0

    #if have an archive already downloaded (e.g. using the provided one) and cuts have been made
    path_to_archive = "/Users/a.t.stevenson@bham.ac.uk/adam/Software/eccentriciPy/adapted_sample.csv"


    #------------------------------------------------------------

    usable_NASA_exoplanets=pd.read_csv(path_to_archive,sep=",")
    usable_NASA_exoplanets=usable_NASA_exoplanets[usable_NASA_exoplanets.updated_ecc>0]

    usable_NASA_exoplanets=usable_NASA_exoplanets[(usable_NASA_exoplanets.pl_orbper>=p_low)&(usable_NASA_exoplanets.pl_orbper<p_upp)]
    usable_NASA_exoplanets=usable_NASA_exoplanets[(usable_NASA_exoplanets.pl_bmasse>=m_low)&(usable_NASA_exoplanets.pl_bmasse<m_upp)]

    if multi==True and single==False:
        usable_NASA_exoplanets=usable_NASA_exoplanets[usable_NASA_exoplanets.sy_pnum>1]
    elif multi==False and single==True:
        usable_NASA_exoplanets=usable_NASA_exoplanets[usable_NASA_exoplanets.sy_pnum==1]
    else:
        usable_NASA_exoplanets=usable_NASA_exoplanets
        

    print(len(usable_NASA_exoplanets))

    all_ecc = np.array(usable_NASA_exoplanets.updated_ecc)
    all_ecc.sort()


    def edf(data, alpha=.05, x0=None, x1=None ):
        x0 = data.min() if x0 is None else x0
        x1 = data.max() if x1 is None else x1
        #x = np.linspace(x0, x1, 10000)
        x=np.unique(data)
        N = data.size
        y = np.zeros_like(x)
        l = np.zeros_like(x)
        u = np.zeros_like(x)
        e = np.sqrt(1.0/(2*N) * np.log(2./alpha))
        for i, xx in enumerate(x):
            y[i] = np.sum(data <= xx)/N
            l[i] = y[i]-e #np.maximum( y[i] - e, 0 )
            u[i] = y[i]+e #np.minimum( y[i] + e, 1 )
        return x, y, l, u, e

    all_x, all_y, all_l, all_u, all_e = edf(all_ecc, alpha=0.32)
    all_dkw_err_array = (all_u-all_l)/2

    # compute some poisson errors for comparison.

    # for each entry in this vector count number of entries in original e vector less than or equal to this value.
    counts = []
    unique_ecc=np.unique(all_ecc)

    for i in unique_ecc:
        counts.append(len(np.where(all_ecc<=i)[0]))
        
    error_array=[]
    for i in np.arange(len(counts)):
        error_array.append(np.sqrt(counts[i]))

    error_array_CDF = error_array/np.max(counts)

    with open('eccentriciPy_results.txt', 'a') as the_file:
        the_file.write(f"{len(all_ecc)} planets/  {len(unique_ecc)} unique eccentricities: Are you sure this is enough? \n")
        the_file.write("-----------------\n")

    with open('eccentriciPy_results.txt', 'a') as the_file:
        the_file.write(f'EDF method:\n')
        the_file.write("\n")

    # Global functions

    def beta(x, a, b):
        return scipy.stats.beta.cdf(x,a,b)

    def kumara(x, alpha,beta):
        return chaospy.Kumaraswamy(alpha,beta,lower=0,upper=1).cdf(x)

    def RE(x,alpha,l,sig): #normalised - Hugh helped me figure this out!
        rayleigh_scale = sig
        expon_scale = 1/l
        frac = alpha

        rayleigh=scipy.stats.rayleigh(0,scale=rayleigh_scale)
        expon=scipy.stats.expon(0,expon_scale)

        cdf = (1-frac) * (rayleigh.cdf(x) / rayleigh.cdf(1)) + (frac) * (expon.cdf(x) / expon.cdf(1))
        return cdf

    #def ST08(x,a,k):
    #    cdf=(1/k)*((1/((1-a)) * 1/((1+x)**(a-1)) - (x**2)/(2**(a+1))))
    #    return (cdf - np.min(cdf))/np.max((cdf - np.min(cdf)))

    def ST08(x,a):
        cdf=((1/((1-a)) * 1/((1+x)**(a-1)) - (x**2)/(2**(a+1))))
        return (cdf - np.min(cdf))/np.max((cdf - np.min(cdf)))


    def Gamma(x,alpha,beta):
        gamma = scipy.stats.gamma(a=alpha,scale=1/beta)
        #include normalisation to =1 at e=1. 
        cdf = (gamma.cdf(x) / gamma.cdf(1))
        return cdf

    def Rice(x,b,loc,s):
        rice = scipy.stats.rice(b=b,loc=loc,scale=s)
        cdf = rice.cdf(x)/ rice.cdf(1)
        return cdf


    #---------------------------------------------------------------------------
    # Beta

    # create Solver class

    class BetaModelPyMultiNest(Solver):
        # define the prior parameters
        amin = 0
        amax = 10
        bmin = 0  
        bmax = 10


        def __init__(self, data, abscissa, modelfunc, sigma, **kwargs):
            # set the data
            self._data = data         # oberserved data
            self._abscissa = abscissa # points at which the observed data are taken
            self._sigma = sigma       # standard deviation(s) of the data
            self._logsigma = np.log(sigma) # log sigma here to save computations in the likelihood
            self._ndata = len(data)   # number of data points
            self._model = modelfunc   # model function

            Solver.__init__(self, **kwargs)

        def Prior(self, cube):

            aprime = cube[0]
            bprime = cube[1]

            a = aprime*(self.amax-self.amin) + self.amin  # convert back to a
            b = bprime*(self.bmax-self.bmin) + self.bmin  # convert back to b

            return np.array([a, b])

        def LogLikelihood(self, cube):

            a = cube[0]
            b = cube[1]

            # calculate the model
            model = self._model(self._abscissa, a, b)

            # normalisation
            norm = np.sum(-0.5*self._ndata*LN2PI - self._ndata*self._logsigma)/self._ndata

            # chi-squared
            chisq = np.sum(((self._data - model)/(self._sigma))**2)

            return norm - 0.5*chisq


    x=all_x
    sigma=all_dkw_err_array

    nlive = 1024 # number of live points
    ndim = 2     # number of parameters
    tol = 0.5   # stopping criterion

    # run the algorithm
    solution = BetaModelPyMultiNest(all_y, all_x, beta, all_dkw_err_array, n_dims=ndim, n_live_points=nlive, evidence_tolerance=tol);
    #solution = BetaModelPyMultiNest(y_n0[x_n0>=0.2], x_n0[x_n0>=0.2], beta, err_n0[x_n0>=0.2], n_dims=ndim, n_live_points=nlive, evidence_tolerance=tol);
    #solution = BetaModelPyMultiNest(all_y[all_x>=0.2], all_x[all_x>=0.2], beta, all_dkw_err_array[all_x>=0.2], n_dims=ndim, n_live_points=nlive, evidence_tolerance=tol);

    logZpymnest = solution.logZ        # value of log Z
    logZerrpymnest = solution.logZerr  # estimate of the statistcal uncertainty on logZ

    print('Marginalised evidence is {} ± {}'.format(logZpymnest, logZerrpymnest))

    achain_pymnest = solution.samples[:,0] # extract chain of m values
    bchain_pymnest = solution.samples[:,1] # extract chain if c values

    postsamples = np.vstack((achain_pymnest, bchain_pymnest)).T
    a_low, a_med, a_upp = corner.quantile(achain_pymnest,q=[0.16,0.50,0.84])
    b_low, b_med, b_upp = corner.quantile(bchain_pymnest,q=[0.16,0.50,0.84])

    with open('eccentriciPy_results.txt', 'a') as the_file:
        the_file.write(f'Beta: a={a_med:.4f} + {(a_upp-a_med):.4f} - {(a_med-a_low):.4f}, b={b_med:.4f} + {(b_upp-b_med):.4f} - {(b_med-b_low):.4f}\n')
        the_file.write(f'Evidence is {logZpymnest:.2f} ± {logZerrpymnest:.2f}\n')
        the_file.write("-----------------\n")
        
        
    #----------------------------------------------------------
    #Kumaraswamy

    x=all_x
    sigma=all_dkw_err_array

    # create Solver class
    class KumaraModelPyMultiNest(Solver):
        
        # define the prior parameters
        alphamin = 0
        alphamax = 10
        betamin = 0
        betamax = 10


        def __init__(self, data, abscissa, modelfunc, sigma, **kwargs):
            # set the data
            self._data = data         # oberserved data
            self._abscissa = abscissa # points at which the observed data are taken
            self._sigma = sigma       # standard deviation(s) of the data
            self._logsigma = np.log(sigma) # log sigma here to save computations in the likelihood
            self._ndata = len(data)   # number of data points
            self._model = modelfunc   # model function

            Solver.__init__(self, **kwargs)

        def Prior(self, cube):

            # extract values
            alphaprime = cube[0]
            betaprime = cube[1]

            alpha = alphaprime*(self.alphamax-self.alphamin) + self.alphamin  # convert back to alpha
            beta = betaprime*(self.betamax-self.betamin) + self.betamin  # convert back to beta

            return np.array([alpha, beta])

        def LogLikelihood(self, cube):

            # extract parameters
            alpha = cube[0]
            beta = cube[1]

            # calculate the model
            model = self._model(self._abscissa, alpha, beta)

            # normalisation
            norm = np.sum(-0.5*self._ndata*LN2PI - self._ndata*self._logsigma)/self._ndata

            # chi-squared
            chisq = np.sum(((self._data - model)/(self._sigma))**2)

            return norm - 0.5*chisq

    nlive = 1024 # number of live points
    ndim = 2     # number of parameters
    tol = 0.5    # stopping criterion

    # run the algorithm
    solution = KumaraModelPyMultiNest(all_y, all_x, kumara, all_dkw_err_array, n_dims=ndim,
                                            n_live_points=nlive, evidence_tolerance=tol);

    logZpymnest = solution.logZ        # value of log Z
    logZerrpymnest = solution.logZerr  # estimate of the statistcal uncertainty on logZ

    print('Marginalised evidence is ± {}'.format(logZpymnest, logZerrpymnest))

    achain_pymnest = solution.samples[:,0] # extract chain of a values
    bchain_pymnest = solution.samples[:,1] # extract chain if b values

    postsamples = np.vstack((achain_pymnest, bchain_pymnest)).T

    print('Number of posterior samples is {}'.format(postsamples.shape[0]))

    a_low, a_med, a_upp = corner.quantile(achain_pymnest,q=[0.16,0.50,0.84])
    b_low, b_med, b_upp = corner.quantile(bchain_pymnest,q=[0.16,0.50,0.84])

    with open('eccentriciPy_results.txt', 'a') as the_file:
        the_file.write(f'Kumaraswamy: alpha={a_med:.4f} + {(a_upp-a_med):.4f} - {(a_med-a_low):.4f}, beta={b_med:.4f} + {(b_upp-b_med):.4f} - {(b_med-b_low):.4f}\n')
        the_file.write(f'Evidence is {logZpymnest:.2f} ± {logZerrpymnest:.2f}\n')
        the_file.write("-----------------\n")



    #-----------------------------------------------------------------------
    # Rayleigh + Exponential

    x=all_x
    sigma=all_dkw_err_array

    # create Solver class
    class RE_ModelPyMultiNest(Solver):

        # define the prior parameters
        alphamin = 0. 
        alphamax = 1 #this is a fraction 0 to 1. contributions of R or E adding to 1.
        lmin = 0. 
        lmax = 10
        sigmin = 0. 
        sigmax = 10
        

        def __init__(self, data, abscissa, modelfunc, sigma, **kwargs):
            # set the data
            self._data = data         # oberserved data
            self._abscissa = abscissa # points at which the observed data are taken
            self._sigma = sigma       # standard deviation(s) of the data
            self._logsigma = np.log(sigma) # log sigma here to save computations in the likelihood
            self._ndata = len(data)   # number of data points
            self._model = modelfunc   # model function

            Solver.__init__(self, **kwargs)

        def Prior(self, cube):
            # extract values
            alphaprime = cube[0]
            lprime = cube[1]
            sigprime = cube[2]


            alpha = alphaprime*(self.alphamax-self.alphamin) + self.alphamin  # convert back to alpha
            l = lprime*(self.lmax-self.lmin) + self.lmin # convert to lambda
            sig = sigprime*(self.sigmax-self.sigmin) + self.sigmin #convert to signa

            return np.array([alpha, l, sig])

        def LogLikelihood(self, cube):

            # extract parameters
            alpha = cube[0]
            l = cube[1]
            sig = cube[2]

            # calculate the model
            model = self._model(self._abscissa, alpha, l, sig)

            # normalisation
            norm = np.sum(-0.5*self._ndata*LN2PI - self._ndata*self._logsigma)/self._ndata

            # chi-squared
            chisq = np.sum(((self._data - model)/(self._sigma))**2)

            return norm - 0.5*chisq

    nlive = 1024 # number of live points
    ndim = 3     # number of parameters
    tol = 0.5   # stopping criterion

    # run the algorithm
    solution = RE_ModelPyMultiNest(all_y, all_x, RE, all_dkw_err_array, n_dims=ndim,
                                            n_live_points=nlive, evidence_tolerance=tol, multimodal=False);

    logZpymnest = solution.logZ        # value of log Z
    logZerrpymnest = solution.logZerr  # estimate of the statistcal uncertainty on logZ

    print('Marginalised evidence is ± {}'.format(logZpymnest, logZerrpymnest))

    alphachain_pymnest = solution.samples[:,0] # extract chain of m values
    lchain_pymnest = solution.samples[:,1] # extract chain if c values
    sigchain_pymnest = solution.samples[:,2]

    postsamples = np.vstack((alphachain_pymnest, lchain_pymnest, sigchain_pymnest)).T

    print('Number of posterior samples is {}'.format(postsamples.shape[0]))

    alpha_low, alpha_med, alpha_upp = corner.quantile(alphachain_pymnest,q=[0.16,0.50,0.84])
    l_low, l_med, l_upp = corner.quantile(lchain_pymnest,q=[0.16,0.50,0.84])
    sig_low, sig_med, sig_upp = corner.quantile(sigchain_pymnest,q=[0.16,0.50,0.84])

    with open('eccentriciPy_results.txt', 'a') as the_file:
        the_file.write(f'RE: alpha={alpha_med:.4f} + {(alpha_upp-alpha_med):.4f} - {(alpha_med-alpha_low):.4f}, lambda={l_med:.4f} + {(l_upp-l_med):.4f} - {(l_med-l_low):.4f}, sigma={sig_med:.4f} + {(sig_upp-sig_med):.4f} - {(sig_med-sig_low):.4f}\n')
        the_file.write(f'Evidence is {logZpymnest:.2f} ± {logZerrpymnest:.2f}\n')
        the_file.write("-----------------\n")

    #----------------------------------------------------------------
    #ST08


    class ST08_ModelPyMultiNest(Solver):

        # define the prior parameters
        amin = 0. 
        amax = 100
        #kmin = 0. 
        #kmax = 100
        

        def __init__(self, data, abscissa, modelfunc, sigma, **kwargs):
            # set the data
            self._data = data         # oberserved data
            self._abscissa = abscissa # points at which the observed data are taken
            self._sigma = sigma       # standard deviation(s) of the data
            self._logsigma = np.log(sigma) # log sigma here to save computations in the likelihood
            self._ndata = len(data)   # number of data points
            self._model = modelfunc   # model function

            Solver.__init__(self, **kwargs)

        def Prior(self, cube):

            # extract values
            aprime = cube[0]
            #kprime = cube[1]

            a = aprime*(self.amax-self.amin) + self.amin  # convert back to a
            #k = kprime*(self.kmax-self.kmin) + self.kmin  # convert back to k

            return np.array([a])

        def LogLikelihood(self, cube):

            # extract parameters
            a = cube[0]
            #k = cube[1]

            # calculate the model
            model = self._model(self._abscissa, a)

            # normalisation
            norm = np.sum(-0.5*self._ndata*LN2PI - self._ndata*self._logsigma)/self._ndata

            # chi-squared
            chisq = np.sum(((self._data - model)/(self._sigma))**2)

            return norm - 0.5*chisq

    x=all_x
    sigma=all_dkw_err_array

    LN2PI = np.log(2.*np.pi)
    LNSIGMA = np.log(sigma)

    nlive = 1024 # number of live points
    ndim = 1     # number of parameters
    tol = 0.5   # stopping criterion

    # run the algorithm
    solution = ST08_ModelPyMultiNest(all_y, all_x, ST08, all_dkw_err_array, n_dims=ndim,
                                            n_live_points=nlive, evidence_tolerance=tol, multimodal=False);

    logZpymnest = solution.logZ        # value of log Z
    logZerrpymnest = solution.logZerr  # estimate of the statistcal uncertainty on logZ

    print('Marginalised evidence is ± {}'.format(logZpymnest, logZerrpymnest))

    achain_pymnest = solution.samples[:,0] # extract chain of m values


    print('Number of posterior samples is {}'.format(postsamples.shape[0]))

    a_low, a_med, a_upp = corner.quantile(achain_pymnest,q=[0.16,0.50,0.84])

    with open('eccentriciPy_results.txt', 'a') as the_file:
        the_file.write(f'ST08: a={a_med:.4f} + {(a_upp-a_med):.4f} - {(a_med-a_low):.4f}\n')
        the_file.write(f'Evidence is {logZpymnest:.2f} ± {logZerrpymnest:.2f}\n')
        the_file.write("-----------------\n")
        
        
    #------------------------------------------------------------------------
    #Gamma 

    class GammaModelPyMultiNest(Solver):
        # define the prior parameters
        amin = 0
        amax = 10
        bmin = 0  
        bmax = 10


        def __init__(self, data, abscissa, modelfunc, sigma, **kwargs):
            # set the data
            self._data = data         # oberserved data
            self._abscissa = abscissa # points at which the observed data are taken
            self._sigma = sigma       # standard deviation(s) of the data
            self._logsigma = np.log(sigma) # log sigma here to save computations in the likelihood
            self._ndata = len(data)   # number of data points
            self._model = modelfunc   # model function

            Solver.__init__(self, **kwargs)

        def Prior(self, cube):

            aprime = cube[0]
            bprime = cube[1]

            a = aprime*(self.amax-self.amin) + self.amin  # convert back to a
            b = bprime*(self.bmax-self.bmin) + self.bmin  # convert back to b

            return np.array([a, b])

        def LogLikelihood(self, cube):

            a = cube[0]
            b = cube[1]

            # calculate the model
            model = self._model(self._abscissa, a, b)

            # normalisation
            norm = np.sum(-0.5*self._ndata*LN2PI - self._ndata*self._logsigma)/self._ndata

            # chi-squared
            chisq = np.sum(((self._data - model)/(self._sigma))**2)

            return norm - 0.5*chisq

    x=all_x
    sigma=all_dkw_err_array

    LN2PI = np.log(2.*np.pi)
    LNSIGMA = np.log(sigma)

    nlive = 1024 # number of live points
    ndim = 2     # number of parameters
    tol = 0.5   # stopping criterion

    # run the algorithm
    solution = GammaModelPyMultiNest(all_y, all_x, Gamma, all_dkw_err_array, n_dims=ndim,n_live_points=nlive, evidence_tolerance=tol, multimodal=False)

    logZpymnest = solution.logZ        # value of log Z
    logZerrpymnest = solution.logZerr  # estimate of the statistcal uncertainty on logZ

    print('Marginalised evidence is ± {}'.format(logZpymnest, logZerrpymnest))

    achain_pymnest = solution.samples[:,0] # extract chain of m values
    bchain_pymnest = solution.samples[:,1] # extract chain if c values


    postsamples = np.vstack((achain_pymnest, bchain_pymnest)).T

    print('Number of posterior samples is {}'.format(postsamples.shape[0]))

    a_low, a_med, a_upp = corner.quantile(achain_pymnest,q=[0.16,0.50,0.84])
    b_low, b_med, b_upp = corner.quantile(bchain_pymnest,q=[0.16,0.50,0.84])

    with open('eccentriciPy_results.txt', 'a') as the_file:
        the_file.write(f'Gamma: alpha={a_med:.4f} + {(a_upp-a_med):.4f} - {(a_med-a_low):.4f}, beta={b_med:.4f} + {(b_upp-b_med):.4f} - {(b_med-b_low):.4f}\n')
        the_file.write(f'Evidence is {logZpymnest:.2f} ± {logZerrpymnest:.2f}\n')
        the_file.write("-----------------\n")
    
    # END OF EDF METHOD
    #----------------------------------------------------------------
 
    with open('eccentriciPy_results.txt', 'a') as the_file:
        the_file.write("\n")
        the_file.write(f'PDF method:\n')
        the_file.write("n.b only compare evidences within each method, not between each other! \n")
        the_file.write("\n")

    #PDF method 
    #------------------------------------------------------------------



    models = []
    idatas = []

    with pm.Model() as beta_model:

        # priors

        a = pm.Uniform('a', lower=0.1, upper=5)
        b = pm.Truncated('b', pm.Uniform.dist(lower=0.1, upper=10), lower=a) 
    
         # likelihood
        likelihood = pm.Beta('like', alpha=a, beta=b, observed=all_ecc)


         # run MCMC
        #had issues for multiprocessing. testing this bit....
        trace_beta = pm.sample(4000, cores=4)

    
        idata = pm.sample_smc(random_seed=42)
        models.append(beta_model)
        idatas.append(idata)

    #print(idatas[-1].sample_stats["log_marginal_likelihood"])
    #print(idatas[-1].sample_stats["log_marginal_likelihood"][0])
    output = az.summary(trace_beta, round_to=4).values
    #ev = idatas[-1].sample_stats["log_marginal_likelihood"].mean().values.item()
    #ev_std = idatas[-1].sample_stats["log_marginal_likelihood"].std().values.item()
    ev = idatas[-1].sample_stats['log_marginal_likelihood'].values[-1][-1]


    with open('eccentriciPy_results.txt', 'a') as the_file:
        the_file.write(f'Beta: a={output[0][0]:.4f} ± {output[0][1]:.4f}, b={output[1][0]:.4f} ± {output[1][1]:.4f}\n')
        the_file.write(f'Evidence is {ev:.2f}\n')# ± {ev_std:.2f}\n')
        #the_file.write(f'Evidence is '+str(ev)+'\n')# ± {ev_std:.2f}\n')
        the_file.write("-----------------\n")
  
    #--------------------------------------------------------
    # RE

    with pm.Model() as mixture_model:

        # priors

        w = pm.Dirichlet('w', a=np.array([1, 1]))  # 2 mixture weights

        lam = pm.Uniform('lam', lower=0.1, upper=10)

        sig = pm.Uniform('sig', lower=0.05, upper=10)


        exp = pm.Exponential.dist(lam=lam)

        ray = pm.Weibull.dist(alpha=2, beta=np.sqrt(2) * sig)

        components = [pm.Truncated.dist(exp, upper=1), pm.Truncated.dist(ray, upper=1)]


        # likelihood

        likelihood = pm.Mixture('like', w=w, comp_dists=components, observed=all_ecc)

        trace_mixture = pm.sample(4000, cores=4)
    
        idata = pm.sample_smc(random_seed=42)
        models.append(mixture_model)
        idatas.append(idata)

    output = az.summary(trace_mixture, round_to=4).values
    ev = idatas[-1].sample_stats["log_marginal_likelihood"].mean().values.item()
    #ev_std = idatas[-1].sample_stats["log_marginal_likelihood"].std().values.item()
    #ev = idatas[-1].sample_stats["log_marginal_likelihood"].values[-1][-1]


    with open('eccentriciPy_results.txt', 'a') as the_file:
        the_file.write(f'RE: alpha={output[0][0]:.4f} ± {output[0][1]:.4f}, lambda={output[2][0]:.4f} ± {output[2][1]:.4f}, sigma={output[3][0]:.4f} ± {output[3][1]:.4f}\n')
        the_file.write(f'Evidence is {ev:.2f}\n')# ± {ev_std:.2f}\n')
        #the_file.write(f'Evidence is '+str(ev)+'\n')# ± {ev_std:.2f}\n')
        the_file.write("-----------------\n")
    
    
    
    #---------------------------------------------------------
    #Gamma

    with pm.Model() as gamma_model:

        # priors

        a = pm.Uniform('a', lower=0.1, upper=5)
        b = pm.Truncated('b', pm.Uniform.dist(lower=0.1, upper=15), lower=a) 
        # likelihood

        likelihood = pm.Gamma('like', alpha=a, beta=b, observed=all_ecc)

        # run MCMC
        trace_gamma = pm.sample(4000, cores=4)

        idata = pm.sample_smc(random_seed=42)
        models.append(gamma_model)
        idatas.append(idata)

    output = az.summary(trace_gamma, round_to=4).values
    #ev = idatas[-1].sample_stats["log_marginal_likelihood"].mean().values.item()
    #ev_std = idatas[-1].sample_stats["log_marginal_likelihood"].std().values.item()
    ev = idatas[-1].sample_stats['log_marginal_likelihood'].values[-1][-1]


    with open('eccentriciPy_results.txt', 'a') as the_file:
        the_file.write(f'Gamma: alpha={output[0][0]:.4f} ± {output[0][1]:.4f}, beta={output[1][0]:.4f} ± {output[1][1]:.4f}\n')
        the_file.write(f'Evidence is {ev:.2f}\n')# ± {ev_std:.2f}\n')
        #the_file.write(f'Evidence is '+str(ev)+'\n')# ± {ev_std:.2f}\n')
        the_file.write("-----------------\n")


if __name__=='__main__':
    main()