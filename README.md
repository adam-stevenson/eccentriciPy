Code to fit distributions to a sample of RV exoplanet eccentricities. Evidences are calculate to compare the 'best' distribution, penalising additional unnecessary parameters. The input archive can be tailored depending on what type of systems or planets you are interested in, to use as a more-informative prior that reflects the underlying population. 

At present, can be modified for single/multiple planet systems, period range, mass range ('OPTIONS' adjusted in first few lines of script). The inital archive used, with my modifications for planets with eccentricity fixed to zero, is also uploaded.

Packages imported in the script (some may be legacy, or needed in future): pandas, numpy, matplotlib, scipy, pymultinest, corner, os, sys, scipy, chaospy, astropy, astroquery, requests, pyvo, pymc, arviz. See individual sites for installation (most should be easy to pip-install, pymultinest may require a bit more effort). 

