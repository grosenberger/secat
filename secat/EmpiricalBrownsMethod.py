
"""
Created on Sun Jun 21 12:22:12 2015

@author:  William Poole: wpoole@caltech.edu
"""

import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF #empirical cumulative distribution function
from scipy.special import chdtrc as chi2_cdf
from scipy.stats import pearsonr


#Input: An m x n data matrix with each of m rows representing a variable and each of n columns representing a sample. Should be of type numpy.array
#       A vector of m P-values to combine. May be a list or of type numpy.array.
#Output: A combined P-value using the Empirical Brown's Method (EBM).
#        If extra_info == True: also returns the p-value from Fisher's method, the scale factor c, and the new degrees of freedom from Brown's Method
def EmpiricalBrownsMethod(data_matrix, p_values, extra_info = False):
    covar_matrix = CalculateCovariances(data_matrix)
    return CombinePValues(covar_matrix, p_values, extra_info)


#Input: raw data vector (of one variable) with no missing samples. May be a list or an array.
#Output Transforemd data vector w.
def TransformData(data_vector):
    m = np.mean(data_vector)
    sd = np.std(data_vector)
    s = [(d-m)/sd for d in data_vector]
    W = lambda x: -2*np.log(ECDF(s)(x))
    return np.array([W(x) for x in s])

#Input: An m x n data matrix with each of m rows representing a variable and each of n columns representing a sample. Should be of type numpy.array.
#       Note: Method does not deal with missing values within the data.
#Output: An m x m matrix of pairwise covariances between transformed raw data vectors
def CalculateCovariances(data_matrix):
    transformed_data_matrix = np.array([TransformData(f) for f in data_matrix])
    covar_matrix = np.cov(transformed_data_matrix)

    return covar_matrix
    
#Input: A m x m numpy array of covariances between transformed data vectors and a vector of m p-values to combine.
#Output: A combined P-value. 
#        If extra_info == True: also returns the p-value from Fisher's method, the scale factor c, and the new degrees of freedom from Brown's Method
def CombinePValues(covar_matrix, p_values, extra_info = False):
    m = int(covar_matrix.shape[0])
    #print "m", m
    df_fisher = 2.0*m
    Expected = 2.0*m
    cov_sum = 0
    for i in range(m):
        for j in range(i+1, m):
            cov_sum += covar_matrix[i, j]
    
    #print "cov sum", cov_sum
    Var = 4.0*m+2*cov_sum
    c = Var/(2.0*Expected)
    df_brown = 2.0*Expected**2/Var
    if df_brown > df_fisher:
        df_brown = df_fisher
        c = 1.0

    x = 2.0*sum([-np.log(p) for p in p_values])
    #print "x", x
    p_brown = chi2_cdf(df_brown, 1.0*x/c)
    p_fisher = chi2_cdf(df_fisher, 1.0*x)
    
    if extra_info:
        return p_brown, p_fisher, c, df_brown
    else:
        return p_brown

#Input: An m x n data matrix with each of m rows representing a variable and each of n columns representing a sample. Should be of type numpy.array
#       A vector of m P-values to combine. May be a list or of type numpy.array.
#Output: A combined P-value using Kost's Method.
#        If extra_info == True: also returns the p-value from Fisher's method, the scale factor c, and the new degrees of freedom from Brown's Method
def KostsMethod(data_matrix, p_values, extra_info = False):
    covar_matrix = CalculateKostCovariance(data_matrix)
    return CombinePValues(covar_matrix, p_values, extra_info = extra_info)
    
#Input correlation between two n x n data vectors.
#Output: Kost's approximation of the covariance between the -log cumulative distributions. This is calculated with a cubic polynomial fit.
def KostPolyFit(cor):
    a1, a2, a3 = 3.263, .710, .027 #Kost cubic coeficients
    return a1*cor+a2*cor**2+a3*cor**3
    
#Input: An m x n data matrix with each of m rows representing a variable and each of n columns representing a sample. Should be of type numpy.array.
#       Note: Method does not deal with missing values within the data.
#Output: An m x m matrix of pairwise covariances between the data vectors calculated using Kost's polynomial fit and numpy's pearson correlation function.
def CalculateKostCovariance(data_matrix):
    m = data_matrix.shape[0]
    covar_matrix = np.zeros((m, m))
    for i in range(m):
        for j in range(i+1, m):
            cor, p_val = pearsonr(data_matrix[i, :], data_matrix[j, :])
            covar = KostPolyFit(cor)
            covar_matrix[i, j] = covar
            covar_matrix[j, i] = covar
    return covar_matrix
    
    
    
    



        
    