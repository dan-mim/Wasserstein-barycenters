# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 10:03:18 2023

@author: mimounid
"""
import time
import numpy as np


def barycenter_IBP(b, M_dist, computation_time=500, iterations_min=100, iterations_max=100000, lambda_sinkhorn=1000):
    """
    Input:
    *b: (R x n) vector collection of n probability distribution 
    *M_dist:(R x R) is the distance matrix of every pixels 
    *lambda_sinkhorn: (float) regularization parameter
    *iterations_IBP: (int) number of iterations for the method

    Output:
    *p: (R x 1) the probability distribution of the calculated barycenter
    *Time: execution time 
    * Precision
    *iteration_IBP: (int) number of iterations 

    Infos:
    This function compute the barycenter of probability distribution in the case of a support identical for all distribution and for the targeted barycenter
    This algorithm is inspired from 'Iterative Bregman Projections for Regularized Transportation Problems'

    (c) Daniel Mimouni 2023
    """""

    # Time management:
    start = time.time()

    # parameters:
    # dimension of the support (number of atoms):
    R = len(b[0])
    # number of probability measures:
    M = len(b)

    # Keep Track:
    Precision = np.zeros(iterations_max + 1)
    l_time = np.zeros(iterations_max + 1)
    P = np.zeros((R, iterations_max + 1))

    # probability measures
    Q = np.array(b).T
    # initialization :
    U = np.ones((R, M))

    # I should divide M_dist by the mean to get it normalized:
    max_D = np.max(M_dist)
    M_dist = M_dist / max_D
    K = np.exp(- lambda_sinkhorn * M_dist)

    # Get UKv:
    sumK = np.sum(K, axis=0)
    UKv = K @ np.divide(Q, np.tile(sumK, (M, 1)).T)  # sumK.T

    # Get U:
    expUKv = np.exp(np.mean(np.log(UKv), axis=1))
    U = np.divide(np.tile(expUKv, (M, 1)).T, UKv)

    # Loop:
    iteration_IBP = 0
    while (iteration_IBP < iterations_min) or (
            time.time() - start < computation_time and iteration_IBP < iterations_max):
        iteration_IBP = iteration_IBP + 1
        iteration_start_time = time.time()

        # Get UKv:
        UKv = np.multiply(U, K @ np.divide(Q, K.T @ U))

        # Get U:
        expUKv = np.exp(np.mean(np.log(UKv), axis=1))
        U = np.multiply(U, np.tile(expUKv, (M, 1)).T)
        U = np.divide(U, UKv)

        # Barycenter Output
        p = np.mean(UKv, axis=1)

        # Precision:
        Precision[iteration_IBP] = np.sum(np.std(UKv, axis=1))
        l_time[iteration_IBP] = time.time() - iteration_start_time
        P[:, iteration_IBP] = p

    # Time management
    Time = time.time() - start
    # Outputs:
    return (p, P, l_time, Time, Precision, iteration_IBP)








