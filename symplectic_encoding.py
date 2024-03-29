from qecsim.paulitools import bsp
import numpy as np
import itertools
# Implementation of Algorithm 1 from https://arxiv.org/pdf/1803.06987.pdf
# Finds an encoding matrix F such that xF = y where x are binary symplectic vectors representing
# the unencoded stabilizers and y are the encoded stabilizers
def generate_unencoded_stabilizers_bsf(n,k):
    unencoded_stabilizers = []
    for i in range(k,n):
        stabilizer = np.zeros(2*n,dtype=int)
        stabilizer[i+n] = 1
        unencoded_stabilizers.append(stabilizer)
    return unencoded_stabilizers


def find_anticommuting_symplectic_vector(x, y):
    # loop through binary vectors of length len(x)
    # and find one that anticommutes with x and y
    for z in itertools.product([0, 1], repeat=len(x)):
        z = np.array(z, dtype=int)
        if bsp(x, z) == 1 and bsp(y, z) == 1:
            return z


def find_specified_symplectic_vector(x, y, Y):
    # loop through binary vectors of length len(x)
    # and find one that anticommutes with x and y and commutes with all elements in Y
    for z in itertools.product([0, 1], repeat=len(x)):
        z = np.array(z, dtype=int)
        commutes_with_all_Y = True
        for y_i in Y:
            if bsp(y_i, z) == 1:
                commutes_with_all_Y = False
        if commutes_with_all_Y == True:
            if bsp(x, z) == 1 and bsp(y, z) == 1:
                return z


def symplectic_transvection(h):
    m = int(len(h) / 2)
    Omega = np.block([[np.zeros((m, m)), np.eye(m)], [np.eye(m), np.zeros((m, m))]])
    Omega = Omega.astype(int)
    Fh = (np.eye(2 * m, dtype=int) + (Omega @ np.outer(h, h)) % 2) % 2
    return Fh.astype(int)


def generate_a_symplectic_encoding_matrix(n, k, stabilizers):
    # arguments: n,k, stabilizers in binary symplectic form
    unencoded_stabilizers = generate_unencoded_stabilizers_bsf(n, k)
    x = unencoded_stabilizers
    y = stabilizers
    n = len(x[0])
    # intial step:
    F = np.eye(n, dtype=int)
    if bsp(x[0], y[0]) == 1:
        h1 = (x[0] + y[0]) % 2
        F = (F @ symplectic_transvection(h1) % 2).astype(int)
    else:
        w1 = find_anticommuting_symplectic_vector(x[0], y[0])
        h11 = (y[0] + w1) % 2
        h11 = h11.astype(int)
        h12 = (x[0] + w1) % 2
        h12 = h12.astype(int)

        F = (F @ symplectic_transvection(h11) @ symplectic_transvection(h12) % 2).astype(int)
    for i in range(1, len(x)):
        tilde_xi = (x[i] @ F) % 2
        tilde_xi = tilde_xi.astype(int)
        if not np.array_equal(tilde_xi, y[i]):
            if bsp(tilde_xi, y[i]) == 1:
                hi = (tilde_xi + y[i]) % 2
                F = (F @ symplectic_transvection(hi.astype(int)) % 2).astype(int)
            else:
                wi = find_specified_symplectic_vector(tilde_xi, y[i], y[:i])
                hi1 = (y[i] + wi) % 2
                hi1 = hi1.astype(int)
                hi2 = (tilde_xi + wi) % 2
                hi2 = hi2.astype(int)
                F = (F @ symplectic_transvection(hi1) @ symplectic_transvection(hi2) % 2).astype(int)

    return F

def inverse_symplectic(F):

    #F is assumed to be a square matrix
    r, c = np.shape(F)
    m = int(r/2)
    Omega = np.block([[np.zeros((m, m)), np.eye(m)], [np.eye(m), np.zeros((m, m))]])
    Omega = Omega.astype(int)
    return (Omega@np.transpose(F)@Omega)%2