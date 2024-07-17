# -*- coding: utf-8 -*-
"""
Convert two-port representations between each other (S and ABCD matrix).

Notably can convert to scattering matrix S with asymmetric input and output
impedances.

Works on numpy arrays where the 2x2 matrices are along the two last axes.

See Section 4.4 of "Microwave Engineering" by David Pozar (Wiley),
and Appendix B of "Asymmetric Passive Components in Microwave Integrated Circuits" by Hee-Ran Ahn (Wiley).

Note: a reciprocal network (no active components, ferrites, or plasma) fulfills:

- Z is symmetric (Zij = Zji)
- Y is symmetric (Yij = Yji)
- S is symmetric (Sij = Sji)

Lossless network fulfills:

- Re Zij = 0 for any i, j
- Re Yij = 0 for any i, j
- S is unitary
"""

import numpy as np


def S_to_ABCD(S, Z01, Z02=None):
    """
    Convert scttering matrix S to ABCD matrix.

    Requires characteristic impedance of the input and output port, because
    S matrix depends on them, while ABCD matrix is independent of them.

    Parameters
    ----------
    S : numpy.ndarray
        S matrix
    Z01 : float
        Characteristic impedance of input port.
    Z02 : float or None, optional
        Characteristic impedance of output port.
        If None, then the same as Z01.
        The default is None.

    Returns
    -------
    numpy.ndarray
        ABCD matrix
    """
    assert S.shape[-2:] == (2, 2)
    if Z02 is None:
        Z02 = Z01
    S11, S12 = S[...,0,0], S[...,0,1]
    S21, S22 = S[...,1,0], S[...,1,1]
    A = np.sqrt(Z01*Z02) * (-S11*S22 + S11 + S12*S21 - S22 + 1) / (2*Z02*S21)
    B = np.sqrt(Z01*Z02) * (S11*S22 + S11 - S12*S21 + S22 + 1) / (2*S21)
    C = np.sqrt(Z01*Z02)**3 * (S11*S22 - S11 - S12*S21 - S22 + 1) / (2 * Z01**2 * Z02**2 * S21)
    D = Z02 * (-S11*S22 - S11 + S12*S21 + S22 + 1) / (2 * np.sqrt(Z01*Z02) * S21)
    ABCD = np.array([[A, B], [C, D]])
    return ABCD.transpose(tuple(range(2,ABCD.ndim)) + (0,1))


def ABCD_to_S(ABCD, Z01, Z02=None):
    """
    Convert ABCD matrix to scattering matrix S.

    Requires characteristic impedance of the input and output port, because
    S matrix depends on them, while ABCD matrix is independent of them.

    Parameters
    ----------
    ABCD : numpy.ndarray
        ABCD matrix
    Z01 : float
        Characteristic impedance of input port.
    Z02 : float, optional
        Characteristic impedance of output port.
        If None, then the same as Z01.
        The default is None.

    Returns
    -------
    numpy.ndarray
        S matrix
    """
    assert ABCD.shape[-2:] == (2, 2)
    if Z02 is None:
        Z02 = Z01
    A, B, C, D = ABCD[...,0,0], ABCD[...,0,1], ABCD[...,1,0], ABCD[...,1,1]
    d = A*Z02 + B + C*Z01*Z02 + D*Z01
    S11 = (A*Z02 + B - C*Z01*Z02 - D*Z01) / d
    S12 = 2*np.sqrt(Z01*Z02) * (A*D - B*C) / d
    S21 = 2*np.sqrt(Z01*Z02) / d
    S22 = (-A*Z02 + B - C*Z01*Z02 + D*Z01) / d
    S = np.array([[S11, S12], [S21, S22]])
    return S.transpose(tuple(range(2,S.ndim)) + (0,1))


def S_to_Z(S, Z0):
    """
    Convert scattering matrix S to impedance matrix Z.

    For now only supports symmetric port impedance Z0. See Hee-Ran Ahn book
    if you want to update this.

    Parameters
    ----------
    S : numpy.ndarray
        Scattering matrix
    Z0 : float
        Impedance of input and output port.

    Returns
    -------
    numpy.ndarray
        Impedance matrix Z
    """
    assert S.shape[-2:] == (2, 2)
    S11, S12 = S[...,0,0], S[...,0,1]
    S21, S22 = S[...,1,0], S[...,1,1]
    Z11 = Z0 * ((1+S11)*(1-S22) + S12*S21) / ((1-S11)*(1-S22) - S12*S21)
    Z12 = Z0 * 2*S12 / ((1-S11)*(1-S22) - S12*S21)
    Z21 = Z0 * 2*S21 / ((1-S11)*(1-S22) - S12*S21)
    Z22 = Z0 * ((1-S11)*(1+S22) + S12*S21) / ((1-S11)*(1-S22) - S12*S21)
    Z = np.array([[Z11, Z12], [Z21, Z22]]) # shape (2,2,...)
    return Z.transpose(tuple(range(2,Z.ndim)) + (0,1))


def Z_to_Ztee(Z):
    """
    Convert impedance matrix Z to T-topology::

        --Z1---Z2--
             |
             Z3
             |
        -----------

    This requires impedance matrix to be symmetric, i.e. no active components.

    If the network is symmetric (input behaves same as output), then Z1 = Z2.

    Parameters
    ----------
    Z : numpy.ndarray
        Z matrix

    Returns
    -------
    3-tuple of floats
        (Z1, Z2, Z3)
    """
    assert Z.shape[-2:] == (2, 2)
    assert np.allclose(Z[...,0,1], Z[...,1,0]), "Z must be a symmetric matrix"
    Z11, Z12, Z22 = Z[...,0,0], Z[...,0,1], Z[...,1,1]
    Z1 = Z11 - Z12
    Z2 = Z22 - Z12
    Z3 = Z12
    return Z1, Z2, Z3


def tline_ABCD(beta_l, Z0):
    """
    ABCD matrix of transmission line with characteristic impedance Z0 and
    propagation beta_l = beta * length.

    Note that phase velocity = v_p = omega / beta = wavelength * frequency

    For example for a lambda quarter resonator: length=wavelength/2, then
    beta*length = pi.
    """
    M = np.array([
        [np.cos(beta_l), 1j*Z0 * np.sin(beta_l)],
        [1j/Z0 * np.sin(beta_l), np.cos(beta_l)]
    ])
    # switch 2x2 ABCD to last axes
    M = M.transpose((tuple(range(2,M.ndim)) + (0,1)))
    return M


def impedance_ABCD(Z):
    """
    ABCD matrix of impedance Z connecting two ports.::

        --Z--

        -----

    Parameters
    ----------
    Z : float
        Impedance

    Returns
    -------
    numpy.ndarray
        ABCD matrix
    """
    ABCD = np.array([
        [1, Z], [0, 1]])
    return ABCD.transpose((tuple(range(2,ABCD.ndim)) + (0,1)))


def admittance_ABCD(Y):
    """
    ABCD matrix of admittance Y connecting two ports.::

        -----
          |
          Y
          |
        ------

    Parameters
    ----------
    Y : float
        Admittance

    Returns
    -------
    numpy.ndarray
        ABCD matrix
    """
    ABCD = np.array([
        [1, 0], [Y, 1]])
    return ABCD.transpose((tuple(range(2,ABCD.ndim)) + (0,1)))


def Ztee_ABCD(Z1, Z2, Z3):
    """
    ABCD matrix of impedance T-topology::

        --Z1---Z2--
             |
             Z3
             |
        -----------

    Parameters
    ----------
    Z1 : float
    Z2 : float
    Z3 : float

    Returns
    -------
    numpy.ndarray
        ABCD matrix
    """
    ABCD = np.array([
        [1+Z1/Z3, Z1+Z2+Z1*Z2/Z3],
        [1/Z3, 1+Z2/Z3]])
    return ABCD.transpose((tuple(range(2,ABCD.ndim)) + (0,1)))
