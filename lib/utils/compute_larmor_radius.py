#!/usr/bin/env python


import astropy.units as u

import numpy as np
import astropy.constants as const
from collections import namedtuple

avg_B = 35e3 * u.nT


def compute_larmor_radius(v, Z=1, A=1):
    m = Z * const.m_p + (A - Z) * const.m_n
    lorentz = lambda v: 1 / np.sqrt(1 - (v.to('km/s') / const.c.to('km/s')))
    print(lorentz(v))
    R = lorentz(v).value * v * m / (Z*const.e.to('C') * avg_B)
    E = lorentz(v) * m * const.c**2
    return R, E


def model(v, Z, A):
    # Assign units
    v = v * u.kilometer/u.second
    radius = compute_larmor_radius(v, Z, A)

def apply_units(variable):
    """

    Parameters
    ----------
    variable : namedtuple

    Returns
    -------

    """
    return u.Quantity(variable.quantity, unit = variable.unit)





def compute_relativistic_velocity(E, E_unit, Z=1, A=0):
    """ compute the velocity of a cosmic ray with energy E
    E = gamma * m * c^2
    gamma = 1/sqrt(1-(v/c)^2)

    Parameters
    ----------
    E : Energy of particle
    E_unit : units of E
    Z : Atomic number
    A : Mass number
    Returns
    -------

    """
    # E = (gamma - 1)mc^2

    E = u.Quantity(E, unit=E_unit)
    m = Z * const.m_p.to('kg') + (A - Z) * const.m_n.to('kg') # mass of in kg
    rest_E = m * const.c**2
    v = const.c.to('m/s') * np.sqrt(1 - (rest_E.to('J') / E.to('J'))**2)
    return v




if __name__ == '__main__':
    pass