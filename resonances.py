# -*- coding: utf-8 -*-
r"""
Module with fit functions for typical resonance shapes in IQ plane.

Also provides factory functions to make `lmfit` model instances with some
parameter hints preset. Note that you can modify parameter hints afterwards.

### Complex values

The resonances we want to fit are complex valued (as usually measured by
microwave instrumentation). Common fit functions like
`scipy.optimize.curve_fit()` and lmfit only work with real values.  As a
work-around we use `numpy.ndarray.view(float)`. This way real and imaginary
parts are fitted independently of each other (as they should be, since they
are orthogonal).

This means that you have to also use `.view(float)` for the data supplied to
`curve_fit()`. When evaluating the functions to retrieve the model, you have
to undo this expansion by doing applying `.view(complex)`

### Units

The factors of $2\pi$ have to cancel out, you can either use quantities in
angular units (i.e. $\omega$, $\kappa$, delay $t_d/2\pi$) or frequencies
(i.e. $f$, $f_r$, width, delay $t_d$). Fit functions here are written in the
latter.

### Background signal

Allows a linear background signal with electrical delay $t_d$ (slope in phase)
and amplitude slope $A_\text{slope}$.

$$A (1 + A_\text{slope} (f - f_r))  e^{i (2\pi t_d (f-f_r) + \theta)}$$

The number of fit parameters in `scipy.optimize.curve_fit()` is either all, or
the length of `p0`. So you can only vary the first few parameters of the
function. The parameters are ordered so that the ones more likely to be fixed
are in the end of the argument list.

### Tips

For fit stability it is recommended to

- scale frequencies close to unity, by fitting e.g. in GHz units,
- remove the phase delay and slope before fitting and not vary them in the fit.
"""

from lmfit import Model


def reflection(f, fr, external, internal, A, theta=0, delay=0, Aslope=0):
    """
    Resonance in reflection measurement: a Lorentzian dip at resonance.

    The resonance dip goes down to S=0 if external and internal quality factor
    are equal.

    You might want to fit the complex conjugate of your signal if its phase has for
    some reason the opposite sign of the model. The model is:
    Undercoupled (external < internal) has a phase wiggle going down then up.
    Overcoupled (external > internal) has a 2pi phase shift down.

    Complex conjugate of eq. (2.16) in thesis
    of Palacios-Laloy (CEA Saclay, 2010), section 2.1.3.2.

    Parameters
    ----------
    f : float
        Frequencies to evaluate function at.
    fr : float
        Resonance frequency
    external : float
        External coupling
    internal : float
        Internal losses
    A : float
        Amplitude
    theta : float, optional
        Phase offset. The default is 0.
    delay : float, optional
        Electrical delay. The default is 0.
    Aslope : float, optional
        Amplitude slope. The default is 0.

    Returns
    -------
    float
        Data with Im values concatenated after Re values.
    """
    import numpy as np # for multiprocess
    S = np.conj(((external-internal) + 2j*(f-fr)) / ((external+internal) - 2j*(f-fr)))
    S *= A * (1+Aslope*(f-fr)) * np.exp(1j*(theta + delay*(f-fr)))
    return S.view(float)


def reflection_asym(f, fr, external, internal, A, phi, y, theta=0, delay=0, Aslope=0):
    """
    Asymmetric resonance in reflection measurement: a Lorentzian dip at resonance.

    In the reflection case the asymmetry is non-trivial. The parameters of
    the scatterer phi and y, are often strongly correlated with the other
    parameters, which makes the fitting unstable, and gives large
    uncertainties.

    Parameters
    ----------
    f : float
        Frequencies to evaluate function at.
    fr : float
        Resonance frequency
    external : float
        External coupling
    internal : float
        Internal losses
    A : float
        Amplitude
    phi : float
        Asymmetry phase
    y : float
        Asymmetry, relative amplitude of scatterer
    theta : float, optional
        Phase offset. The default is 0.
    delay : float, optional
        Electrical delay. The default is 0.
    Aslope : float, optional
        Amplitude slope. The default is 0.

    Returns
    -------
    float
        Data with Im values concatenated after Re values.
    """
    import numpy as np # for multiprocess
    Sbare = np.conj(((external-internal) + 2j*(f-fr)) / ((external+internal) - 2j*(f-fr)))
    S = y + np.exp(1j*phi) * Sbare
    S *= A * (1+Aslope*(f-fr)) * np.exp(1j*(theta + delay*(f-fr)))
    return S.view(float)


def transmission(f, fr, width, A, theta=0, delay=0, Aslope=0):
    """
    Resonance in transmission: a Lorentzian peak at resonance.

    You might want to fit the complex conjugate of your signal if its phase has for
    some reason the opposite sign of the model. The model has a pi phase shift down.

    Note that it is typically not possible to fit internal losses, because
    different contributions to the width are degenerate with amplitude

    Simplified version of equation seen in thesis
    of Palacios-Laloy (CEA Saclay, 2010), section 2.1.3.2.

    Parameters
    ----------
    f : float
        Frequencies to evaluate function at.
    fr : float
        Resonance frequency
    width : float
        Total width
    A : float
        Amplitude
    theta : float, optional
        Phase offset. The default is 0.
    delay : float, optional
        Electrical delay. The default is 0.
    Aslope : float, optional
        Amplitude slope. The default is 0.

    Returns
    -------
    float
        Data with Im values concatenated after Re values.
    """
    import numpy as np # for multiprocess
    S = np.conj(width / (width - 2j*(f-fr)))
    S *= A * (1+Aslope*(f-fr)) * np.exp(1j*(theta + delay*(f-fr)))
    return S.view(float)


def hanger_transmission(f, fr, external, internal, A, theta=0, delay=0, Aslope=0):
    """
    Transmission in hanger geometry: a Lorentzian dip at resonance.

    Different from the Lorentzian peak of the simple reflection measurement,
    the value of transmission on resonance is given by `external/(external+internal)`.

    Parameters
    ----------
    f : float
        Frequencies to evaluate function at.
    fr : float
        Resonance frequency
    external : float
        External coupling
    internal : float
        Internal losses
    A : float
        Amplitude
    theta : float, optional
        Phase offset. The default is 0.
    delay : float, optional
        Electrical delay. The default is 0.
    Aslope : float, optional
        Amplitude slope. The default is 0.

    Returns
    -------
    float
        Data with Im values concatenated after Re values.
    """
    import numpy as np # for multiprocess
    S = 1 - external / (external + internal + 2j*(f-fr))
    S *= A * (1+Aslope*(f-fr)) * np.exp(1j*(theta + delay*(f-fr)))
    return S.view(float)


def hanger_transmission_asym(f, fr, external, internal, phi, A, theta=0, delay=0, Aslope=0):
    """
    Transmission in hanger geometry with asymmetry.

    The asymmetry usually comes from additional scattering/resonances along the
    microwave line or in the hanger transmission line.

    The resonance frequency and external coupling factor depend on this
    additional scatterer.

    Parameters
    ----------
    f : float
        Frequencies to evaluate function at.
    fr : float
        Resonance frequency
    external : float
        External coupling
    internal : float
        Internal losses
    phi : float
        Asymmetry
    A : float
        Amplitude
    theta : float, optional
        Phase offset. The default is 0.
    delay : float, optional
        Electrical delay. The default is 0.
    Aslope : float, optional
        Amplitude slope. The default is 0.

    Returns
    -------
    float
        Data with Im values concatenated after Re values.
    """
    import numpy as np # for multiprocess
    S = 1 - np.exp(1j*phi) * external / (external + internal + 2j*(f-fr))
    S *= A * (1+Aslope*(f-fr)) * np.exp(1j*(theta + delay*(f-fr)))
    return S.view(float)


def hanger_reflection(f, fr, width, A, theta=0, delay=0, Aslope=0):
    """
    Hanger resonance in reflection: a Lorentzian peak at resonance.

    Note that it is typically not possible to differentiate the coupling
    into interal / external, because that is degenerate with amplitude.

    Parameters
    ----------
    f : float
        Frequencies to evaluate function at.
    fr : float
        Resonance frequency
    width : float
        Total width
    A : float
        Amplitude
    theta : float, optional
        Phase offset. The default is 0.
    delay : float, optional
        Electrical delay. The default is 0.
    Aslope : float, optional
        Amplitude slope. The default is 0.

    Returns
    -------
    float
        Data with Im values concatenated after Re values.
    """
    import numpy as np # for multiprocess
    S = width / (width - 2j*(f-fr))
    S *= A * (1+Aslope*(f-fr)) * np.exp(1j*(theta + delay*(f-fr)))
    return S.view(float)


def hanger_reflection_asym(f, fr, width, A, phi, y=0, theta=0, delay=0, Aslope=0):
    """
    Asymmetric hanger resonance in reflection: a Lorentzian peak at resonance.

    In the reflection case the asymmetry is non-trivial. The parameters of
    the scatterer phi and y, are often strongly correlated with the other
    parameters, which makes the fitting unstable, and gives large
    uncertainties.

    Parameters
    ----------
    f : float
        Frequencies to evaluate function at.
    fr : float
        Resonance frequency
    width : float
        Total width
    A : float
        Amplitude
    phi : float
        Asymmetry phase
    y : float
        Asymmetry, scattering amplitude
    theta : float, optional
        Phase offset. The default is 0.
    delay : float, optional
        Electrical delay. The default is 0.
    Aslope : float, optional
        Amplitude slope. The default is 0.

    Returns
    -------
    float
        Data with Im values concatenated after Re values.
    """
    import numpy as np # for multiprocess
    Sbare = width / (width - 2j*(f-fr))
    S = y + np.exp(1j*phi) * Sbare
    S *= A * (1+Aslope*(f-fr)) * np.exp(1j*(theta + delay*(f-fr)))
    return S.view(float)


def make_reflection_model():
    """
    `lmfit.model.Model` for `reflection()`

    Creates a new instance, so you can modify anything.

    Returns
    -------
    model : lmfit.model.Model
    """
    model = Model(reflection)
    model.set_param_hint('fr', min=0)
    model.set_param_hint('external', min=0)
    model.set_param_hint('internal', min=0)
    model.set_param_hint('A', min=0)
    return model


def make_asym_reflection_model():
    """
    `lmfit.model.Model` for `reflection_asym()`

    Creates a new instance, so you can modify anything.

    Returns
    -------
    model : lmfit.model.Model
    """
    model = Model(reflection_asym)
    model.set_param_hint('fr', min=0)
    model.set_param_hint('external', min=0)
    model.set_param_hint('internal', min=0)
    model.set_param_hint('A', min=0)
    model.set_param_hint('y', min=0)
    return model


def make_transmission_model():
    """
    `lmfit.model.Model` for `transmission()`

    Creates a new instance, so you can modify anything.

    Returns
    -------
    model : lmfit.model.Model
    """
    model = Model(transmission) #(f, fr, width, A, theta=0, delay=0, Aslope=0)
    model.set_param_hint('fr', min=0)
    model.set_param_hint('width', min=0)
    model.set_param_hint('A', min=0)
    return model
