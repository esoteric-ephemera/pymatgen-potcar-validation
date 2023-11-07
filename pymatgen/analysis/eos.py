"""
This module implements various equation of states.

Note: Most of the code were initially adapted from ASE and deltafactor by
@gmatteo but has since undergone major refactoring.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABCMeta, abstractmethod
from copy import deepcopy

import numpy as np
from scipy.optimize import leastsq, minimize
from typing import Sequence

from pymatgen.core.units import FloatWithUnit
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig_plt, pretty_plot

__author__ = "Kiran Mathew, gmatteo"
__credits__ = "Cormac Toher"

logger = logging.getLogger(__file__)

class EOSCore(metaclass=ABCMeta):
    """
    Abstract class that must be subclassed by all equation of state
    implementations.

    Defines EOS from the physical parameters rather than data.
    """

    def __init__(self, E0 : float, B0 : float, B1 : float, V0: float):
        self.e0 = E0
        self.v0 = V0
        self.b0 = B0
        self.b1 = B1
        return
    
    @staticmethod
    def _provenance() -> dict | list[dict]:
        """ To be implemented by user style dict or list of dicts for the origin of the EOS code """
        return

    def provenance(self) -> str:
        """ Should be a bibtex style dict for the origin of the EOS code """

        refs = self._provenance()
        retscalar = False
        if not isinstance(refs,list):
            reflist = [refs]
            retscalar = True

        bibtex_refs = []
        for ref in reflist:

            bibtex_header_str = ""
            bibtex_str = ""
            for key in self._provenance():
                if key == "@type":
                    bibtex_header_str += f"@{self._provenance()[key]}{{"
                    if "doi" in self._provenance():
                        bibtex_header_str += f"{self._provenance()['doi']}"
                    else:
                        bibtex_header_str += f"{self.__class__.__name__}"
                    bibtex_header_str += ",\n"
                else:
                    bibtex_str += f"    {key} = {{{self._provenance()[key]}}},\n"
            bibtex_refs.append(bibtex_header_str + bibtex_str + "}\n")

        return bibtex_refs if retscalar else bibtex_refs[0]

    @classmethod
    def from_dict(cls, par_dict : dict):
        """
        par_dict should contain:
            "E0" : energy at E(V0)
            "B0" : bulk modulus at V0
            "B1" : d B / dP (V0)
            "V0" : equilibrium volume
        """
        return cls( par_dict["E0"], par_dict["B0"], par_dict["B1"], par_dict["V0"])
    
    @property
    def as_dict(self):
        """ Return equation of state parameters and name of function  """
        return {
            "e0": self.e0,
            "b0": self.b0,
            "b1": self.b1,
            "v0": self.v0,
            "name": self.__class__.__name__
        }    

    @staticmethod
    def _func(V : Sequence, params : Sequence, der : int = 0):
        """
        The equation of state function. This must be implemented by user.'
        
        der is the order of the derivative d^(der) E / d V^(der)
        e.g., - _func(V,params,der=1) returns the pressure
        """
        return NotImplementedError
        
    def func(self, volume : Sequence, der : int = 0):
        """
        The equation of state function with the parameters other than volume set
        to the ones obtained from fitting.

        Args:
             volume (list/numpy.array)
             der (int), order of derivative

        Returns:
            numpy.array
        """
        return self._func(volume, (self.e0, self.b0, self.b1, self.v0), der=der)

    def Energy(self, volume : Sequence):
        """ Utility function to calculate energy of EOS as function of arbitrary volume """
        return self._func(volume, (self.e0, self.b0, self.b1, self.v0), der=0)

    def Pressure(self, volume : Sequence):
        """ 
        Utility function to calculate pressure P from EOS as function of volume

        P = - d E / dV
        """
        return -self.func(volume,der=1)
    
    def BulkModulus(self, volume : Sequence):
        """ 
        Utility function to calculate bulk modulus B from EOS as function of volume

        B = -V dP / d V = V d^2 E / d V^2 
        """
        return volume*self.func(volume,der=2)
    
    def BulkModulusPrime(self, volume : Sequence):
        """
        Utility function to calculate d B / d P from EOS as function of volume

        B' = d B / d P
           = -( 1 + V^2 / B(V) * d^3 E / d V^3 )
        """
        return -( 1. + volume**2*self.func(volume,der=3)/self.BulkModulus(volume) )        


class EOSBase(EOSCore):
    """
    Abstract class that adds energy vs. volume data fitting to EOSCore
    """
    def __init__(self, volumes : Sequence, energies : Sequence):
        """
        Args:
            volumes (list/numpy.array): volumes in Ang^3
            energies (list/numpy.array): energy in eV
        """
        self.volumes = np.array(volumes)
        self.energies = np.array(energies)
        # minimum energy(e0), buk modulus(b0),
        # derivative of bulk modulus wrt pressure(b1), minimum volume(v0)
        self._params = self._initial_guess()
        # the eos function parameters. It is the same as _params except for
        # equation of states that uses polynomial fits(deltafactor and
        # numerical_eos)
        self.eos_params = self._params.copy()
        return super().__init__(*self.eos_params)

    def _initial_guess(self):
        """
        Fit to E(V) = a + b / V + c / V^2 to determine initial parameters (analytic):
        V0 = -2c/b
        B0 = 2(b V0 + 3 c)/V0^3
        B1 = 5

        Returns:
        tuple: (e0, b0, b1, v0)
        """
        pfp = np.polyfit(1./np.maxmium(1.e-15,self.volumes),self.energies,2)
        V0_init = -2.*pfp[0]/pfp[1]
        E0_init = pfp[2] + pfp[1]/V0_init + pfp[0]/V0_init**2
        B0_init = 2*(pfp[1]*V0_init + 3.*pfp[0])/V0_init**3
        B1_init = 5.

        if not self.volumes.min() <= V0_init and V0_init <= self.volumes.max():
            raise EOSError("The initally-guessed minimum volume is out of the input volume range\n.")

        return E0_init, B0_init, B1_init, V0_init

    def fit(self):
        """
        Do the fitting. Does least square fitting. If you want to use custom
        fitting, must override this.
        """
        # the objective function that will be minimized in the least square
        # fitting
        self._params = self._initial_guess()
        self.eos_params, ierr = leastsq(
            lambda pars : self.energies - self._func(*pars,self.volumes,der=0),
            self._params
        )

        # e0, b0, b1, v0
        self.e0, self.b0, self.b1, self.v0 = self.eos_params
        self._params = self.eos_params.copy()
        
        if ierr not in [1, 2, 3, 4]:
            raise EOSError("Optimal parameters not found")
        
    @property
    def e0(self):
        """
        Returns the min energy.
        """
        return self._params[0]

    @property
    def b0(self):
        """
        Returns the bulk modulus.
        Note: the units for the bulk modulus: unit of energy/unit of volume^3.
        """
        return self._params[1]

    @property
    def b0_GPa(self):
        """
        Returns the bulk modulus in GPa.
        Note: This assumes that the energy and volumes are in eV and Ang^3
            respectively
        """
        return FloatWithUnit(self.b0, "eV ang^-3").to("GPa")

    @property
    def b1(self):
        """
        Returns the derivative of bulk modulus wrt pressure(dimensionless)
        """
        return self._params[2]

    @property
    def v0(self):
        """
        Returns the minimum or the reference volume in Ang^3.
        """
        return self._params[3]


    def plot(self, width=8, height=None, plt=None, dpi=None, **kwargs):
        """
        Plot the equation of state.

        Args:
            width (float): Width of plot in inches. Defaults to 8in.
            height (float): Height of plot in inches. Defaults to width *
                golden ratio.
            plt (matplotlib.pyplot): If plt is supplied, changes will be made
                to an existing plot. Otherwise, a new plot will be created.
            dpi:
            kwargs (dict): additional args fed to pyplot.plot.
                supported keys: style, color, text, label

        Returns:
            Matplotlib plot object.
        """
        # pylint: disable=E1307
        plt = pretty_plot(width=width, height=height, plt=plt, dpi=dpi)

        color = kwargs.get("color", "r")
        label = kwargs.get("label", f"{type(self).__name__} fit")
        lines = [
            f"Equation of State: {type(self).__name__}",
            f"Minimum energy = {self.e0:1.2f} eV",
            f"Minimum or reference volume = {self.v0:1.2f} Ang^3",
            f"Bulk modulus = {self.b0:1.2f} eV/Ang^3 = {self.b0_GPa:1.2f} GPa",
            f"Derivative of bulk modulus wrt pressure = {self.b1:1.2f}",
        ]
        text = "\n".join(lines)
        text = kwargs.get("text", text)

        # Plot input data.
        plt.plot(self.volumes, self.energies, linestyle="None", marker="o", color=color)

        # Plot eos fit.
        vmin, vmax = min(self.volumes), max(self.volumes)
        vmin, vmax = (vmin - 0.01 * abs(vmin), vmax + 0.01 * abs(vmax))
        vfit = np.linspace(vmin, vmax, 100)

        plt.plot(vfit, self.Energy(vfit), linestyle="dashed", color=color, label=label)

        plt.grid(True)
        plt.xlabel("Volume $\\AA^3$")
        plt.ylabel("Energy (eV)")
        plt.legend(loc="best", shadow=True)
        # Add text with fit parameters.
        plt.text(0.4, 0.5, text, transform=plt.gca().transAxes)

        return plt

    @add_fig_kwargs
    def plot_ax(self, ax=None, fontsize=12, **kwargs):
        """
        Plot the equation of state on axis `ax`

        Args:
            ax: matplotlib :class:`Axes` or None if a new figure should be created.
            fontsize: Legend fontsize.
            color (str): plot color.
            label (str): Plot label
            text (str): Legend text (options)

        Returns:
            Matplotlib figure object.
        """
        # pylint: disable=E1307
        ax, fig, plt = get_ax_fig_plt(ax=ax)

        color = kwargs.get("color", "r")
        label = kwargs.get("label", f"{type(self).__name__} fit")
        lines = [
            f"Equation of State: {type(self).__name__}",
            f"Minimum energy = {self.e0:1.2f} eV",
            f"Minimum or reference volume = {self.v0:1.2f} Ang^3",
            f"Bulk modulus = {self.b0:1.2f} eV/Ang^3 = {self.b0_GPa:1.2f} GPa",
            f"Derivative of bulk modulus wrt pressure = {self.b1:1.2f}",
        ]
        text = "\n".join(lines)
        text = kwargs.get("text", text)

        # Plot input data.
        ax.plot(self.volumes, self.energies, linestyle="None", marker="o", color=color)

        # Plot eos fit.
        vmin, vmax = min(self.volumes), max(self.volumes)
        vmin, vmax = (vmin - 0.01 * abs(vmin), vmax + 0.01 * abs(vmax))
        vfit = np.linspace(vmin, vmax, 100)

        ax.plot(vfit, self.Energy(vfit), linestyle="dashed", color=color, label=label)

        ax.grid(True)
        ax.set_xlabel("Volume $\\AA^3$")
        ax.set_ylabel("Energy (eV)")
        ax.legend(loc="best", shadow=True)
        # Add text with fit parameters.
        ax.text(
            0.5,
            0.5,
            text,
            fontsize=fontsize,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

        return fig


class Murnaghan(EOSBase):
    """ Murnaghan EOS """

    @staticmethod
    def _provenance() -> dict:
        return {
            "@type": "article",
            "title" : "First-principles calculation of the equilibrium ground-state properties of transition metals: Applications to Nb and Mo",
            "author" : "Fu, C. -L. and Ho, K. -M.",
            "journal" : "Phys. Rev. B",
            "volume" : "28",
            "issue" : "10",
            "pages" : "5480--5486",
            "year" : "1983",
            "doi" : "10.1103/PhysRevB.28.5480"
        }

    @staticmethod
    def _func(volume: Sequence, params: Sequence, der: int = 0):
        E0, B0, B1, V0 = params
        nu = volume/V0
        b1m1 = B1 - 1.
        prefactor = B0*V0/(B1*b1m1)
        if der == 0:
            return E0 - B0*V0/b1m1 + prefactor * (1./nu**b1m1 + b1m1*nu)
        elif der == 1:
            return B0/B1 * (1. - 1./nu**B1)
        elif der == 2:
            return B0/(V0*nu**(B1+1))
        elif der == 3:
            return B0*(B1+1)/(V0**2*nu**(B1+1))

class Birch(EOSBase):
    """
    Birch or Birch-Euler EOS.
    """

    @staticmethod
    def _func(volume: Sequence, params: Sequence, der : int = 0):
        """
        Birch-Euler EOS and low order derivatives
        """
        E0, B0, B1, V0 = params
        nu = volume/V0
        c0, c2, c4, c6 = (6. - B1, 3*B1 - 16., 14. - 3*B1, B1 - 4.)
        if der == 0:
            return E0 + 9.*B0*V0/16.*(c0 + c2/nu**(2./3.) + c4/nu**(4./3.) + c6/nu**2)
        elif der == 1:
            return -3.*B0/8.*( c2/nu**(5./3.) + 2*c4/nu**(7./3.) + 3*c6/nu**3 )
        elif der == 2:
            return B0/(8.*V0)*( 5*c2/nu**(8./3.) + 14*c4/nu**(10./3.) + 27*c6/nu**4 )
        elif der == 3:
            return -B0/(6.*V0**2)*( 10*c2/nu**(11./3.) + 35*c4/nu**(13./3.) + 81*c6/nu**5 )
    
    @staticmethod
    def _provenance() -> dict:
        """ Birch EOS provenance """
        return dict(
            type = "@article",
            title = "Finite Elastic Strain of Cubic Crystals",
            author = "Birch, Francis",
            journal = "Phys. Rev.",
            volume = "71",
            issue = "11",
            pages = "809--824",
            numpages = "0",
            year = "1947",
            month = "Jun",
            publisher = "American Physical Society",
            doi = "10.1103/PhysRev.71.809",
            url = "https://link.aps.org/doi/10.1103/PhysRev.71.809"
        )

class BirchMurnaghan(EOSBase):
    """ Birch-Murnaghan or Birch-Lagrange EOS """

    @staticmethod
    def _func(volume : Sequence, params: Sequence, der : int = 0):
        """ BM EOS and low order derivatives """
        E0, B0, B1, V0 = params
        nu = volume/V0
        c0, c2, c4, c6 = (B1 + 2., -(4. + 3*B1), 2 + 3*B1, -B1)
        if der == 0:
            return E0 + 9*V0*B0/16.*(c0 + c2*nu**(2./3.) + c4*nu**(4./3.) + c6*nu**2)
        elif der == 1:
            return 3*B0/8.*(c2*nu**(-1./3.) + 2*c4*nu**(1./3.) + 3*c6*nu)
        elif der == 2:
            en = B0/(8.*V0)*(-c2*nu**(-4./3.) + 2*c4*nu**(-2./3.) + 9*c6)
        elif der == 3:
            en = B0/(6.*V0**2)*(c2*nu**(-7./3.) - c4*nu**(-5./3.))
        return en
    
    @staticmethod
    def _provenance() -> dict:
        """ Birch EOS provenance """
        return dict(
            type = "@article",
            title = "Finite Elastic Strain of Cubic Crystals",
            author = "Birch, Francis",
            journal = "Phys. Rev.",
            volume = "71",
            issue = "11",
            pages = "809--824",
            numpages = "0",
            year = "1947",
            month = "Jun",
            publisher = "American Physical Society",
            doi = "10.1103/PhysRev.71.809",
            url = "https://link.aps.org/doi/10.1103/PhysRev.71.809"
        )


class PourierTarantola(EOSBase):
    """
    Pourier-Tarantola EOS
    """

    @staticmethod
    def _func(volume : Sequence, params: Sequence, der = 0):
        """
        Pourier-Tarantola EOS and low order derivatives
        """
        E0, B0, B1, V0 = params
        nu = volume/V0
        lnnu = np.log(nu)
        if der == 0:
            return E0 + B0*V0/6.*lnnu**2*(3. + (2. - B1)*lnnu)
        elif der == 1:
            return B0*lnnu/(3.*nu)*(2. + (2. - B1)*lnnu)
        elif der == 2:
            return B0/(3*V0*nu**2)*(2. + 2.*(1. - B1)*lnnu + (B1 - 2.)*lnnu**2)
        elif der == 3:
            return 2*B0/(3*V0**2*nu**3)*(1. + B1 + (3*B1 - 4.)*lnnu + (B1 - 2.)*lnnu**2)

    @staticmethod
    def _provenance() -> dict:
        """ PT provenance """
        return dict(
            type = "@article",
            title = "A logarithmic equation of state",
            journal = "Phys. Earth and Planet. Inter.",
            volume = "109",
            number = "1",
            pages = "1-8",
            year = "1998",
            issn = "0031-9201",
            doi = "https://doi.org/10.1016/S0031-9201(98)00112-5",
            url = "https://www.sciencedirect.com/science/article/pii/S0031920198001125",
            author = "J.-P Poirier and A Tarantola",
            keywords = "Finite strain, Density, Pressure, Equation of state",
            abstract = "The isothermal Eulerian Birch–Murnaghan equation of state is currently used in geophysics, despite its recognized shortcomings for very large compressive strains. We propose an equation of state constructed from the Hencky logarithmic strain, rather than from the Eulerian strain. The logarithmic equation of state has a simple expression and is valid in a greater range of pressures than the Birch–Murnaghan equation of state."
        )

class Vinet(EOSBase):
    """
    Vinet EOS.
    """

    @staticmethod
    def _func(volume : Sequence, params: Sequence, der = 0):
        """
        Vinet equation from PRB 70, 224107
        """
        E0, B0, B1, V0 = params
        A = 4*B0*V0/(B1 - 1.)**2
        B = 3*(B1 - 1)/2.
        t = B*((volume/V0)**(1./3.) - 1.)
        expt = np.exp(-t)

        if der > 0:
            """
            For t = b[ (V/V0)**(1/3) - 1 ],
            d/dV = b**3/[ 3 V0 (C + t)**2 ] d/dt
            """
            cfac = B**3/(3*V0)
            dedt = A*t*expt
        if der > 1:
            dedt_2 = A*(1. - t)*expt

        if der == 0:
            return E0 + A*(1. - (1. - t)*expt )
        elif der == 1:
            return cfac*dedt
        elif der == 2:
            return cfac**2/(B + t)**5*( - 2*dedt + (B + t)*dedt_2 )
        elif der == 3:
            dedt_3 = A*(t - 2.)*expt
            return cfac**3/(B + t)**8* ( 10*dedt - 6*(B + t)*dedt_2 + (B + t)**2*dedt_3 )
    
    @staticmethod
    def _provenance() -> dict:
        return dict(
            type = "@article",
            author = "Vinet, P. and Ferrante, J. and Rose, J. H. and Smith, J. R.",
            title = "Compressibility of solids",
            journal = "J. Geophys. Research: Solid Earth",
            volume = "92",
            number = "B9",
            pages = "9319-9325",
            doi = "https://doi.org/10.1029/JB092iB09p09319",
            url = "https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JB092iB09p09319",
            eprint = "https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/JB092iB09p09319",
            abstract = "We have discovered that the isothermal equation of state for solids in compression has a simple, universal form. This single form is shown to accurately describe the pressure and bulk modulus as a function of volume for ionic, metallic, covalent, and rare gas solids.",
            year = "1987"
        )

class MieGruneisen(EOSBase):
    """ Mie-Gruneisen EOS """
    @staticmethod
    def _func(volume : Sequence, params: Sequence, der = 0):
        """
        Vinet equation from PRB 70, 224107
        """
        E0, B0, B1, V0 = params
        eta = (volume/V0)**(1./3.)
        c = 3*B1 - 7.
        if der == 0:
            return E0 + 9*B0*V0/c - 9*B0*V0/(c - 1.)*(1./eta - 1./(c*eta**c))
        elif der == 1:
            return 3*B0/(c - 1.)*(1./eta**4 - 1./eta**(c + 3))
        elif der == 2:
            return -B0/(V0*(c - 1.))*(4./eta**7 - (c+3.)/eta**(c + 6))
        elif der == 3:
            return B0/(3*V0**2*(c - 1.))*(28./eta**7 - (c+3.)*(c+6.)/eta**(c + 9))
    
    @staticmethod
    def _provenance() -> list[dict]:
        mie_ref = dict(
            type = "@article",
            author = "Mie, Gustav",
            title = "Zur kinetischen Theorie der einatomigen Körper",
            journal = "Annalen der Physik",
            volume = "316",
            number = "8",
            pages = "657-697",
            doi = "https://doi.org/10.1002/andp.19033160802",
            url = "https://onlinelibrary.wiley.com/doi/abs/10.1002/andp.19033160802",
            year = "1903"
        )
        grun_ref = dict(
            type = "@article",
            author = "Grüneisen, E.",
            title = "Theorie des festen Zustandes einatomiger Elemente",
            journal = "Ann. der Phys.",
            volume = "344",
            number = "12",
            pages = "257-306",
            doi = "https://doi.org/10.1002/andp.19123441202",
            url = "https://onlinelibrary.wiley.com/doi/abs/10.1002/andp.19123441202",
            year = "1912"
        )
        return [mie_ref,grun_ref]

class Tait(EOSBase):
    """ Tait EOS """

    @staticmethod
    def _func(volume: Sequence, params: Sequence, der: int = 0):
        E0, B0, B1, V0 = params
        nu = V/V0
        C = 1. + B1
        expf = np.exp(C*(1. - nu))
        if der == 0:
            return E0 + B0*V0/C * (nu - 1. + (expf - 1.)/C)
        elif der == 1:
            return B0/C*(1. - expf)
        elif der == 2:
            return B0*expf/V0
        elif der == 3:
            return -B0*C*expf/V0**2
    
    @staticmethod
    def _provenance() -> dict:
        return dict(
            type = "@article",
            author = "Dymond, J. H. and Malhotra, R.",
            title = "The Tait equation: 100 years on",
            journal = "Int. J. Thermophys.",
            volume = "9",
            pages = "941--951",
            year = "1988",
            doi = "10.1007/BF01133262"
        )


class SJEOS(EOSBase):
    """ Stabilized Jellium EOS """
    
    @staticmethod
    def _func(volume: Sequence, params: Sequence, der: int = 0):

        """
        Eqs. (24) and (30)-(33) of the SJEOS paper
        """
        E0, B0, B1, V0 = params
        nu = V/V0
        A = 9.*B0*V0*(B1 - 3.)/2.
        B = 9.*B0*V0*(10. - 3*B1)/2.
        C = -9.*B0*V0*(11. - 3*B1)/2.
        D = E0 + 9.*B0*V0*(4. - B1)/2.
        if der == 0:
            return A/nu + B/nu**(2./3.) + C/nu**(1./3.) + D
        elif der == 1:
            return -(3*A/nu**2 + 2*B/nu**(5./3.) + C/nu**(4./3.))/(3.*V0)
        elif der == 2:
            return 2*(9*A/nu**3 + 5*B/nu**(8./3.) + 2*C/nu**(7./3.))/(3.*V0)**2
        elif der == 3:
            return -2*(81*A/nu**4 + 40*B/nu**(11./3.) + 14*C/nu**(10./3.))/(3.*V0)
    
    @staticmethod
    def _provenance() -> dict:
        return dict(
            type = "@article",
            title = "Energy and pressure versus volume: Equations of state motivated by the stabilized jellium model",
            author = "Alchagirov, Alim B. and Perdew, John P. and Boettger, Jonathan C. and Albers, R. C. and Fiolhais, Carlos",
            journal = "Phys. Rev. B",
            volume = "63",
            issue = "22",
            pages = "224115",
            numpages = "16",
            year = "2001",
            month = "May",
            publisher = "American Physical Society",
            doi = "10.1103/PhysRevB.63.224115",
            url = "https://link.aps.org/doi/10.1103/PhysRevB.63.224115"
        )


class PolynomialEOS(EOSBase):
    """
    Derives from EOSBase. Polynomial based equations of states must subclass
    this.
    """

    @staticmethod
    def _func(volume : Sequence, params : Sequence, der : int = 0, alpha : float = 1):
        poly = np.poly1d(list(params))

        arg = volume**alpha
        if der == 0:
            return poly(arg)

        if alpha == 1:
            return poly.deriv(m=der)(arg)

        if der == 1:
            return alpha*volume**(alpha - 1)*poly.deriv(m=1)(arg)
        elif der == 2:
            return alpha*( 
                alpha*volume**(2*(alpha - 1))*poly.deriv(m=2)(arg)
                + (alpha - 1)*volume**(alpha - 2)*poly.deriv(m=1)(arg)
            )
        elif der == 3:
            return alpha*(
                alpha**2*volume**(3*(alpha - 1))*poly.deriv(m=3)(arg)
                + (
                    2*alpha*(alpha - 1)*volume**(2*alpha - 3)
                    + (alpha - 1)**2*volume**((alpha-1)*(alpha-2))
                )*poly.deriv(m=2)(arg)
                + (alpha - 1)*(alpha-2)*volume**(alpha - 3)*poly.deriv(m=1)(arg)
            )

    def fit(self, order):
        """
        Do polynomial fitting and set the parameters. Uses numpy polyfit.

        Args:
             order (int): order of the fit polynomial
        """
        self.eos_params = np.polyfit(self.volumes, self.energies, order)
        self._set_params()

    def _set_params(self):
        """
        Use the fit polynomial to compute the parameter e0, b0, b1 and v0
        and set to the _params attribute.
        """
        fit_poly = np.poly1d(self.eos_params)
        # the volume at min energy, used as the initial guess for the
        # optimization wrt volume.
        v_e_min = self.volumes[np.argmin(self.energies)]
        # evaluate e0, v0, b0 and b1
        min_wrt_v = minimize(fit_poly, v_e_min)
        e0, v0 = min_wrt_v.fun, min_wrt_v.x[0]
        pderiv2 = np.polyder(fit_poly, 2)
        pderiv3 = np.polyder(fit_poly, 3)
        b0 = v0 * np.poly1d(pderiv2)(v0)
        db0dv = np.poly1d(pderiv2)(v0) + v0 * np.poly1d(pderiv3)(v0)
        # db/dp
        b1 = -v0 * db0dv / b0
        self._params = [e0, b0, b1, v0]


class DeltaFactor(PolynomialEOS):
    """
    Fitting a polynomial EOS using delta factor.
    """

    @staticmethod
    def _func(volume : Sequence, params : Sequence, der : int = 0, alpha : float = 2./3.):
        return super()._func(volume, params, der=der, alpha = alpha)

    def fit(self, order=3):
        """
        Overridden since this eos works with volume**(2/3) instead of volume.
        """
        x = self.volumes ** (-2./3.)
        self.eos_params = np.polyfit(x, self.energies, order)
        self._set_params()

    def _set_params(self):
        """
        Overridden to account for the fact the fit with volume**(2/3) instead
        of volume.
        """
        deriv0 = np.poly1d(self.eos_params)
        deriv1 = np.polyder(deriv0, 1)
        deriv2 = np.polyder(deriv1, 1)
        deriv3 = np.polyder(deriv2, 1)

        for x in np.roots(deriv1):
            if x > 0 and deriv2(x) > 0:
                v0 = x ** (-3 / 2.0)
                break
        else:
            raise EOSError("No minimum could be found")

        derivV2 = 4 / 9 * x**5 * deriv2(x)
        derivV3 = -20 / 9 * x ** (13 / 2.0) * deriv2(x) - 8 / 27 * x ** (15 / 2.0) * deriv3(x)
        b0 = derivV2 / x ** (3 / 2.0)
        b1 = -1 - x ** (-3 / 2.0) * derivV3 / derivV2

        # e0, b0, b1, v0
        self._params = [deriv0(v0 ** (-2 / 3.0)), b0, b1, v0]


class NumericalEOS(PolynomialEOS):
    """
    A numerical EOS.
    """

    def fit(self, min_ndata_factor=3, max_poly_order_factor=5, min_poly_order=2):
        """
        Fit the input data to the 'numerical eos', the equation of state employed
        in the quasiharmonic Debye model described in the paper:
        10.1103/PhysRevB.90.174107.

        credits: Cormac Toher

        Args:
            min_ndata_factor (int): parameter that controls the minimum number
                of data points that will be used for fitting.
                minimum number of data points =
                    total data points-2*min_ndata_factor
            max_poly_order_factor (int): parameter that limits the max order
                of the polynomial used for fitting.
                max_poly_order = number of data points used for fitting -
                                 max_poly_order_factor
            min_poly_order (int): minimum order of the polynomial to be
                considered for fitting.
        """
        warnings.simplefilter("ignore", np.RankWarning)

        def get_rms(x, y):
            return np.sqrt(np.sum((np.array(x) - np.array(y)) ** 2) / len(x))

        # list of (energy, volume) tuples
        e_v = list(zip(self.energies, self.volumes))
        ndata = len(e_v)
        # minimum number of data points used for fitting
        ndata_min = max(ndata - 2 * min_ndata_factor, min_poly_order + 1)
        rms_min = np.inf
        # number of data points available for fit in each iteration
        ndata_fit = ndata
        # store the fit polynomial coefficients and the rms in a dict,
        # where the key=(polynomial order, number of data points used for
        # fitting)
        all_coeffs = {}

        # sort by energy
        e_v = sorted(e_v, key=lambda x: x[0])
        # minimum energy tuple
        e_min = e_v[0]
        # sort by volume
        e_v = sorted(e_v, key=lambda x: x[1])
        # index of minimum energy tuple in the volume sorted list
        emin_idx = e_v.index(e_min)
        # the volume lower than the volume corresponding to minimum energy
        v_before = e_v[emin_idx - 1][1]
        # the volume higher than the volume corresponding to minimum energy
        v_after = e_v[emin_idx + 1][1]
        e_v_work = deepcopy(e_v)

        # loop over the data points.
        while (ndata_fit >= ndata_min) and (e_min in e_v_work):
            max_poly_order = ndata_fit - max_poly_order_factor
            e = [ei[0] for ei in e_v_work]
            v = [ei[1] for ei in e_v_work]
            # loop over polynomial order
            for i in range(min_poly_order, max_poly_order + 1):
                coeffs = np.polyfit(v, e, i)
                pder = np.polyder(coeffs)
                a = np.poly1d(pder)(v_before)
                b = np.poly1d(pder)(v_after)
                if a * b < 0:
                    rms = get_rms(e, np.poly1d(coeffs)(v))
                    rms_min = min(rms_min, rms * i / ndata_fit)
                    all_coeffs[(i, ndata_fit)] = [coeffs.tolist(), rms]
                    # store the fit coefficients small to large,
                    # i.e a0, a1, .. an
                    all_coeffs[(i, ndata_fit)][0].reverse()
            # remove 1 data point from each end.
            e_v_work.pop()
            e_v_work.pop(0)
            ndata_fit = len(e_v_work)

        logger.info(f"total number of polynomials: {len(all_coeffs)}")

        norm = 0.0
        fit_poly_order = ndata
        # weight average polynomial coefficients.
        weighted_avg_coeffs = np.zeros((fit_poly_order,))

        # combine all the filtered polynomial candidates to get the final fit.
        for k, v in all_coeffs.items():
            # weighted rms = rms * polynomial order / rms_min / ndata_fit
            weighted_rms = v[1] * k[0] / rms_min / k[1]
            weight = np.exp(-(weighted_rms**2))
            norm += weight
            coeffs = np.array(v[0])
            # pad the coefficient array with zeros
            coeffs = np.lib.pad(coeffs, (0, max(fit_poly_order - len(coeffs), 0)), "constant")
            weighted_avg_coeffs += weight * coeffs

        # normalization
        weighted_avg_coeffs /= norm
        weighted_avg_coeffs = weighted_avg_coeffs.tolist()
        # large to small(an, an-1, ..., a1, a0) as expected by np.poly1d
        weighted_avg_coeffs.reverse()

        self.eos_params = weighted_avg_coeffs
        self._set_params()


class EOS:
    """
    Convenient wrapper. Retained in its original state to ensure backward
    compatibility.

    Fit equation of state for bulk systems.

    The following equations are supported::

        murnaghan: PRB 28, 5480 (1983)

        birch: Intermetallic compounds: Principles and Practice, Vol I:
            Principles. pages 195-210

        birch_murnaghan: PRB 70, 224107

        pourier_tarantola: PRB 70, 224107

        vinet: PRB 70, 224107

        deltafactor

        numerical_eos: 10.1103/PhysRevB.90.174107.

    Usage::

       eos = EOS(eos_name='murnaghan')
       eos_fit = eos.fit(volumes, energies)
       eos_fit.plot()
    """

    MODELS = {
        "murnaghan": Murnaghan,
        "birch": Birch,
        "birch_murnaghan": BirchMurnaghan,
        "pourier_tarantola": PourierTarantola,
        "vinet": Vinet,
        "mie_gruneisen": MieGruneisen,
        "tait": Tait,
        "sjeos": SJEOS,
        "deltafactor": DeltaFactor,
        "numerical_eos": NumericalEOS,
    }

    def __init__(self, eos_name="murnaghan"):
        """
        Args:
            eos_name (str): Type of EOS to fit.
        """
        if eos_name not in self.MODELS:
            raise EOSError(
                f"The equation of state {eos_name!r} is not supported. "
                f"Please choose one from the following list: {list(self.MODELS)}"
            )
        self._eos_name = eos_name
        self.model = self.MODELS[eos_name]

    def fit(self, volumes, energies):
        """
        Fit energies as function of volumes.

        Args:
            volumes (list/np.array)
            energies (list/np.array)

        Returns:
            EOSBase: EOSBase object
        """
        eos_fit = self.model(np.array(volumes), np.array(energies))
        eos_fit.fit()
        return eos_fit


class EOSError(Exception):
    """
    Error class for EOS fitting.
    """

if __name__ == "__main__":
    
    env = np.array([[33.51118,-3.5334649],[34.13374,-3.5367674],
    [34.78934,-3.5394347],[35.42758,-3.5413234],[36.09958,-3.5425646],
    [36.78003,-3.5431276],[37.44232,-3.5430619],[38.13948,-3.5423918],
    [38.81795,-3.5411959],[39.53203,-3.5393638],[40.22686,-3.5371432],
    [40.95807,-3.5343387]])

    MEOS = Murnaghan(env[:,0],env[:,1])
    print(MEOS.provenance())
    BEOS = Birch(env[:,0],env[:,1])
    VEOS = Vinet(env[:,0],env[:,1])
    SJEOS = SJEOS(env[:,0],env[:,1])
    for _eos in [MEOS,BEOS,VEOS,SJEOS]:
        _eos.fit()

    print("Murnaghan: ", MEOS.as_dict())
    print("Birch: ", BEOS.as_dict())
    print("Vinet: ", VEOS.as_dict())
    print("SJEOS: ", SJEOS.as_dict())
    
    import matplotlib.pyplot as plt
    vl = np.linspace(env[:,0].min(),env[:,0].max(),5000)
    plt.plot(vl,MEOS.Energy(vl))
    plt.plot(vl,BEOS.Energy(vl),linestyle="--")
    plt.plot(vl,VEOS.Energy(vl),linestyle="-.")
    plt.plot(vl,SJEOS.Energy(vl),linestyle=":")
    plt.scatter(env[:,0],env[:,1])
    plt.show()
