"""Disk Element module.

This module defines the DiskElement classes which will be used to represent equipments
attached to the rotor shaft, which add mainly mass and inertia to the system.
There're 2 options, an element with 4 or 6 degrees of freedom.
"""

import numpy as np
from sympy.solvers import solve
from sympy import var, Eq 
import toml
from plotly import graph_objects as go

from ross.element import Element
from ross.units import check_units
from ross.utils import read_table_file
from ross.materials import steel

__all__ = ["DiskElement", "DiskElement6DoF"]


class DiskElement(Element):
    """A disk element.

    This class creates a disk element from input data of inertia and mass.

    Parameters
    ----------
    n: int
        Node in which the disk will be inserted.
    m : float, pint.Quantity
        Mass of the disk element.
    Id : float, pint.Quantity
        Diametral moment of inertia.
    Ip : float, pint.Quantity
        Polar moment of inertia
    material: ross.Material, optional
        Disk material. Default is steel.
    di : float, pint.Quantity, optional
        Inner diameter. If not provided, it will be calculated from mass and inertia.
    do : float, pint.Quantity, optional
        Outer diameter. If not provided, it will be calculated from mass and inertia.
    width : float, pint.Quantity, optional
        Disk width. If not provided, it will be calculated from mass and inertia
    tag : str, optional
        A tag to name the element
        Default is None
    scale_factor: float or str, optional
        The scale factor is used to scale the disk drawing.
        For disks it is also possible to provide 'mass' as the scale factor.
        In this case the code will calculate scale factors for each disk based
        on the disk with the higher mass. Notice that in this case you have to
        create all disks with the scale_factor='mass'.
        Default is 1.
    color : str, optional
        A color to be used when the element is represented.
        Default is 'Firebrick'.

    Examples
    --------
    >>> disk = DiskElement(n=0, m=32, Id=0.2, Ip=0.3)
    >>> disk.Ip
    0.3
    """

    @check_units
    def __init__(self, n, m, Id, Ip, material=steel, di=None, do=None, width=None, tag=None, scale_factor=1.0, color="Firebrick"):
        self.n = int(n)
        self.n_l = n
        self.n_r = n

        self.m = float(m)
        self.Id = float(Id)
        self.Ip = float(Ip)
        self.tag = tag
        self.color = color
        self.scale_factor = scale_factor
        self.dof_global_index = None
        self.material = material
        if any([el is None for el in [di, do, width]]):
            print("Calculating disk geometry from mass and inertia...")
            d_i, d_o, w = var('d_i, d_o, w', real=True, positive=True)
            eq1 = Eq(1/8 * m * (d_o**2 + d_i**2), Ip)
            eq2 = Eq(self.material.rho * np.pi * w * (d_o**2 - d_i**2) / 4, m)
            eq3 = Eq(1/2 * Ip + 1/12 * m * w**2, Id)
            eq4 = d_o > d_i
            sol = solve([eq1, eq2, eq3], [d_i, d_o, w], dict=True)
            if len(sol) == 0:
                raise ValueError("Impossible to calculate disk geometry with the given parameters.")
            elif len(sol) > 1:
                raise ValueError("Multiple solutions found when calculating disk geometry.")
            di = sol[0][d_i]
            do = sol[0][d_o]
            width = sol[0][w]
        self.di = di
        self.do = do
        self.w = width


    def __eq__(self, other):
        """Equality method for comparasions.

        Parameters
        ----------
        other: object
            The second object to be compared with.

        Returns
        -------
        bool
            True if the comparison is true; False otherwise.

        Examples
        --------
        >>> disk1 = disk_example()
        >>> disk2 = disk_example()
        >>> disk1 == disk2
        True
        """
        false_number = 0
        if "DiskElement" in other.__class__.__name__:
            for i in self.__dict__:
                try:
                    if np.allclose(self.__dict__[i], other.__dict__[i]):
                        pass
                    else:
                        false_number += 1

                except TypeError:
                    if self.__dict__[i] == other.__dict__[i]:
                        pass
                    else:
                        false_number += 1

                except KeyError:
                    false_number += 1
        else:
            false_number += 1

        if false_number == 0:
            return True
        else:
            return False

    def __repr__(self):
        """Return a string representation of a disk element.

        Returns
        -------
        A string representation of a disk element object.

        Examples
        --------
        >>> disk = disk_example()
        >>> disk # doctest: +ELLIPSIS
        DiskElement(Id=0.17809, Ip=0.32956...
        """
        return (
            f"{self.__class__.__name__}"
            f"(Id={self.Id:{0}.{5}}, Ip={self.Ip:{0}.{5}}, "
            f"m={self.m:{0}.{5}}, color={self.color!r}, "
            f"n={self.n}, scale_factor={self.scale_factor}, tag={self.tag!r})"
            f"di={self.di:{0}.{5}}, do={self.do:{0}.{5}}, width={self.w:{0}.{5}}"
        )

    def __str__(self):
        """Convert object into string.

        Returns
        -------
        The object's parameters translated to strings

        Example
        -------
        >>> print(DiskElement(n=0, m=32, Id=0.223, Ip=0.31223, tag="Disk"))
        Tag:                      Disk
        Node:                     0
        Mass           (kg):      32.0
        Diam. inertia  (kg*m**2): 0.223
        Polar. inertia (kg*m**2): 0.31223
        di             (m):       0.0
        do             (m):       0.0
        width          (m):       0.0
        """
        return (
            f"Tag:                      {self.tag}"
            f"\nNode:                     {self.n}"
            f"\nMass           (kg):      {self.m:{2}.{5}}"
            f"\nDiam. inertia  (kg*m**2): {self.Id:{2}.{5}}"
            f"\nPolar. inertia (kg*m**2): {self.Ip:{2}.{5}}"
            f"\ndi             (m):       {self.di:{2}.{5}}"
            f"\ndo             (m):       {self.do:{2}.{5}}"
            f"\nwidth          (m):       {self.w:{2}.{5}}"
        )

    def __hash__(self):
        return hash(self.tag)

    def dof_mapping(self):
        """Degrees of freedom mapping.

        Returns a dictionary with a mapping between degree of freedom and its index.

        Returns
        -------
        dof_mapping : dict
            A dictionary containing the degrees of freedom and their indexes.

        Examples
        --------
        The numbering of the degrees of freedom for each node.

        Being the following their ordering for a node:

        x_0 - horizontal translation
        y_0 - vertical translation
        alpha_0 - rotation around horizontal
        beta_0  - rotation around vertical

        >>> disk = disk_example()
        >>> disk.dof_mapping()
        {'x_0': 0, 'y_0': 1, 'alpha_0': 2, 'beta_0': 3}
        """
        return dict(x_0=0, y_0=1, alpha_0=2, beta_0=3)

    def M(self):
        """Mass matrix for an instance of a disk element.

        This method will return the mass matrix for an instance of a disk element.

        Returns
        -------
        M : np.ndarray
            A matrix of floats containing the values of the mass matrix.

        Examples
        --------
        >>> disk = DiskElement(0, 32.58972765, 0.17808928, 0.32956362)
        >>> disk.M()
        array([[32.58972765,  0.        ,  0.        ,  0.        ],
               [ 0.        , 32.58972765,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.17808928,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.17808928]])
        """
        m = self.m
        Id = self.Id
        # fmt: off
        M = np.array([[m, 0,  0,  0],
                      [0, m,  0,  0],
                      [0, 0, Id,  0],
                      [0, 0,  0, Id]])
        # fmt: on
        return M

    def K(self):
        """Stiffness matrix for an instance of a disk element.

        This method will return the stiffness matrix for an instance of a disk
        element.

        Returns
        -------
        K : np.ndarray
            A matrix of floats containing the values of the stiffness matrix.

        Examples
        --------
        >>> disk = disk_example()
        >>> disk.K()
        array([[0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]])
        """
        K = np.zeros((4, 4))

        return K

    def C(self):
        """Damping matrix for an instance of a disk element.

        This method will return the damping matrix for an instance of a disk
        element.

        Returns
        -------
        C : np.ndarray
            A matrix of floats containing the values of the damping matrix.

        Examples
        --------
        >>> disk = disk_example()
        >>> disk.C()
        array([[0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]])
        """
        C = np.zeros((4, 4))

        return C

    def G(self):
        """Gyroscopic matrix for an instance of a disk element.

        This method will return the gyroscopic matrix for an instance of a disk
        element.

        Returns
        -------
        G: np.ndarray
            Gyroscopic matrix for the disk element.

        Examples
        --------
        >>> disk = DiskElement(0, 32.58972765, 0.17808928, 0.32956362)
        >>> disk.G()
        array([[ 0.        ,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.32956362],
               [ 0.        ,  0.        , -0.32956362,  0.        ]])
        """
        Ip = self.Ip
        # fmt: off
        G = np.array([[0, 0,   0,  0],
                      [0, 0,   0,  0],
                      [0, 0,   0, Ip],
                      [0, 0, -Ip,  0]])
        # fmt: on
        return G

    def _patch(self, position, fig):
        """Disk element patch.

        Patch that will be used to draw the shaft element using plotly library.

        Parameters
        ----------
        position : float
            Position in which the patch will be drawn.
        fig : plotly.graph_objects.Figure
            The figure object which traces are added on.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The figure object which traces are added on.
        """
        zpos, ypos, yc_pos, scale_factor = position
        if scale_factor is None:
            z_upper = [zpos, zpos + self.w / 2, zpos - self.w / 2, zpos]
            y_upper = [y_pos, y_pos + self.do / 2, y_pos + self.do / 2, y_pos]
            z_lower = z_upper
            y_lower = [-y for y in y_upper]
        else:
            radius = scale_factor / 8

            # coordinates to plot disks elements
            z_upper = [zpos, zpos + scale_factor / 25, zpos - scale_factor / 25, zpos]
            y_upper = [ypos, ypos + 2 * scale_factor, ypos + 2 * scale_factor, ypos]

            z_lower = [zpos, zpos + scale_factor / 25, zpos - scale_factor / 25, zpos]
            y_lower = [-ypos, -ypos - 2 * scale_factor, -ypos - 2 * scale_factor, -ypos]

        z_pos = z_upper
        z_pos.append(None)
        z_pos.extend(z_lower)

        y_pos = y_upper
        y_upper.append(None)
        y_pos.extend(y_lower)

        customdata = [self.n, self.Ip, self.Id, self.m]
        hovertemplate = (
            f"Disk Node: {customdata[0]}<br>"
            + f"Polar Inertia: {customdata[1]:.3e}<br>"
            + f"Diametral Inertia: {customdata[2]:.3e}<br>"
            + f"Disk mass: {customdata[3]:.3f}<br>"
        )

        fig.add_trace(
            go.Scatter(
                x=z_pos,
                y=[y + yc_pos if y is not None else None for y in y_pos],
                customdata=[customdata] * len(z_pos),
                text=hovertemplate,
                mode="lines",
                fill="toself",
                fillcolor=self.color,
                opacity=0.8,
                line=dict(width=2.0, color=self.color),
                showlegend=False,
                name=self.tag,
                legendgroup="disks",
                hoveron="points+fills",
                hoverinfo="text",
                hovertemplate=hovertemplate,
                hoverlabel=dict(bgcolor=self.color),
            )
        )

        fig.add_shape(
            dict(
                type="circle",
                xref="x",
                yref="y",
                x0=zpos - radius,
                y0=y_upper[1] - radius + yc_pos,
                x1=zpos + radius,
                y1=y_upper[1] + radius + yc_pos,
                fillcolor=self.color,
                line_color=self.color,
            )
        )
        fig.add_shape(
            dict(
                type="circle",
                xref="x",
                yref="y",
                x0=zpos - radius,
                y0=y_lower[1] - radius + yc_pos,
                x1=zpos + radius,
                y1=y_lower[1] + radius + yc_pos,
                fillcolor=self.color,
                line_color=self.color,
            )
        )

        return fig

    @classmethod
    @check_units
    def from_geometry(
        cls, n, material, width, i_d, o_d, tag=None, scale_factor=1.0, color="Firebrick"
    ):
        """Create a disk element from geometry properties.

        This class method will create a disk element from geometry data.
        Properties are calculated as per :cite:`friswell2010dynamics`, appendix 1
        for a hollow cylinder:

        Mass:

        :math:`m = \\rho \\pi w (d_o^2 - d_i^2) / 4`

        Polar moment of inertia:

        :math:`I_p = m (d_o^2 + d_i^2) / 8`

        Diametral moment of inertia:

        :math:`I_d = \\frac{1}{2} I_p + \\frac{1}{12} m w^2`

        Where :math:`\\rho` is the material density, :math:`w` is the disk width,
        :math:`d_o` is the outer diameter and :math:`d_i` is the inner diameter.

        Parameters
        ----------
        n : int
            Node in which the disk will be inserted.
        material: ross.Material
             Disk material.
        width : float, pint.Quantity
            The disk width.
        i_d : float, pint.Quantity
            Inner diameter.
        o_d : float, pint.Quantity
            Outer diameter.
        tag : str, optional
            A tag to name the element
            Default is None
        scale_factor: float, optional
            The scale factor is used to scale the disk drawing.
            Default is 1.
        color : str, optional
            A color to be used when the element is represented.
            Default is 'Firebrick' (Cardinal).

        Attributes
        ----------
        m : float
            Mass of the disk element.
        Id : float
            Diametral moment of inertia.
        Ip : float
            Polar moment of inertia

        Examples
        --------
        >>> from ross.materials import steel
        >>> disk = DiskElement.from_geometry(0, steel, 0.07, 0.05, 0.28)
        >>> disk.Ip
        0.32956362089137037
        """
        m = material.rho * np.pi * width * (o_d**2 - i_d**2) / 4
        Ip = m * (o_d**2 + i_d**2) / 8
        Id = 1 / 2 * Ip + 1 / 12 * m * width**2

        tag = tag

        return cls(n, m, Id, Ip, 
                   material=material, 
                   do=o_d, d_i=i_d, 
                   width=width, tag=tag, 
                   scale_factor=scale_factor, color=color)

    @classmethod
    def from_table(cls, file, sheet_name=0, tag=None, scale_factor=None, color=None):
        """Instantiate one or more disks using inputs from an Excel table.

        A header with the names of the columns is required. These names should
        match the names expected by the routine (usually the names of the
        parameters, but also similar ones). The program will read every row
        bellow the header until they end or it reaches a NaN.

        Parameters
        ----------
        file : str
            Path to the file containing the disk parameters.
        sheet_name : int or str, optional
            Position of the sheet in the file (starting from 0) or its name.
            If none is passed, it is assumed to be the first sheet in the file.
        tag_list : list, optional
            list of tags for the disk elements.
            Default is None
        scale_factor: list, optional
            List of scale factors for the disk elements patches.
            The scale factor is used to scale the disk drawing.
            Default is 1.
        color : list, optional
            A color to be used when the element is represented.
            Default is 'Firebrick'.

        Returns
        -------
        disk : list
            A list of disk objects.

        Examples
        --------
        >>> import os
        >>> file_path = os.path.dirname(os.path.realpath(__file__)) + '/tests/data/shaft_si.xls'
        >>> list_of_disks = DiskElement.from_table(file_path, sheet_name="More")
        >>> list_of_disks[0]
        DiskElement(Id=0.0, Ip=0.0, m=15.12, color='Firebrick', n=3, scale_factor=1, tag=None)
        """
        parameters = read_table_file(file, "disk", sheet_name=sheet_name)
        if tag is None:
            tag = [None] * len(parameters["n"])
        if scale_factor is None:
            scale_factor = [1] * len(parameters["n"])
        if color is None:
            color = ["Firebrick"] * len(parameters["n"])

        list_of_disks = []
        for i in range(0, len(parameters["n"])):
            list_of_disks.append(
                cls(
                    n=parameters["n"][i],
                    m=parameters["m"][i],
                    Id=float(parameters["Id"][i]),
                    Ip=float(parameters["Ip"][i]),
                    tag=tag[i],
                    scale_factor=scale_factor[i],
                    color=color[i],
                )
            )
        return list_of_disks


class DiskElement6DoF(DiskElement):
    """A disk element for 6 DoFs.

    This class will create a disk element with 6 DoF from input data of inertia and
    mass.

    Parameters
    ----------
    n: int
        Node in which the disk will be inserted.
    m : float, pint.Quantity
        Mass of the disk element.
    Id : float, pint.Quantity
        Diametral moment of inertia.
    Ip : float, pint.Quantity
        Polar moment of inertia
    tag : str, optional
        A tag to name the element
        Default is None
    scale_factor: float, optional
        The scale factor is used to scale the disk drawing.
        Default is 1.
    color : str, optional
        A color to be used when the element is represented.
        Default is 'Firebrick'.

    Examples
    --------
    >>> disk = DiskElement6DoF(n=0, m=32, Id=0.2, Ip=0.3)
    >>> disk.Ip
    0.3
    """

    def dof_mapping(self):
        """Degrees of freedom mapping.

        Returns a dictionary with a mapping between degree of freedom and its index.

        Returns
        -------
        dof_mapping : dict
            A dictionary containing the degrees of freedom and their indexes.

        Examples
        --------
        The numbering of the degrees of freedom for each node.

        Being the following their ordering for a node:

        x_0 - horizontal translation
        y_0 - vertical translation
        z_0 - axial translation
        alpha_0 - rotation around horizontal
        beta_0  - rotation around vertical
        theta_0 - torsion around axial

        >>> disk = disk_example_6dof()
        >>> disk.dof_mapping()
        {'x_0': 0, 'y_0': 1, 'z_0': 2, 'alpha_0': 3, 'beta_0': 4, 'theta_0': 5}
        """
        return dict(x_0=0, y_0=1, z_0=2, alpha_0=3, beta_0=4, theta_0=5)

    def M(self):
        """Mass matrix for an instance of a 6 DoF disk element.

        This method will return the mass matrix for an instance of a disk
        element with 6DoFs.

        Returns
        -------
        M : np.ndarray
            Mass matrix for the 6DoFs disk element.

        Examples
        --------
        >>> disk = DiskElement6DoF(0, 32.58972765, 0.17808928, 0.32956362)
        >>> disk.M().round(2)
        array([[32.59,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
               [ 0.  , 32.59,  0.  ,  0.  ,  0.  ,  0.  ],
               [ 0.  ,  0.  , 32.59,  0.  ,  0.  ,  0.  ],
               [ 0.  ,  0.  ,  0.  ,  0.18,  0.  ,  0.  ],
               [ 0.  ,  0.  ,  0.  ,  0.  ,  0.18,  0.  ],
               [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.33]])
        """
        m = self.m
        Id = self.Id
        Ip = self.Ip
        # fmt: off
        M = np.array([
            [m,  0,  0,  0,  0,  0],
            [0,  m,  0,  0,  0,  0],
            [0,  0,  m,  0,  0,  0],
            [0,  0,  0, Id,  0,  0],
            [0,  0,  0,  0, Id,  0],
            [0,  0,  0,  0,  0, Ip],
        ])
        # fmt: on
        return M

    def K(self):
        """Stiffness matrix for an instance of a 6 DoF disk element.

        This method will return the stiffness matrix for an instance of a disk
        element with 6DoFs.

        Returns
        -------
        K : np.ndarray
            A matrix of floats containing the values of the stiffness matrix.

        Examples
        --------
        >>> disk = disk_example_6dof()
        >>> disk.K().round(2)
        array([[0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.]])
        """
        K = np.zeros((6, 6))

        return K

    def Kdt(self):
        """Dynamic stiffness matrix for an instance of a 6 DoF disk
        element.

        Stiffness matrix for the 6 DoF disk element associated with the
        transient motion. It needs to be multiplied by the angular
        acceleration when considered in time dependent analyses.

        Returns
        -------
        Kdt : np.ndarray
            A matrix of floats containing the values of the dynamic
            stiffness matrix.

        Examples
        --------
        >>> disk = disk_example_6dof()
        >>> disk.Kdt().round(2)
        array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.33, 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]])
        """
        Ip = self.Ip
        # fmt: off
        Kdt = np.array([[0, 0, 0,  0, 0, 0],
                        [0, 0, 0,  0, 0, 0],
                        [0, 0, 0,  0, 0, 0],
                        [0, 0, 0,  0, 0, 0],
                        [0, 0, 0, Ip, 0, 0],
                        [0, 0, 0,  0, 0, 0]])
        # fmt: on
        return Kdt

    def C(self):
        """Damping matrix for an instance of a 6 DoF disk element.

        Returns
        -------
        C : np.ndarray
            A matrix of floats containing the values of the damping matrix.

        Examples
        --------
        >>> disk = disk_example_6dof()
        >>> disk.C()
        array([[0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.]])
        """
        C = np.zeros((6, 6))

        return C

    def G(self):
        """Gyroscopic matrix for an instance of a 6 DoF disk element.

        This method will return the gyroscopic matrix for an instance of a disk
        element.

        Returns
        -------
        G : np.ndarray
            Gyroscopic matrix for the disk element.

        Examples
        --------
        >>> disk = DiskElement6DoF(0, 32.58972765, 0.17808928, 0.32956362)
        >>> disk.G().round(2)
        array([[ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
               [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
               [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
               [ 0.  ,  0.  ,  0.  ,  0.  ,  0.33,  0.  ],
               [ 0.  ,  0.  ,  0.  , -0.33,  0.  ,  0.  ],
               [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ]])
        """
        Ip = self.Ip
        # fmt: off
        G = np.array([
            [0, 0, 0,   0,  0, 0],
            [0, 0, 0,   0,  0, 0],
            [0, 0, 0,   0,  0, 0],
            [0, 0, 0,   0, Ip, 0],
            [0, 0, 0, -Ip,  0, 0],
            [0, 0, 0,   0,  0, 0],
        ])
        # fmt: on
        return G


def disk_example():
    """Create an example of disk element.

    This function returns an instance of a simple disk. The purpose is to make available
    a simple model so that doctest can be written using it.

    Returns
    -------
    disk : ross.DiskElement
        An instance of a disk object.

    Examples
    --------
    >>> disk = disk_example()
    >>> disk.Ip
    0.32956362
    """
    disk = DiskElement(0, 32.589_727_65, 0.178_089_28, 0.329_563_62)
    return disk


def disk_example_6dof():
    """Create an example of disk element.

    This function returns an instance of a simple disk. The purpose is to make available
    a simple model so that doctest can be written using it.

    Returns
    -------
    disk : ross.DiskElement6DoF
        An instance of a disk object.

    Examples
    --------
    >>> disk = disk_example_6dof()
    >>> disk.Ip
    0.32956362
    """
    disk = DiskElement6DoF(0, 32.589_727_65, 0.178_089_28, 0.329_563_62)
    return disk

if __name__ == "__main__":
    dsk = disk_example()
    print("Disk: ", dsk)

    dsk6d = disk_example_6dof()
    print("Disk 6DOF: ", dsk6d)

    dskBenchmark = DiskElement6DoF(0, 147.2150318, 2.514923460, 4.784488534, material=steel) # rho = 7810 / d_i = 0.1 / d_o = 0.5 / width = 0.1
    print("Disk Benchmark 6DOF: ", dskBenchmark) 