"""Screw Element module.

This module defines the ScrewElement classes which can be used to represent
Screw rotors.
"""

import warnings
import numpy as np
from plotly import graph_objects as go
from ross.units import Q_, check_units
import shapefile
from shapely.geometry import shape, Polygon, Point, LineString, MultiLineString
import shapely as sh
from scipy.integrate import dblquad

from ross.shaft_element import ShaftElement6DoF


__all__ = ["ScrewElement"]


class ScrewElement(ShaftElement6DoF):
    """
    A screw rotor element.
    
    This class creates a screw element from screw parameters.

    Parameters
    ----------
    L : float, pint.Quantity
        Element length (m).
    crossSection : Polygon or string
        A file with the cross section shape or the cross section shape itself.
    units : string, optional
        The units used in the cross section file.
        Default is 'm'.
    material : ross.Material
        Shaft material.
    n : int, optional
        Element number (coincident with it's first node).
        If not given, it will be set when the rotor is assembled
        according to the element's position in the list supplied to
        the rotor constructor.
    axial_force : float, optional
        Axial force (N).
    torque : float, optional
        Torque (N*m).
    shear_effects : bool, optional
        Determine if shear effects are taken into account.
        Default is True.
    rotary_inertia : bool, optional
        Determine if rotary_inertia effects are taken into account.
        Default is True.
    gyroscopic : bool, optional
        Determine if gyroscopic effects are taken into account.
        Default is True.
    shear_method_calc : str, optional
        Determines which shear calculation method the user will adopt
        Default is 'cowper'
    alpha : float, optional
        Mass proportional damping factor.
        Default is zero.
    beta : float, optional
        Stiffness proportional damping factor.
        Default is zero.
    tag : str, optional
        Element tag.
        Default is None.
    color : str, optional
        A color to be used when the element is represented.
        Default is 'Purple'.

    Returns
    -------
    screw_element : ross.ScrewElement
        A 6 degrees of freedom screw rotor element object.

    Attributes
    ----------
    Poisson : float
        Poisson coefficient for the element.
    A : float
        Element section area at half length (m**2).
    A_l : float
        Element section area at left end (m**2).
    A_r : float
        Element section area at right end (m**2).
    beam_cg : float
        Element center of gravity local position (m).
    axial_cg_pos : float
        Element center of gravity global position (m).
        This should be used only after the rotor is built.
        Default is None.
    Ie : float
        Ie is the second moment of area of the cross section about
        the neutral plane (m**4).
    phi : float
        Constant that is used according to :cite:`friswell2010dynamics` to
        consider rotary inertia and shear effects. If these are not considered
        :math:`\phi=0`.
    kappa : float
        Shear coefficient for the element.

    References
    ----------
    .. bibliography::
        :filter: docname in docnames
    
    Examples
    --------
    >>> from ross.materials import steel
    >>> screw = ScrewElement(     
    ...            material=steel, L=0.5,
    ...            crossSection='polygon.shp',
    ...            units="mm",
    ...            rotary_inertia=False,
    ...            shear_effects=False
    ... )
    >>> screw.L
    0.5
    """
    
    @check_units
    def __init__(
        self,
        L,
        crossSection,
        units='m',
        material=None,
        n=None,
        axial_force=0,
        torque=0,
        shear_effects=True,
        rotary_inertia=True,
        gyroscopic=True,
        shear_method_calc="cowper",
        tag=None,
        alpha=0,
        beta=0
    ):
        self.units = units
        self._cs = None
        if type(crossSection) is Polygon:
            self.crossSection = crossSection
        elif type(crossSection) is str:
            self.crossSection = self.load_rotor_shape(crossSection, show=False)
        else:
            raise AssertionError("The cross-section should be a shapely.Polygon object or a path to a shapefile (*.shp).")

        r, theta = self.vertices_polar(self.crossSection)
        x, y = self.vertices_cartesian(self.crossSection)

        # print("inner radius: ", min(abs(r)))
        idl = 0
        odl = 2*max(abs(r))
        idr = idl
        odr = odl
        super().__init__(L, idl, odl, idr=idr, odr=odr, material=material, n=n, axial_force=axial_force, torque=torque, shear_effects=shear_effects, rotary_inertia=rotary_inertia, gyroscopic=gyroscopic, shear_method_calc=shear_method_calc, tag=tag, alpha=alpha, beta=beta)
        
        ###  Overwrite ShaftElement class data
        # A_l = cross section area from the left side of the element
        # A_r = cross section area from the right side of the element
        A_l = self._cs.area
        A_r = A_l
        self.A_l = A_l
        self.A_r = A_r

        # Second moment of area of the cross section from the left side
        # of the element
        Ix, _, _ = self.second_moments_of_inertia()
        self.Ie_l = Ix
        self.Ie_r = Ix

        self.volume = self.A_l*self.L
        self.m = self.material.rho * self.volume

        # geometrical coefficients - Not as relevant for this type of rotor as delta_ro and delta_ri is zero
        roj = odl / 2
        rij = idl / 2
        rok = odr / 2
        rik = idr / 2

        delta_ro = rok - roj # 0
        delta_ri = rik - rij # 0
        a1 = 2 * np.pi * (roj * delta_ro - rij * delta_ri) / self.A_l # 0
        a2 = np.pi * (roj**3 * delta_ro - rij**3 * delta_ri) / self.Ie_l # 0
        b1 = np.pi * (delta_ro**2 - delta_ri**2) / self.A_l # 0
        b2 = 3 * np.pi * (roj**2 * delta_ro**2 - rij**2 * delta_ri**2) / (2 * self.Ie_l) # 0
        gama = np.pi * (roj * delta_ro**3 - rij * delta_ri**3) / self.Ie_l # 0
        delta = np.pi * (delta_ro**4 - delta_ri**4) / (4 * self.Ie_l) # 0

        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2
        self.gama = gama
        self.delta = delta

        # the area is calculated from the cross section located in the middle
        # of the element
        self.A = A_l * (1 + a1 * 0.5 + b1 * 0.5**2) # A = A_l

        # Ie is the second moment of area of the cross section - located in
        # the middle of the element - about the neutral plane
        Ie = self.Ie_l * (1 + a2 * 0.5 + b2 * 0.5**2 + gama * 0.5**3 + delta * 0.5**4)
        self.Ie = Ie # Ie = Ie_l

        # geometric center
        c1 = roj**2 + 2 * roj * rok + 3 * rok**2 - rij**2 - 2 * rij * rik - 3 * rik**2
        c2 = (roj**2 + roj * rok + rok**2) - (rij**2 + rij * rik + rik**2)
        self.beam_cg = L * c1 / (4 * c2) # L/2
        self.axial_cg_pos = None

        # Slenderness ratio of beam elements (G*A*L**2) / (E*I) = G/E*l^2 with l the effective slenderness
        sld = (self.material.G_s * self.A * self.L**2) / (self.material.E * self.Ie)
        self.slenderness_ratio = sld

        # Moment of inertia
        self.Im = self.polar_moment_of_inertia()

        # picking a method to calculate the shear coefficient
        # List of avaible methods:
        # hutchinson - kappa as per Hutchinson (2001)
        # cowper - kappa as per Cowper (1996)
        if shear_effects:
            r_equiv = np.sqrt(self.A_l/np.pi)
            if shear_method_calc == "hutchinson":
                # Shear coefficient (phi)
                # kappa as per Hutchinson (2001)
                # fmt: off
                kappa = 6 * (1 + self.material.Poisson)**2 / (7 + 12 * self.material.Poisson + 4 * self.material.Poisson**2)
                # fmt: on
            elif shear_method_calc == "cowper":
                # kappa as per Cowper (1996)
                # fmt: off
                kappa = 6 * (
                    (1 + self.material.Poisson)
                    / (7 + 6 *self.material.Poisson)
                )
                # fmt: on
            else:
                raise Warning(
                    "This method of calculating shear coefficients is not implemented. See guide for futher informations."
                )

            # Quasi-static check
            phi = 0
            # fmt: off
            phi = 12 * self.material.E * self.Ie / (self.material.G_s * kappa * self.A * L ** 2)
            # fmt: on
            self.kappa = kappa

        self.phi = phi
        self.dof_global_index = None
        

    @classmethod
    def load_rotor_shape(cls, fileName, show=False):
        r = shapefile.Reader(fileName)
        shapes = r.shapes()
        polygon = shape(shapes[0])
        if show:
            fig = go.Figure(data=go.Scatter(x=polygon.exterior.xy[0].tolist(), y=polygon.exterior.xy[1].tolist(), mode='lines'))
            fig.update_xaxes(constrain='domain')  
            fig.update_yaxes(scaleanchor= 'x')
            fig.show()
        return polygon

    @classmethod
    def vertices_cartesian(cls, cs:Polygon):
        data = cs.exterior.xy
        x, y = data[0], data[1]
        return np.array(x), np.array(y)
    
    @classmethod
    def vertices_polar(cls, cs:Polygon):
        x, y = cls.vertices_cartesian(cs)
        r = np.sqrt(x**2 + y**2)
        theta = np.atan2(x, y)
        return r, theta
    
     
    @property
    def crossSection(self):
        return self._cs
    
    @crossSection.setter
    def crossSection(self, cs:Polygon):
        if not cs.is_closed:
            cs = Polygon(list(cs.exterior.coords) + [cs.exterior.coords[0]])

        if self.units == 'm':
            cs_scaled = cs
        elif self.units == 'cm':
            cs_scaled = sh.transform(cs, lambda x: x * [1e-2, 1e-2])
        elif self.units == 'mm':
            cs_scaled = sh.transform(cs, lambda x: x * [1e-3, 1e-3])
        else:
            raise ValueError("Units do not exist. Only 'm', 'cm' and 'mm' are implemented.")
        
        centr = cs_scaled.centroid
        if centr.x > 1e-6 or centr.y > 1e-6:
            cs_shift = sh.transform(cs_scaled, lambda e: e - [centr.x, centr.y])
        else:
            cs_shift = cs_scaled
        
        self._cs = cs_shift


    def second_moments_of_inertia(self):
        x, y = self.vertices_cartesian(self.crossSection)
        Ixx = 0
        Iyy = 0
        Ixy = 0

        for i in range(len(x) - 1):
            xi, yi = x[i], y[i]
            xi1, yi1 = x[i + 1], y[i + 1]

            common = (xi * yi1 - xi1 * yi)
            Ixx += (yi**2 + yi * yi1 + yi1**2) * common
            Iyy += (xi**2 + xi * xi1 + xi1**2) * common
            Ixy += (xi * yi1 + 2 * xi * yi + 2 * xi1 * yi1 + xi1 * yi) * common
        
        # Absolute value, because sign tells whether coordinates are (counter-)clockwisely defined.
        Ixx = float(abs(Ixx) / 12.0)
        Iyy = float(abs(Iyy) / 12.0)
        Ixy = float(abs(Ixy) / 24.0)
        return Ixx, Iyy, Ixy
    

    def __integral_over_distance_squared(self, r_min=0, theta_bounds=[0, 2*np.pi], niveau=0, debug=False, max_recursion=4):
        ### Integrate a Shapely Polygon over distance^2. Also, non-convex polygons can be integrated.
        if niveau > (max_recursion - 1):
            warnings.warn("[__integral_over_distance_squared] The maximal recursive depth {} is reached.".format(str(max_recursion)))
            return 0
        
        def dist(r, theta):
            # [__integral_over_distance_squared] Integrandum
            return r**3 # r^2*dxdy = r^3 dr dtheta
        
        # Determine bounds of polygon
        _, _, x_max, y_max = self.crossSection.bounds
        r_max = np.sqrt(x_max**2 + y_max**2)

        def search_area_plot(center=[0,0], radius_min=0.5, radius_max=1, start_angle=0, end_angle=90, n=50):
            # [__integral_over_distance_squared] Debug function: create a SVG figure that indicates a section of two concentric circles from 
            # start_angle to end_angle with the inner circle's radius being radius_min and the outer circle's 
            # radius is radius_max. The circles are centered around center. The parameter is used to determine 
            # the resolution of the circle arcs.
            t_min = np.linspace(start_angle, end_angle, n)
            x_min = center[0] + radius_min*np.cos(t_min)
            y_min = center[1] + radius_min*np.sin(t_min)
            t_max = t_min[::-1]
            x_max = center[0] + radius_max*np.cos(t_max)
            y_max = center[1] + radius_max*np.sin(t_max)
            path = f"M {x_min[0]},{y_min[0]}"
            for xc, yc in zip(x_min[1:], y_min[1:]):
                path += f" L{xc},{yc}"
            path += f" L{x_max[0]}, {y_max[0]}"
            for xc, yc in zip(x_max[1:], y_max[1:]):
                path += f" L{xc},{yc}"
            return path + " Z"
        
        if debug:
            # With the debug flag on, the sections that are integrated are plotted on the cross section polygon
            path_sect = search_area_plot(center=[0, 0], radius_min=r_min, radius_max=r_max, start_angle=theta_bounds[0], end_angle=theta_bounds[1])
            fig = go.Figure()
            fig.update_layout(xaxis_range=[-1.1*r_max, 1.1*r_max], yaxis_range=[- 1.1*r_max, 1.1*r_max], title=str(niveau),
                    shapes=[dict(type="path",
                                path=path_sect,
                                fillcolor="LightPink",
                                line_color="Crimson", opacity=0.5)])
            
            # Plot the cross section
            x, y = self.vertices_cartesian(self.crossSection)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines"
                )
            )
            fig.update_yaxes(scaleanchor = "x", # These yaxis settings ensure that the circle is non-deformed
                            scaleratio = 1)
            fig.show()
        

        def create_line(r_min, theta, extension=0.0001):
            # [__integral_over_distance_squared] Create a Shapely Line extending from r_min to r_max along the angle theta. The extension 
            # parameter allows to extend beyond r_min and r_max.
            r_min_ext = r_min * (1 - extension)
            r_max_ext = r_max * (1 + extension)
            return sh.LineString([[r_min_ext * np.cos(theta), r_min_ext * np.sin(theta)], [r_max_ext*np.cos(theta), r_max_ext * np.sin(theta)]])
        

        def filter_crossings(cross, eps=1e-4):
            # [__integral_over_distance_squared] Check whether there are no small numerical artefacts that causes the multiple-crossings 
            # algorithm. The parameter eps determines the threshold to remove artefacts. 
            if type(cross) is MultiLineString:
                lines = []
                for c in cross.geoms:
                    x_c, y_c = c.xy
                    dx = abs(x_c[1] - x_c[0])
                    dy = abs(y_c[1] - y_c[0])
                    if not (dx < eps and dy < eps):
                        line = [[x_c[0], y_c[0]], [x_c[1], y_c[1]]]
                        lines.append(line)

                if len(lines) == 1:
                    return LineString(lines[0])
                else:
                    return MultiLineString(lines)
            else:
                return cross


        def new_intersection(theta, eps=np.pi/180):
            # [__integral_over_distance_squared] If a new distant area is found that is radially disconnected from the 
            # previous part along a single angle, the radial and angle's bounds are 
            # returned that encompasses the area. The first observation of the radially 
            # disconnected area was at angle theta. The search area for theta will occur 
            # from theta - eps, with eps a small value.
            theta_min = theta - eps
            th = np.arange(theta_min, 2*np.pi + theta_min, 0.01*np.pi/180)
            
            # Initialise
            r_min_new = r_min * 0.9
            theta_bounds = [0, 0]
            crossed_bool = False

            for thi in th:
                # Make full circle and check where the radially disconnected area starts and ends 
                # by drawing a radial line and observing the number of crossings.
                l = create_line(r_min, thi)
                c = filter_crossings(self.crossSection.intersection(l), eps=r_max/1e4)

                if type(c) is MultiLineString and not crossed_bool:
                    # A new radially disconnected area is observed
                    x_c, y_c = c.geoms[1].xy
                    x = x_c[1]
                    y = y_c[1]
                    r_min_new = np.sqrt(x**2 + y**2)
                    theta_bounds[0] = thi
                    crossed_bool = True

                elif type(c) is LineString and crossed_bool:
                    # The radially disconnected area disappeared, return the minimal radius and the theta bounds for which it is present.
                    theta_bounds[1] = thi
                    return r_min_new, theta_bounds
                elif crossed_bool:
                    # While crossing the radially disconnected area, check if the minimal radius is still valid
                    x_c, y_c = c.geoms[1].xy
                    x = x_c[0]
                    y = y_c[0]
                    r_temp = np.sqrt(x**2 + y**2)
                    r_min_new = r_temp if r_temp < r_min_new else r_min_new
                
                
        def is_checked(theta, theta_bounds):
            # [__integral_over_distance_squared] Check whether the radially disconnected area that is detected at angle theta was already 
            # integraded by checking a list of previously integrated theta bounds.
            for theta_b in theta_bounds:
                if theta >= theta_b[0] and theta <= theta_b[1]:
                    return True
            return False

            

        number_of_lines_prev = 1
        integ_add = 0
        theta_list = []

        def y_limits(theta):
            # [__integral_over_distance_squared] This function returns the radial integration boundaries of the cross seciton for a certain angle theta. It implements 
            # a recursive interpretation if for one angle theta multiple crossing with the polygon are detected.
            nonlocal number_of_lines_prev, integ_add, theta_list

            # Find the intersections for a certain angle theta between r_min and r_max
            line = create_line(r_min, theta)
            cross = filter_crossings(self.crossSection.intersection(line), eps=r_max/1e4)
            number_of_lines = len(cross.geoms) if type(cross) is MultiLineString else 1 # Number of crossings
            if number_of_lines > number_of_lines_prev:
                # There is an increase of number of crossings
                if is_checked(theta, theta_list):
                    # As dblquad does not systematically go through the theta range, avoid integrating multiple times the same area
                    integ_prev = 0
                else:
                    # extra intersection so recursive integrate
                    intersect, theta_bounds = new_intersection(theta)
                    theta_list.append(theta_bounds)
                    integ_prev = self.__integral_over_distance_squared(intersect, theta_bounds, niveau = niveau + 1, debug=debug)
                    if debug:
                        print("r_min: ", intersect, " | theta: ", theta_bounds, " | integ_prev: ", integ_prev)
                integ_add += integ_prev
            
            number_of_lines_prev = number_of_lines

            # Integrate the area the closest to r_min
            if number_of_lines > 1:
                cross_base = cross.geoms[0]
            else:
                cross_base = cross

            # Calculate and return the boundaries for the closest area to r_min
            x_c, y_c = cross_base.xy
            x_c = x_c[1]
            y_c = y_c[1]
            r_c = np.sqrt(x_c**2 + y_c**2)
            return r_min, r_c


        def gfun(theta):
            # [__integral_over_distance_squared] Return the lowest boundary r_min for the integration algorithm.
            return y_limits(theta)[0]

        def hfun(theta):
            # [__integral_over_distance_squared] Return the highest boundary r_max for the integration algorithm.
            return y_limits(theta)[1]
        
        # Integration function
        integ, error = dblquad(dist, theta_bounds[0], theta_bounds[1], gfun, hfun)
        return integ + integ_add
            


    def polar_moment_of_inertia(self):
        # Calculate the polar moment of inertia Im.
        integ = self.__integral_over_distance_squared(debug=False)
        return integ * self.m / self.volume * self.L
    

    def show(self):
        # Plot the cross section.
        x, y = self.vertices_cartesian(self.crossSection)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines"
            )
        )
        fig.update_yaxes(scaleanchor = "x", # These yaxis settings ensure that the circle is non-deformed
                            scaleratio = 1)
        fig.show()


    def create_modified(self, **attributes):
        """Return a new screw element based on the current instance.

        Any attribute passed as an argument will be used to modify the corresponding
        attribute of the instance. Attributes not provided as arguments will retain
        their values from the current instance.

        Parameters
        ----------
        L : float, pint.Quantity, optional
            Element length (m). Default is equal to value of current instance.
        crossSection : Polygon or string
            A file with the cross section shape or the cross section shape itself.
        units : string, optional
            The units used in the cross section file.
            Default is 'm'.
        material : ross.Material, optional
            Shaft material. Default is equal to value of current instance.
        n : int, optional
            Element number (coincident with it's first node).
            Default is equal to value of current instance.
        axial_force : float, optional
            Axial force (N). Default is equal to value of current instance.
        torque : float, optional
            Torque (N*m). Default is equal to value of current instance.
        shear_effects : bool, optional
            Determine if shear effects are taken into account.
            Default is equal to value of current instance.
        rotary_inertia : bool, optional
            Determine if rotary_inertia effects are taken into account.
            Default is equal to value of current instance.
        gyroscopic : bool, optional
            Determine if gyroscopic effects are taken into account.
            Default is equal to value of current instance.
        shear_method_calc : str, optional
            Determines which shear calculation method the user will adopt
            Default is equal to value of current instance.
        alpha : float, optional
            Mass proportional damping factor.
            Default is equal to value of current instance.
        beta : float, optional
            Stiffness proportional damping factor.
            Default is equal to value of current instance.
        tag : str, optional
            Element tag.
            Default is None.

        Returns
        -------
        screw_element : ross.ScrewElement
            An instance of the modified screw element.
        """
        return self.__class__(
            L=attributes.get("L", self.L),
            crossSection=attributes.get("crossSection", self.crossSection),
            units=attributes.get("units", 'm'),
            material=attributes.get("material", self.material),
            n=attributes.get("n", self.n),
            axial_force=attributes.get("axial_force", self.axial_force),
            torque=attributes.get("torque", self.torque),
            shear_effects=attributes.get("shear_effects", self.shear_effects),
            rotary_inertia=attributes.get("rotary_inertia", self.rotary_inertia),
            gyroscopic=attributes.get("gyroscopic", self.gyroscopic),
            shear_method_calc=attributes.get(
                "shear_method_calc", self.shear_method_calc
            ),
            tag=attributes.get("tag", None),
            alpha=attributes.get("alpha", self.alpha),
            beta=attributes.get("beta", self.beta),
        )