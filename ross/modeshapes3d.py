from .results import ModalResults
from .rotor_assembly import Rotor
from .multi_rotor import MultiRotor
from .shaft_element import ShaftElement
from .gear_element import GearElement, GearElement6DoF

from mayavi import mlab
import numpy as np

class ShapeVisualiser():
    def __init__(self, rotor_system:Rotor, modal_results:ModalResults, mode_number):
        self.rotor_system = rotor_system
        self.modal_results = modal_results
        self.eigenvalue = self.modal_results.evalues[mode_number - 1]
        self.eigenvector = self.modal_results.evectors[:, mode_number - 1]
        
        self.figure = mlab.figure(bgcolor=(1, 1, 1), size=(800, 600))
        self._build_geometry()

    def show(self):
        mlab.show()

    def _build_geometry(self):
        """Builds the static 3D geometry of the rotor system."""
        if isinstance(self.rotor_system, MultiRotor):
            for j, rotor in enumerate(self.rotor_system.rotors):
                rotor_nodes = [rn + self.rotor_system.start_nodes_multirotor[j] for rn in rotor.nodes]
                for shaft in rotor.shaft_elements:
                    i = rotor_nodes.index(shaft.n + self.rotor_system.start_nodes_multirotor[j]) + self.rotor_system.start_nodes_multirotor[j]
                    self._draw_shaft_element(shaft, self.rotor_system.nodes_pos[i], self.rotor_system.center_line_pos[i])
                for disk in rotor.disk_elements:
                    i = rotor_nodes.index(disk.n + self.rotor_system.start_nodes_multirotor[j]) + self.rotor_system.start_nodes_multirotor[j]
                    self._draw_disk_element(disk, self.rotor_system.nodes_pos[i], self.rotor_system.center_line_pos[i])
                for bearing in rotor.bearing_elements:
                    i = rotor_nodes.index(bearing.n + self.rotor_system.start_nodes_multirotor[j]) + self.rotor_system.start_nodes_multirotor[j]
                    self._draw_bearing_element(self.rotor_system.nodes_pos[i], self.rotor_system.center_line_pos[i])


    def _draw_shaft_element(self, shaft:ShaftElement, start_position, axis_position):
        """Draws a shaft element as a cylinder."""
        length = shaft.L
        radius = shaft.odl/2
        start = np.array([start_position, 0, axis_position])
        end = np.array([start_position + length, 0, axis_position])
        mlab.plot3d(*zip(start, end), tube_radius=radius, tube_sides=30, color=(0.5, 0.5, 0.5))
        mlab.mesh(*self._plot_cirlce(start, radius, 'x'), tube_radius=radius/10, color=(0.3, 0.3, 0.3))
        mlab.mesh(*self._plot_cirlce(end, radius, 'x'), tube_radius=radius/10, color=(0.3, 0.3, 0.3))


    def _draw_disk_element(self, disk, position, axis_position):
        """Draws a disk element as a cylinder."""

        radius = min(disk.base_radius * 1.1 + 0.05, 1) if isinstance(disk, (GearElement, GearElement6DoF)) else disk.scale_factor / 8
        thickness = 0.10 * radius  # Arbitrary thickness for visualization

        start = np.array([position - thickness/2, 0, axis_position])
        end = np.array([position + thickness/2, 0, axis_position])
        mlab.plot3d(*zip(start, end), tube_radius=radius, tube_sides=30, color=(0.8, 0.2, 0.2))
        mlab.mesh(*self._plot_cirlce(start, radius, 'x'), tube_radius=radius/10, color=(0.6, 0.1, 0.1))
        mlab.mesh(*self._plot_cirlce(end, radius, 'x'), tube_radius=radius/10, color=(0.6, 0.1, 0.1))

    def _draw_bearing_element(self, position, axis_position, size=0.1):
        """Draws a bearing element."""
        pos = np.array([position, 0, axis_position])
        start_h = np.array([position, -size/2, axis_position])
        end_h = np.array([position, size/2, axis_position])
        mlab.plot3d(*zip(start_h, end_h), tube_radius=0.005, color=(0.1, 0.6, 0.1))
        start_v = np.array([position, 0, axis_position-size/2])
        end_v = np.array([position, 0, axis_position + size/2])
        mlab.plot3d(*zip(start_v, end_v), tube_radius=0.005, color=(0.1, 0.6, 0.1))


    def _plot_cirlce(self, pos, radius, axis, num_points=30):
            theta = np.linspace(0, 2 * np.pi, num_points)
            r = np.linspace(0, radius, num_points)
            theta, r = np.meshgrid(theta, r)
            if axis == 'x':
                y = r * np.cos(theta) + pos[1]
                z = r * np.sin(theta) + pos[2]
                x = pos[0] * np.ones_like(y)
            elif axis == 'y':
                x = r * np.cos(theta) + pos[0]
                z = r * np.sin(theta) + pos[2]
                y = pos[1] * np.ones_like(x)
            else:  # axis == 'z'
                x = r * np.cos(theta) + pos[0]
                y = r * np.sin(theta) + pos[1]
                z = pos[2] * np.ones_like(x)
            return x, y, z