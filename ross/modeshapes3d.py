from .results import ModalResults
from .rotor_assembly import Rotor
from .multi_rotor import MultiRotor
from .shaft_element import ShaftElement
from .gear_element import GearElement, GearElement6DoF

from mayavi import mlab
import numpy as np
import matplotlib.pyplot as plt

class ShapeVisualiser():
    def __init__(self, rotor_system:Rotor, modal_results:ModalResults, mode_number):
        self.rotor_system = rotor_system
        self.modal_results = modal_results
        self.eigenvalue = self.modal_results.evalues[mode_number - 1]
        self.eigenvector = self.modal_results.evectors[:, mode_number - 1]

        num_nodes = len(self.eigenvector) // (self.rotor_system.number_dof * 2) # positions and velocities
        self.eigenvector = self.eigenvector.reshape((num_nodes, self.rotor_system.number_dof * 2))

        # Extract only the position dofs, translation dofs and rotation dofs
        self.eigenv_pos = self.eigenvector[:, 0:6]
        self.eigenv_trans = self.eigenvector[:, 0:3]
        self.eigenv_rot = self.eigenvector[:, 3:6]

        self.figure = mlab.figure(bgcolor=(1, 1, 1), size=(800, 600))
        self._build_geometry()

    def show(self):
        mlab.show()

    def _build_geometry(self):
        """Builds the static 3D geometry of the rotor system."""
        if isinstance(self.rotor_system, MultiRotor):
            for j, rotor in enumerate(self.rotor_system.rotors):
                rotor_nodes = [rn + self.rotor_system.start_nodes_multirotor[j] for rn in rotor.nodes]
                get_node = lambda obj: rotor_nodes.index(obj.n + self.rotor_system.start_nodes_multirotor[j]) + self.rotor_system.start_nodes_multirotor[j]
                for shaft in rotor.shaft_elements:
                    i = get_node(shaft)
                    self._draw_shaft_element(shaft, self.rotor_system.nodes_pos[i], self.rotor_system.center_line_pos[i])
                for disk in rotor.disk_elements:
                    i = get_node(disk)
                    self._draw_disk_element(disk, self.rotor_system.nodes_pos[i], self.rotor_system.center_line_pos[i])
                for bearing in rotor.bearing_elements:
                    i = get_node(bearing)
                    self._draw_bearing_element(self.rotor_system.nodes_pos[i], self.rotor_system.center_line_pos[i])


    def _draw_shaft_element(self, shaft:ShaftElement, start_position, axis_position):
        """Draws a shaft element as a cylinder."""
        length = shaft.L
        radius = shaft.odl/2
        start = np.array([start_position, 0, axis_position])
        end = np.array([start_position + length, 0, axis_position])
        mlab.plot3d(*zip(start, end), tube_radius=radius, tube_sides=30, color=(0.5, 0.5, 0.5))
        mlab.mesh(*self._plot_circle(start, radius, 'x'), color=(0.3, 0.3, 0.3))
        mlab.mesh(*self._plot_circle(end, radius, 'x'), color=(0.3, 0.3, 0.3))


    def _draw_disk_element(self, disk, position, axis_position):
        """Draws a disk element as a cylinder."""
        radius = disk.do/2
        r_i = disk.di/2
        thickness = disk.w
        color = (0.6, 0.1, 0.1) if isinstance(disk, (GearElement, GearElement6DoF)) else (0.8, 0.2, 0.2)
        start = np.array([position - thickness/2, 0, axis_position])
        end = np.array([position + thickness/2, 0, axis_position])
        mlab.plot3d(*zip(start, end), tube_radius=radius, tube_sides=50, color=color)
        mlab.mesh(*self._plot_circle(start, radius, 'x', r_i=r_i), color=color)
        mlab.mesh(*self._plot_circle(end, radius, 'x', r_i=r_i), color=color)

    def _draw_bearing_element(self, position, axis_position, size=0.1):
        """Draws a bearing element."""
        pos = np.array([position, 0, axis_position])
        start_h = np.array([position, -size/2, axis_position])
        end_h = np.array([position, size/2, axis_position])
        mlab.plot3d(*zip(start_h, end_h), tube_radius=0.005, color=(0.1, 0.6, 0.1))
        start_v = np.array([position, 0, axis_position-size/2])
        end_v = np.array([position, 0, axis_position + size/2])
        mlab.plot3d(*zip(start_v, end_v), tube_radius=0.005, color=(0.1, 0.6, 0.1))


    def _plot_circle(self, pos, radius, axis, r_i=0.0, num_points=30):
            theta = np.linspace(0, 2 * np.pi, num_points)
            r = np.linspace(r_i, radius, num_points)
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
    
    def mode_shape_quivers(self, scale=1.0):
        """Shows the mode shape on the rotor geometry."""
        num_nodes = self.eigenvector.shape[0]
        modeshape = self.eigenvector / np.max(self.eigenvector[:, :int(self.eigenvector.shape[1]/2)])  # Normalize for visualization
        
        shaft_diams = [se.odl for se in self.rotor_system.shaft_elements]
        ymax = max(shaft_diams) * 1.5

        plt.figure()
        ax1 = plt.subplot(1, 2, 1, projection='3d')
        ax1.scatter(np.real(eigenv[:17, 0]), np.real(eigenv[:17, 1]), self.rotor_system.nodes_pos[:17], linestyle='-', marker='o')
        ax1.scatter(np.abs(eigenv[:17, 0]), np.abs(eigenv[:17, 1]), self.rotor_system.nodes_pos[:17], linestyle='-', marker='*')
        ax2 = plt.subplot(1, 2, 2, projection='3d')
        ax2.scatter(np.real(eigenv[17:, 0]), np.real(eigenv[17:, 1]), self.rotor_system.nodes_pos[17:], linestyle='-', marker='o')
        ax2.scatter(np.abs(eigenv[17:, 0]), np.abs(eigenv[17:, 1]), self.rotor_system.nodes_pos[17:], linestyle='-', marker='*')
        plt.show()
        exit()

        for i in range(num_nodes):
            dof = eigenv[i, :]
            translation = np.real(np.array(dof[:3]) * scale) * ymax
            print("Translation node ", i, ": ", translation)
            rotation = np.array(dof[3:6]) * scale

            # Get original node position
            pos = np.array([
                self.rotor_system.nodes_pos[i],
                0,
                self.rotor_system.center_line_pos[i]
            ])
            new_pos = pos + translation

            # Visualize displacement as arrow
            mlab.quiver3d(
                pos[0], pos[1], pos[2],
                translation[0], translation[1], translation[2],
                scale_factor=1.0, color=(0, 0, 1),
                mode='arrow', line_width=3, scale_mode='vector'
            )

            