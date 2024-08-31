import sys
import vtk
import argparse
from PyQt5 import QtCore, QtGui, QtWidgets
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import numpy as np
import os

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, object_folder, parent=None):
        super(MainWindow, self).__init__(parent)

        self.object_folder = object_folder

        self.frame = QtWidgets.QFrame()
        self.vl = QtWidgets.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)

        self.vl.addWidget(self.vtkWidget)
        
        # Add a done button
        self.done_button = QtWidgets.QPushButton("Done")
        self.done_button.clicked.connect(self.on_done_button_pressed)
        self.vl.addWidget(self.done_button)

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        self.init_vtk()

        self.interactor.Initialize()
        self.interactor.Start()


    def init_vtk(self):
        obj_file_path = os.path.join(self.object_folder, 'textured_mesh.obj')
        reader = vtk.vtkOBJReader()
        reader.SetFileName(obj_file_path)
        reader.Update()

        mesh = reader.GetOutput()
        bounds = mesh.GetBounds()
        print(bounds)

        transform = vtk.vtkTransform()
        transform.Translate(0, 0, 0)

        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetTransform(transform)
        transform_filter.SetInputData(mesh)
        transform_filter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(transform_filter.GetOutputPort())

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)

        self.renderer.AddActor(self.actor)
        self.renderer.SetBackground(0.1, 0.2, 0.4)

        self.add_axes(bounds)

        self.picker = vtk.vtkCellPicker()
        self.interactor.SetPicker(self.picker)
        self.keypoints = []

        self.interactor.AddObserver("LeftButtonPressEvent", self.on_left_button_press)

    def add_axes(self, bounds):
        max_dimension = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        axes_scale = max_dimension * 0.1 

        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(axes_scale, axes_scale, axes_scale)
        self.renderer.AddActor(axes)

    def on_left_button_press(self, obj, event):
        if self.interactor.GetShiftKey():
            click_pos = self.interactor.GetEventPosition()
            self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
            position = self.picker.GetPickPosition()
            self.keypoints.append(position)
            self.add_keypoint(position)
            self.update_keypoints_display()
        return

    def add_keypoint(self, position):
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(position)
        sphere_source.SetRadius(0.01)

        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())

        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(1, 0, 0)  # Red color

        self.renderer.AddActor(sphere_actor)
        self.vtkWidget.GetRenderWindow().Render()

    def update_keypoints_display(self):
        print("Selected keypoints:")
        for idx, point in enumerate(self.keypoints, start=1):
            print(f"{idx}: {point}")

    def on_done_button_pressed(self):
        keypoints_file_path = os.path.join(self.object_folder, 'keypoints.npy')
        np.save(keypoints_file_path, np.array(self.keypoints))
        print(f"Keypoints saved to {keypoints_file_path}")
        exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Object Keypoint Selector")
    parser.add_argument("object_folder", type=str, help="Path to the folder containing the object files (.obj and .npy)")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(args.object_folder)
    window.show()
    sys.exit(app.exec_())
