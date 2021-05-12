# FRAMES3D-pointcloud-curvature-plugin
Plugin for FRAMES3D for finding local curvature of a point cloud using stochastic gradient descent

Written for university. Sadly I'm now allowed to share the software itself.

The plugin operates on a point cloud, and then:

1) iterates over each point (or skips some depending on time optimization)
2) grabs k nearest neighbours
3) initates a sphere
4) calculates the gradient of loss function for fitting the sphere to points
5) updates the parameters of sphere
6) returns radiuses
7) maps the radiuses onto the point cloud in a form of color map

Files:

1) Example.cpp - main file with the plugin
2) mchtr_sgd.cpp - stochastic gradient descent funcionality
