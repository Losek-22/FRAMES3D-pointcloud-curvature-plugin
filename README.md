# FRAMES3D-pointcloud-plugins
Plugisn for FRAMES3D

Original pointcloud I've been working with:

![image](https://user-images.githubusercontent.com/55858107/120467950-f8b92900-c3a0-11eb-8ff7-cdfe24c4b531.png)


Algorithm 1: finding local curvature of a point cloud using stochastic gradient descent

Written for university. Sadly I'm now allowed to share the software itself.

The plugin operates on a point cloud, and then:

1) iterates over each point (or skips some depending on time optimization)
2) grabs k nearest neighbours
3) initates a sphere
4) calculates the gradient of loss function for fitting the sphere to points
5) updates the parameters of sphere
6) returns radiuses
7) maps the radiuses onto the point cloud in a form of color map

Example output:

![image](https://user-images.githubusercontent.com/55858107/120467804-d2938900-c3a0-11eb-8605-c03913634f93.png)

Files:

1) Example.cpp - main file with the plugin
2) mchtr_sgd.cpp - stochastic gradient descent funcionality

Algorithm 2: automatic segmentation of buildings contained in the pointcloud

Steps:

1) Smoothing the pointcloud by projecting points onto planes best fitted to KNNs (getting rid of thermal noise)
2) Checking if each points' normal vector is approximately vertical
3) If it is, and if it also is above ground level point is being marked as one belonging to a building
4) Iterative segmentation of buildings by assigning the point the lowest value of its neighbours

Example output:

![image](https://user-images.githubusercontent.com/55858107/120468077-21d9b980-c3a1-11eb-932b-383e6d7c5ccc.png)

Files:
Example.cpp - main file with the plugin
