#include "mchtr_sgd.h"
#include <math.h>

/*
Stochastic gradient descent functionality cpp file
Author: Przemyslaw Wysocki
*/

inline double mchtr_sgd::x_grad(const mchtr_sgd::sphere& sphere, const ogx::Data::Clouds::Point3D& point) {
	/*
	Calculates value of d/dx(loss function)
	@param		sphere - current sphere used in fitting
				point - point with respect to which the loss is being calculated
	@return		value of d/dx(loss function)
	*/
	return (2 * (sphere.x - point.x()) * (sqrt(pow(sphere.x - point.x(), 2) + pow(sphere.y - point.y(), 2) + pow(sphere.z - point.z(), 2)) - sphere.r)) /
		(sqrt(pow(sphere.x - point.x(), 2) + pow(sphere.y - point.y(), 2) + pow(sphere.z - point.z(), 2)));
}

inline double mchtr_sgd::y_grad(const mchtr_sgd::sphere& sphere, const ogx::Data::Clouds::Point3D& point) {
	/*
	Calculates value of d/dy(loss function)
	@param		sphere - current sphere used in fitting
				point - point with respect to which the loss is being calculated
	@return		value of d/dy(loss function)
	*/
	return (2 * (sphere.y - point.y()) * (sqrt(pow(sphere.x - point.x(), 2) + pow(sphere.y - point.y(), 2) + pow(sphere.z - point.z(), 2)) - sphere.r)) /
		(sqrt(pow(sphere.x - point.x(), 2) + pow(sphere.y - point.y(), 2) + pow(sphere.z - point.z(), 2)));
}

inline double mchtr_sgd::z_grad(const mchtr_sgd::sphere& sphere, const ogx::Data::Clouds::Point3D& point) {
	/*
	Calculates value of d/dz(loss function)
	@param		sphere - current sphere used in fitting
				point - point with respect to which the loss is being calculated
	@return		value of d/dz(loss function)
	*/
	return (2 * (sphere.z - point.z()) * (sqrt(pow(sphere.x - point.x(), 2) + pow(sphere.y - point.y(), 2) + pow(sphere.z - point.z(), 2)) - sphere.r)) /
		(sqrt(pow(sphere.x - point.x(), 2) + pow(sphere.y - point.y(), 2) + pow(sphere.z - point.z(), 2)));
}

inline double mchtr_sgd::r_grad(const mchtr_sgd::sphere& sphere, const ogx::Data::Clouds::Point3D& point) {
	/*
	Calculates value of d/dr(loss function)
	@param		sphere - current sphere used in fitting
				point - point with respect to which the loss is being calculated
	@return		value of d/dr(loss function)
	*/
	return -2 * (sqrt(pow(sphere.x - point.x(), 2) + pow(sphere.y - point.y(), 2) + pow(sphere.z - point.z(), 2)) - sphere.r);
}

double mchtr_sgd::find_sphere_r(const std::vector<ogx::Data::Clouds::Point3D>& data, const ogx::Data::Clouds::Point3D& central_point) {
	/*
	Performs a stochastic gradient descent fitting a sphere to n 3D points.
	@param		data - points which the sphere will be fit to
				central_point - the original point whose "data" points are neighbours of
	@returns	sphere.r - final radius of the sphere
	*/
	constexpr int no_epochs = 30;
	mchtr_sgd::sphere sphere = mchtr_sgd::init_sphere(central_point, 0.01, 0.15);
	for (int i = 0; i < no_epochs; ++i) {
		for (const ogx::Data::Clouds::Point3D& point : data) {
			double x_gradient = mchtr_sgd::x_grad(sphere, point);
			double y_gradient = mchtr_sgd::y_grad(sphere, point);
			double z_gradient = mchtr_sgd::z_grad(sphere, point);
			double r_gradient = mchtr_sgd::r_grad(sphere, point);
			mchtr_sgd::update_parameters(x_gradient, y_gradient, z_gradient, r_gradient, sphere);
		}
	}
	return sphere.r;
}

mchtr_sgd::sphere mchtr_sgd::init_sphere(const ogx::Data::Clouds::Point3D& central_point, float coord_offset, float initial_radius) {
	/*
	Initialises a sphere which will be used for fitting to data.
	@param		central_point - point which KNNs were found (which are the data)
				coord_offset - offsets of central points coordinates, diffrence of which are parameters of returned sphere
				initial_radius - initial sphere radius
	@return		sphere which is near the central point
	*/

	return mchtr_sgd::sphere{ central_point.x() - coord_offset,
							central_point.y() - coord_offset,
							central_point.z() - coord_offset,
							initial_radius };
}

inline double mchtr_sgd::calculate_loss(const mchtr_sgd::sphere& sphere, const ogx::Data::Clouds::Point3D& point) {
	/*
	Calculates a loss function for fitting a sphere to a single point (designed to iterate over many in a single epoch, not vectorised).
	Loss function used is distance between given point and sphere surface squared: L = (sqrt(((x-a)^2)+((y-b)^2)+((z-c)^2))-r)^2
	@param		sphere - current sphere used in fitting
				point - point with respect to which the loss is being calculated
	@return		value of loss function
	*/
	return pow(sqrt(pow(sphere.x - point.x(), 2) + pow(sphere.y - point.y(), 2) + pow(sphere.z - point.z(), 2)) - sphere.r, 2);
}

void mchtr_sgd::update_parameters(const double& x_grad, const double& y_grad, const double& z_grad,
	const double& r_grad, mchtr_sgd::sphere& sphere) {
	/*
	Does a forward step of stochastic gradient descent
	@param		*_grad - partial diffrential of the loss function with respect to *
				sphere - sphere being fit
	*/
	constexpr double xyz_learning_rate = 0.05;
	constexpr double r_learning_rate = 0.25;
	sphere.x = sphere.x - xyz_learning_rate * x_grad;
	sphere.y = sphere.y - xyz_learning_rate * y_grad;
	sphere.z = sphere.z - xyz_learning_rate * z_grad;
	sphere.r = sphere.r - r_learning_rate * r_grad;
	return;
}

