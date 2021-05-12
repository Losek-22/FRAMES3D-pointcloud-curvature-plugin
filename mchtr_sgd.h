#pragma once

#include <ogx/Data/Primitives/PrimitiveHelpers.h>
#include <vector>

/*
Stochastic gradient descent functionality header file
Author: Przemyslaw Wysocki
*/

namespace mchtr_sgd
{
	struct sphere {
		float x, y, z, r;
	};

	double find_sphere_r(const std::vector<ogx::Data::Clouds::Point3D>&, const ogx::Data::Clouds::Point3D&);
	mchtr_sgd::sphere init_sphere(const ogx::Data::Clouds::Point3D&, float, float);
	void update_parameters(const double&, const double&, const double&, const double&, mchtr_sgd::sphere&);
	inline double calculate_loss(const mchtr_sgd::sphere&, const ogx::Data::Clouds::Point3D&);
	inline double x_grad(const mchtr_sgd::sphere&, const ogx::Data::Clouds::Point3D&);
	inline double y_grad(const mchtr_sgd::sphere&, const ogx::Data::Clouds::Point3D&);
	inline double z_grad(const mchtr_sgd::sphere&, const ogx::Data::Clouds::Point3D&);
	inline double r_grad(const mchtr_sgd::sphere&, const ogx::Data::Clouds::Point3D&);
}