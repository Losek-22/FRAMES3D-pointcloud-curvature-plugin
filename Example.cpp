#include <ogx/Plugins/EasyPlugin.h>
#include <ogx/Data/Clouds/CloudHelpers.h>
#include <ogx/Data/Clouds/KNNSearchKernel.h>
#include <ogx/Data/Clouds/SphericalSearchKernel.h>
#include <ogx/Data/Primitives/PrimitiveHelpers.h>
#include <iostream>
#include <vector>
#include "mchtr_sgd.h"

using namespace ogx;
using namespace ogx::Data;

struct local_curvature : public ogx::Plugin::EasyMethod {

	// parameters
	Data::ResourceID node_id;
	int neighbours_count{ 15 };
	
	// inheritance from EasyMethod
	local_curvature() : EasyMethod(L"Przemys³aw Wysocki", L"Calculates curvature of the surface.") {}

	// add input/output parameters
	virtual void DefineParameters(ParameterBank& bank) {
		bank.Add(L"node_id", node_id).AsNode();
		bank.Add(L"neighbours_count", neighbours_count);
	}

	virtual void Run(Context& context) {

		// check neighbours_count validity (user input)
		if (neighbours_count < 1) {
			ReportError(L"K of nearest neighbours lower than 1.");
			return;
		}

		// get access to the node, handle exception
		auto node = context.m_project->TransTreeFindNode(node_id);
		if (!node) {
			ReportError(L"Invalid node id. Failed to run plugin.");
			return;
		}

		// access the element
		auto element = node->GetElement();
		if (!element) {
			OGX_LINE.Msg(ogx::Level::Error, L"Invalid element in the given node.");
			return;
		}

		// get the cloud
		auto cloud = element->GetData<ogx::Data::Clouds::ICloud>();
		if (!cloud) {
			OGX_LINE.Msg(ogx::Level::Error, L"Invalid cloud in the given node.");
			return;
		}
		
		// get access to the points
		ogx::Data::Clouds::PointsRange pointsRange;
		cloud->GetAccess().GetAllPoints(pointsRange);

		// KNN setup
		auto searchKNNKernel = ogx::Data::Clouds::KNNSearchKernel(ogx::Math::Point3D(0, 0, 0), neighbours_count);

		// data collection variables
		std::vector<ogx::Data::Clouds::Point3D> neighbouring_points;
		std::vector<float> curvatures;
		curvatures.reserve(pointsRange.size());

		// progress var for progress bar
		int progress = 0;

		// iterate over all 3D points
		for (const auto& xyz : ogx::Data::Clouds::RangeLocalXYZConst(pointsRange)) {

			// find KNNs
			searchKNNKernel.GetPoint() = xyz.cast<double>();
			ogx::Data::Clouds::PointsRange neighboursRange;
			cloud->GetAccess().FindPoints(searchKNNKernel, neighboursRange);
			auto neighboursXYZ = ogx::Data::Clouds::RangeLocalXYZConst(neighboursRange);

			// clear the vector containing each points' neighbours
			neighbouring_points.clear();

			// iterate over KNNs of given point, add them to vector	
			for (const auto& neighbourXYZ : neighboursXYZ) {
				neighbouring_points.push_back(neighbourXYZ);
			}

			// fit a sphere to neighbouring points (contained in vec) and get the surface curvature
			curvatures.push_back(static_cast<float>(1.0/(mchtr_sgd::find_sphere_r(neighbouring_points, xyz))));

			// progress bar update
			++progress;
			if (!context.Feedback().Update(static_cast<float>(progress) / pointsRange.size())) {
				ReportError(L"Could not update progress bar.");
			}
		}

		// create a new layer
		const auto layer_name = L"Curvatures";
		auto layer = cloud->CreateLayer(layer_name, 0.0);

		// add the layer to point range and set it to curvatures
		pointsRange.SetLayerVals(curvatures, *layer);

		// success message
		OGX_LINE.Msg(ogx::Level::Info, L"Pomyœlnie policzono krzywizny.");
	}
};

struct cut_pancake : public ogx::Plugin::EasyMethod {

	// parameters
	Data::ResourceID node_id;
	int pancake_range{ -1 };
	double center_point_x{ 0 };
	double center_point_y{ 0 };
	double center_point_z{ 0 };

	// constructor
	cut_pancake() : EasyMethod(L"Przemys³aw Wysocki", L"Cuts points outside the area of a circle of given radius and center point.") {}

	// add input/output parameters
	virtual void DefineParameters(ParameterBank& bank) {
		bank.Add(L"node_id", node_id).AsNode();
		bank.Add(L"pancake_range", pancake_range);
		bank.Add(L"center_point_x", center_point_x);
		bank.Add(L"center_point_y", center_point_y);
		bank.Add(L"center_point_z", center_point_z);
	}

	virtual void Run(Context& context) {

		// check pancake_range validity (user input)
		if (pancake_range < 1) {
			ReportError(L"Pancake range cannot be lower than 1.");
			return;
		}

		// get access to the node, handle exception
		auto node = context.m_project->TransTreeFindNode(node_id);
		if (!node) {
			ReportError(L"Invalid node id. Failed to run plugin.");
			return;
		}

		// access the element
		auto element = node->GetElement();
		if (!element) {
			OGX_LINE.Msg(ogx::Level::Error, L"Invalid element in the given node.");
			return;
		}

		// get the cloud
		auto cloud = element->GetData<ogx::Data::Clouds::ICloud>();
		if (!cloud) {
			OGX_LINE.Msg(ogx::Level::Error, L"Invalid cloud in the given node.");
			return;
		}

		// get access to the points
		ogx::Data::Clouds::PointsRange pointsRange;
		cloud->GetAccess().GetAllPoints(pointsRange);

		// progress var for progress bar
		int progress = 0;

		// central point
		const auto central_point = ogx::Math::Point3D(center_point_x, center_point_y, center_point_z);

		// get point iterator
		auto state_range = Data::Clouds::RangeState(pointsRange);
		auto state = state_range.begin();

		// iterate over 3D points
		for (const auto& xyz : ogx::Data::Clouds::RangeLocalXYZConst(pointsRange)) {

			// check if points lie within a circle, if so, delete them
			++state;
			if (Math::CalcPointToPointDistance3D(xyz.cast<Real>(), central_point) > pancake_range) {
				state->set(Data::Clouds::PS_DELETED);
			}

			// progress bar update
			++progress;
			if (!context.Feedback().Update(static_cast<float>(progress) / pointsRange.size())) {
				ReportError(L"Could not update progress bar.");
			}
		}
		OGX_LINE.Msg(ogx::Level::Info, L"Pomyœlnie usuniêto punkty.");
	}
};

OGX_EXPORT_METHOD(local_curvature)
OGX_EXPORT_METHOD(cut_pancake)
