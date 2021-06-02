#include <ogx/Plugins/EasyPlugin.h>
#include <ogx/Data/Clouds/CloudHelpers.h>
#include <ogx/Data/Clouds/KNNSearchKernel.h>
#include <ogx/Data/Clouds/SphericalSearchKernel.h>
#include <ogx/Data/Primitives/PrimitiveHelpers.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace ogx;
using namespace ogx::Data;

struct PrzemyslawWysocki_Task_6_PointCloud_7 : public ogx::Plugin::EasyMethod {

	// parameters
	Data::ResourceID node_id;
	int neighbours_count{ 25 };
	
	// inheritance from EasyMethod
	PrzemyslawWysocki_Task_6_PointCloud_7() : EasyMethod(L"Przemys³aw Wysocki", L"Performs a localization of buildings.") {}

	// add input parameters
	virtual void DefineParameters(ParameterBank& bank) {
		bank.Add(L"node_id", node_id).AsNode();
		bank.Add(L"neighbours_count", neighbours_count);
	}

	inline float dot_product(const Math::Vector3D& a, const Math::Vector3D& b) {
		return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
	}

	inline float vector_magnitude(const Math::Vector3D& a) {
		return std::sqrt(a.x() * a.x() + a.y() * a.y() + a.z() * a.z());
	}

	inline float get_angle_between_vectors(const Math::Vector3D& a, const Math::Vector3D& b) {
		return std::acos(dot_product(a, b) / (vector_magnitude(a) * vector_magnitude(b)));
	}

	void cloud_smoothing(Data::Clouds::ICloud* cloud, Context& context) {
		ogx::Data::Clouds::PointsRange pointsRange;
		cloud->GetAccess().GetAllPoints(pointsRange);

		// KNN setup
		auto searchKNNKernel = ogx::Data::Clouds::KNNSearchKernel(ogx::Math::Point3D(0, 0, 0), neighbours_count);

		// data collection variables
		std::vector<ogx::Data::Clouds::Point3D> neighbouring_points;

		// progress var for progress bar
		int progress = 0;

		// iterate over all 3D points
		for (auto& xyz : ogx::Data::Clouds::RangeLocalXYZ(pointsRange)) {

			// find KNNs
			searchKNNKernel.GetPoint() = xyz.cast<double>();
			ogx::Data::Clouds::PointsRange neighboursRange;
			cloud->GetAccess().FindPoints(searchKNNKernel, neighboursRange);
			auto neighboursXYZ = ogx::Data::Clouds::RangeLocalXYZConst(neighboursRange);

			// clear the vector containing each points' neighbours
			neighbouring_points.clear();

			// iterate over KNNs of given point, add them to a vector
			for (const auto& neighbourXYZ : neighboursXYZ) {
				neighbouring_points.push_back(neighbourXYZ.cast<float>());
			}

			// fit a best fitting plane to KNNs
			Math::Plane3D best_plane = Math::CalcBestPlane3D(neighbouring_points.begin(), neighbouring_points.end());

			// project the point onto a best-fitted plane
			Math::Point3D projected_point = Math::ProjectPointOntoPlane(best_plane, xyz.cast<double>());

			// update the points' coordinates
			xyz = projected_point.cast<float>();

			// progress bar update
			++progress;
			if (!context.Feedback().Update(static_cast<float>(progress) / pointsRange.size())) {
				ReportError(L"Could not update progress bar.");
			}
		}
	}

	void find_roofs(Data::Clouds::ICloud* cloud, Context& context) {
		ogx::Data::Clouds::PointsRange pointsRange;
		cloud->GetAccess().GetAllPoints(pointsRange);

		// KNN setup
		auto searchKNNKernel = ogx::Data::Clouds::KNNSearchKernel(ogx::Math::Point3D(0, 0, 0), neighbours_count);

		// data collection variables
		std::vector<ogx::Data::Clouds::Point3D> neighbouring_points;
		std::vector<float> roofs;
		roofs.reserve(pointsRange.size());

		// vertical vector for calculating relative angles of normal vectors
		Math::Vector3D vertical_vector{ 0, 0, 1 };

		// for numeration of points belonging to roofs
		float current_roof_point = 0;

		// progress var for progress bar
		int progress = 0;

		// plane for reduction of point cloud tilt in Z axis
		std::vector<Math::Point3D> z_plane_points;
		z_plane_points.push_back(Math::Point3D(-22.6403f, 11.2198f, -90.7701f));
		z_plane_points.push_back(Math::Point3D(-35.1771f, -27.5203f, -92.5725f));
		z_plane_points.push_back(Math::Point3D(-1.0683f, -30.5571f, -91.7308f));
		z_plane_points.push_back(Math::Point3D(23.9246f, 0.1567f, -88.2513f));
		Math::Plane3D z_plane = Math::CalcBestPlane3D(z_plane_points.begin(), z_plane_points.end());

		// iterate over all 3D points
		for (const auto& xyz : ogx::Data::Clouds::RangeLocalXYZConst(pointsRange)) {

			// find KNNs
			searchKNNKernel.GetPoint() = xyz.cast<double>();
			ogx::Data::Clouds::PointsRange neighboursRange;
			cloud->GetAccess().FindPoints(searchKNNKernel, neighboursRange);
			auto neighboursXYZ = ogx::Data::Clouds::RangeLocalXYZConst(neighboursRange);

			// clear the vector containing previous points' neighbours
			neighbouring_points.clear();

			// iterate over KNNs of given point, add them to a vector
			for (const auto& neighbourXYZ : neighboursXYZ) {
				neighbouring_points.push_back(neighbourXYZ.cast<float>());
			}

			// fit a best fitting plane to KNNs
			Math::Plane3D best_plane = Math::CalcBestPlane3D(neighbouring_points.begin(), neighbouring_points.end());

			// is the point below or above the Z correction plane? the answer lies in sign of the distance
			float point_z_position = z_plane.signedDistance(xyz.cast<double>());

			// calculate angle between vertical vector and vector normal to the best plane
			float angle = get_angle_between_vectors(best_plane.normal(), vertical_vector);

			// if angle is 0 +- 15 degrees (normal vector is ~horizontal) AND the point is below the Z correction plane, mark the point as a roof incrementally
			if ((angle < 0.2618 || angle > 2.8798) && point_z_position > 0) {
				roofs.push_back(current_roof_point);
				++current_roof_point;
			}

			// else it's nothing important, mark as 0
			else {
				roofs.push_back(0);
			}

			// progress bar update
			++progress;
			if (!context.Feedback().Update(static_cast<float>(progress) / pointsRange.size())) {
				ReportError(L"Could not update progress bar.");
			}
		}

		// create a new layer
		const auto layer_name = L"buildings";
		auto layer = cloud->CreateLayer(layer_name, 0.0);

		// add the layer to point range and set it to curvatures
		pointsRange.SetLayerVals(roofs, *layer);
	}

	void segment_buildings(Data::Clouds::ICloud* cloud, Context& context) {
		ogx::Data::Clouds::PointsRange pointsRange;
		cloud->GetAccess().GetAllPoints(pointsRange);

		auto roof_layers = cloud->FindLayers(L"buildings");
		if (roof_layers.size() != 1) {
			ReportError(std::to_wstring(roof_layers.size()) + L" new layers found instead of 1.");
		}

		// for retrieving layer values
		std::vector<float> roofs;
		roofs.reserve(pointsRange.size());

		// retrieve roof layer values
		auto roof_layer = roof_layers[0];
		pointsRange.GetLayerVals(roofs, *roof_layer);

		// segmentation neighbours range
		const int neighbours_count_segmentation = 100;

		// KNN setup
		auto searchKNNKernel = ogx::Data::Clouds::KNNSearchKernel(ogx::Math::Point3D(0, 0, 0), neighbours_count_segmentation);

		// data collection variables
		std::vector<ogx::Data::Clouds::Point3D> neighbouring_points;
		std::vector<float> neighbouring_points_roof_values;

		// progress var for progress bar
		int progress = 0;

		// for iterating both 3D points AND roofs vector at once
		int current_roof_point = 0;

		// iterate over all 3D points
		for (const auto& xyz : ogx::Data::Clouds::RangeLocalXYZConst(pointsRange)) {

			// only for points marked as roofs
			if (roofs[current_roof_point] != 0) {

				// find KNNs
				searchKNNKernel.GetPoint() = xyz.cast<double>();
				ogx::Data::Clouds::PointsRange neighboursRange;
				cloud->GetAccess().FindPoints(searchKNNKernel, neighboursRange);
				auto neighboursXYZ = ogx::Data::Clouds::RangeLocalXYZConst(neighboursRange);

				// clear the vectors containing previous points' neighbours and their values
				neighbouring_points.clear();
				neighbouring_points_roof_values.clear();

				// retrieve roof values of neighbouring points
				neighboursRange.GetLayerVals(neighbouring_points_roof_values, *roof_layer);

				// delete values of 0 (not-roofs)
				neighbouring_points_roof_values.erase(std::remove(neighbouring_points_roof_values.begin(), neighbouring_points_roof_values.end(), 0), neighbouring_points_roof_values.end());

				// find min roof value in neighbouring points
				float new_value = *std::min_element(neighbouring_points_roof_values.begin(), neighbouring_points_roof_values.end());

				// update current point with new_value (lowest in the vincinity)
				roofs[current_roof_point] = new_value;
			}

			// increment the current vector element counter
			++current_roof_point;
			
			// progress bar update
			++progress;
			if (!context.Feedback().Update(static_cast<float>(progress) / pointsRange.size())) {
				ReportError(L"Could not update progress bar.");
			}
		}

		// set layer to new roof values
		pointsRange.SetLayerVals(roofs, *roof_layer);
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
		
		int steps = 3;
		OGX_LINE.Msg(ogx::Level::Info, L"Algorytm rozpocz¹³ pracê. 0/" + std::to_wstring(steps));
		OGX_LINE.Msg(ogx::Level::Info, L"Wyg³adzanie chmury punktów. 0/" + std::to_wstring(steps));

		// smoothing the point cloud
		cloud_smoothing(cloud, context);
		OGX_LINE.Msg(ogx::Level::Info, L"Chmura punktów zosta³a wyg³adzona. 1/" + std::to_wstring(steps));
		OGX_LINE.Msg(ogx::Level::Info, L"Rozpoczêcie szukania dachów budynków. 1/" + std::to_wstring(steps));

		// finding the roofs of buildings
		find_roofs(cloud, context);
		OGX_LINE.Msg(ogx::Level::Info, L"Znaleziono dachy budynków. 2/" + std::to_wstring(steps));
		OGX_LINE.Msg(ogx::Level::Info, L"Rozpoczêcie segmentacji budynków. 2/" + std::to_wstring(steps));

		// segment buildings into separate entities
		int segmentation_steps = 30;
		for (int i = 0; i < segmentation_steps; ++i) {
			OGX_LINE.Msg(ogx::Level::Info, L"----Krok " + std::to_wstring(i+1) + L"/" + std::to_wstring(segmentation_steps));
			segment_buildings(cloud, context);
		}
		OGX_LINE.Msg(ogx::Level::Info, L"Segmentacja dachów zakoñczona. 3/" + std::to_wstring(steps));

		// plugin has successfully finished working
		OGX_LINE.Msg(ogx::Level::Info, L"Plugin zakoñczy³ pracê.");
	}
};

OGX_EXPORT_METHOD(PrzemyslawWysocki_Task_6_PointCloud_7)
