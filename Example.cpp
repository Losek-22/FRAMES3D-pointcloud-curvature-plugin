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

struct Example : public ogx::Plugin::EasyMethod 
{
	// parameters
	Data::ResourceID node_id;
	ogx::String file_path;
	int neighbours_count{ -1 };
	
	// constructor
	Example() : EasyMethod(L"Przemys³aw Wysocki", L"Finds local curvatures of the point cloud. Uses stochastic gradient descent to fit a sphere to n nearest neighbours and retrieves its radius.") {}

	// add input/output parameters
	virtual void DefineParameters(ParameterBank& bank) {
		bank.Add(L"node_id", node_id).AsNode();
		bank.Add(L"file", file_path).AsFile();
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
		//ogx::Data::Clouds::PointsRange neighboursRange; TO BY£O
		auto searchKNNKernel = ogx::Data::Clouds::KNNSearchKernel(ogx::Math::Point3D(0, 0, 0), neighbours_count);

		// data collection for stochastic gradient descent setup and curvatures
		std::vector<ogx::Data::Clouds::Point3D> neighbouring_points;
		std::vector<float> curvatures;
		curvatures.reserve(pointsRange.size());

		// progress var for progress bar
		int progress = 0;

		// iterate over 3D points
		for (const auto& xyz : ogx::Data::Clouds::RangeLocalXYZConst(pointsRange)) {

			// find KNNs
			searchKNNKernel.GetPoint() = xyz.cast<double>();
			ogx::Data::Clouds::PointsRange neighboursRange;	// dodane
			//neighboursRange.clear(); TO BY£O
			cloud->GetAccess().FindPoints(searchKNNKernel, neighboursRange);
			auto neighboursXYZ = ogx::Data::Clouds::RangeLocalXYZConst(neighboursRange);

			// iterate over KNNs of given point	
			for (const auto& neighbourXYZ : neighboursXYZ) {
				neighbouring_points.push_back(neighbourXYZ);
			}

			// do stochastic gradient descent and fit a sphere, get its radius
			curvatures.push_back(static_cast<float>(mchtr_sgd::find_sphere_r(neighbouring_points, xyz)));

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

		OGX_LINE.Msg(ogx::Level::Info, L"Pomyœlnie policzono krzywizny.");
	}
};

OGX_EXPORT_METHOD(Example)
