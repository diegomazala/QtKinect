
#ifndef _POINT_CLOUD_H_
#define _POINT_CLOUD_H_

#include <Eigen\Dense>
#include <vector>

typedef Eigen::Vector3f PointXYZ;
typedef Eigen::Vector4f PointXYZW;

typedef std::vector<PointXYZ> PointCloudXYZ;
typedef std::vector<PointXYZW> PointCloudXYZW;


#endif // _POINT_CLOUD_H_
