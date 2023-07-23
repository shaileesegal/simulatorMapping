#include "include/OfflineMapPoint.h"

#include <utility>

// Copy Constructor, Creates a new OfflineMapPoint by copying the content of another OfflineMapPoint object
OfflineMapPoint::OfflineMapPoint(const OfflineMapPoint &offlineMapPoint) {
    
    this->point = offlineMapPoint.point;

    // copy the minimum and maximum distance invariance values, to the current OfflineMapPoint class
    this->minDistanceInvariance = offlineMapPoint.minDistanceInvariance;
    this->maxDistanceInvariance = offlineMapPoint.maxDistanceInvariance;

    this->normal = offlineMapPoint.normal;
    this->keyPoints = offlineMapPoint.keyPoints;

    // Clone the descriptors (Mat objects) to avoid shallow copying and share the data
    this->descriptors = std::vector<cv::Mat>();
    for (auto desc : offlineMapPoint.descriptors)
        this->descriptors.push_back(desc.clone());
}

// Parameterized Constructor, Creates a new OfflineMapPoint with specified parameters
OfflineMapPoint::OfflineMapPoint(cv::Point3d point, double minDistanceInvariance, double maxDistanceInvariance, 
                                 cv::Point3d normal, std::vector<std::pair<long unsigned int, cv::KeyPoint>> keyPoints, 
                                 std::vector<cv::Mat> descriptors) {

    this->point = point;

    // Set the minimum and maximum distance invariance values
    this->minDistanceInvariance = minDistanceInvariance;
    this->maxDistanceInvariance = maxDistanceInvariance;

    
    this->normal = normal;

    this->keyPoints = keyPoints;

    // Clone the descriptors (Mat objects) to avoid shallow copying and share the data
    this->descriptors = std::vector<cv::Mat>();
    for (auto desc : descriptors)
        this->descriptors.push_back(desc.clone());
}

// Operator Overload, Checks if the current OfflineMapPoint is equal to a given 3D point
bool OfflineMapPoint::operator==(const cv::Point3d& anotherPoint) {
    return this->point == anotherPoint;
}

// Compares two OfflineMapPoint objects for equality based on their 3D points
bool OfflineMapPoint::compare(OfflineMapPoint offlineMapPoint) {
    // Check if the 3D point coordinates are equal
    if (this->point == offlineMapPoint.point)
        return true;
    return false;
}
