#include "include/simulator.h"

#define RESULT_POINT_X 0.1
#define RESULT_POINT_Y 0.2
#define RESULT_POINT_Z 0.3


//This function is for read descriptors from a file
std::vector<cv::Mat> readDesc(const std::string& filename, int cols)
{
    // Open the file with the parameter filename
    std::ifstream file(filename);
    if (!file.is_open())
    {
        // Handle file open error 
        return cv::Mat();
    }

    // Create a vector to store the descriptor matrices
    std::vector<cv::Mat> descs = std::vector<cv::Mat>();

    
    std::string line;
    int row = 0;
    while (std::getline(file, line)) {
        // Parse each line using an istringstream
        std::istringstream iss(line);

        // Create a cv::Mat object for the descriptor
        cv::Mat mat(1, cols, CV_8UC1);
        int col = 0;
        std::string value;

        // Split the line by commas and convert each value to uchar, storing it in the matrix
        while (std::getline(iss, value, ',')) {
            mat.at<uchar>(row, col) = static_cast<uchar>(std::stoi(value));
            col++;
        }

        // Add the descriptor matrix to the vector
        descs.push_back(mat);
        row++;
    }

    // Close the file
    file.close();

    // Return the vector containing the descriptor matrices
    return descs;
}

// This function is for read keypoints from a file
std::vector<std::pair<long unsigned int, cv::KeyPoint>> readKeyPoints(std::string filename) {
    // Open the file with the parameter filename
    std::ifstream file(filename);
    if (!file.is_open())
    {
        // Handle file open error 
        return std::vector<std::pair<long unsigned int, cv::KeyPoint>>();
    }

    // Create a vector to store the keypoint-frameID pairs
    std::vector<std::pair<long unsigned int, cv::KeyPoint>> keyPoints = std::vector<std::pair<long unsigned int, cv::KeyPoint>>();

    
    std::string line;
    while (std::getline(file, line)) {
        // Parse each line using an istringstream
        std::istringstream iss(line);

        // Create a vector to store the values of the current row
        std::string value;
        std::vector<std::string> row;

        // Split the line by commas and store the values in the row vector
        while (std::getline(iss, value, ',')) {
            row.push_back(value);
        }

        // Extract the values from the row and create a (cv::KeyPoint) object
        std::pair<long unsigned int, cv::KeyPoint> currentKeyPoint;
        long unsigned int frameId = stol(row[0]);
        cv::KeyPoint keyPoint(cv::Point2f(stof(row[1]), stof(row[2])), stof(row[3]), stof(row[4]), stof(row[5]), stoi(row[6]), stoi(row[7]));

        // Assign the frame ID and cv::KeyPoint object to the current pair
        currentKeyPoint.first = frameId;
        currentKeyPoint.second = keyPoint;

        // Add the current keypoint-frameID pair to the vector
        keyPoints.push_back(currentKeyPoint);
    }

    // Close the file
    file.close();

    // Return the vector containing the keypoint-frameID pairs
    return keyPoints;
}


//Function to create simulator settings from a JSON file
void Simulator::createSimulatorSettings() {
    
    // Create a character array to store the current working directory path
    char currentDirPath[256];
    
    getcwd(currentDirPath, 256);
    std::string settingPath = currentDirPath;
    settingPath += "/../demoSettings.json";
    std::ifstream programData(settingPath);
    
    // Parse the JSON data into the mData member of the Simulator class
    programData >> this->mData;

    // Close the file
    programData.close();
}

// Function to initialize points from data files
void Simulator::initPoints() {
    // Open the point data file and related descriptor and keypoints files
    std::ifstream pointData;
    std::ifstream descData;
    std::vector<std::string> row;
    std::string line, word, temp;
    int pointIndex;
    std::vector<std::pair<long unsigned int, cv::KeyPoint>> currKeyPoints;
    std::string currKeyPointsFilename;
    std::vector<cv::Mat> currDesc;
    std::string currDescFilename;

    // Initialize variables to store the point data
    cv::Vec<double, 8> point;
    OfflineMapPoint *offlineMapPoint;

    // Open the point data file for reading
    pointData.open(this->mCloudPointPath, std::ios::in);

    // Loop through the lines of the point data file
    while (!pointData.eof()) {
        // Clear the row vector to store new data
        row.clear();

        // Read a line from the file
        std::getline(pointData, line);

        // Use stringstream to split the line into words (comma-separated values)
        std::stringstream words(line);

        // Skip empty lines
        if (line == "") {
            continue;
        }

        // Split the line into individual words (comma-separated values)
        while (std::getline(words, word, ',')) {
            // Convert each word to double, or set to 0 if out of range
            try {
                std::stod(word);
            } catch (std::out_of_range) {
                word = "0";
            }
            row.push_back(word);
        }

        // Store the data in the cv::Vec for the point
        point = cv::Vec<double, 8>(std::stod(row[1]), std::stod(row[2]), std::stod(row[3]), std::stod(row[4]), std::stod(row[5]), std::stod(row[6]), std::stod(row[7]), std::stod(row[8]));

        // Get the point index and filenames for the descriptor and keypoints files
        pointIndex = std::stoi(row[0]);
        currDescFilename = this->mSimulatorPath + "point" + std::to_string(pointIndex) + "_descriptors.csv";
        currDesc = readDesc(currDescFilename, 32);
        currKeyPointsFilename = this->mSimulatorPath + "point" + std::to_string(pointIndex) + "_keypoints.csv";
        currKeyPoints = readKeyPoints(currKeyPointsFilename);

        // Create an OfflineMapPoint object and store it in the mPoints vector
        offlineMapPoint = new OfflineMapPoint(cv::Point3d(point[0], point[1], point[2]), point[3], point[4], cv::Point3d(point[5], point[6], point[7]), currKeyPoints, currDesc);
        this->mPoints.emplace_back(offlineMapPoint);
    }
    free OfflineMapPoint;
    // Close the point data file
    pointData.close();
}

//Constructor for the Simulator class
Simulator::Simulator() {
    // Create simulator settings
    this->createSimulatorSettings();

    // Initialize the mPoints vector
    this->mPoints = std::vector<OfflineMapPoint*>();

    
    this->mSimulatorPath = this->mData["simulatorPointsPath"];
    this->mCloudPointPath = this->mSimulatorPath + "cloud0.csv";

    this->initPoints();

    // Initialize other variables related to cloud scanning
    this->mCloudScanned = std::vector<OfflineMapPoint*>();

    // Set the real result point and initialize mResultPoint
    this->mRealResultPoint = cv::Point3d(RESULT_POINT_X, RESULT_POINT_Y, RESULT_POINT_Z);
    this->mResultPoint = cv::Point3d();

    // Set titles for the simulator viewer and results window
    this->mSimulatorViewerTitle = "Simulator Viewer";
    this->mResultsWindowTitle = "Results";

    // Set the configuration path and read settings from file
    this->mConfigPath = this->mData["DroneYamlPathSlam"];
    cv::FileStorage fSettings(this->mConfigPath, cv::FileStorage::READ);

    // Get viewpoint settings from the configuration file
    this->mViewpointX = fSettings["Viewer.ViewpointX"];
    this->mViewpointY = fSettings["Viewer.ViewpointY"];
    this->mViewpointZ = fSettings["Viewer.ViewpointZ"];
    this->mViewpointF = fSettings["Viewer.ViewpointF"];

    // Set the starting position of the camera
    double startPointX = this->mData["startingCameraPosX"];
    double startPointY = this->mData["startingCameraPosY"];
    double startPointZ = this->mData["startingCameraPosZ"];
    this->mStartPosition = cv::Point3d(startPointX, startPointY, startPointZ);

    // Set the starting yaw, pitch, and roll angles of the camera
    this->mStartYaw = this->mData["yawRad"];
    this->mStartPitch = this->mData["pitchRad"];
    this->mStartRoll = this->mData["rollRad"];

    // Set the point size and results point size for visualization
    this->mPointSize = fSettings["Viewer.PointSize"];
    this->mResultsPointSize = this->mPointSize * 5;

    // Set identity matrices for camera transformations
    this->mTwc.SetIdentity();
    this->mTcw.SetIdentity();

    // Set the scaling factors for camera movement and rotation
    this->mMovingScale = this->mData["movingScale"];
    this->mRotateScale = this->mData["rotateScale"];

    // Build the simulator window with the parameter 'mSimulatorViewerTitle' of the cuttent class
    this->build_window(this->mSimulatorViewerTitle);

    // Check if ORB SLAM is used and initialize the ORB SLAM system if required
    this->mUseOrbSlam = this->mData["useOrbSlam"];
    this->mVocPath = this->mData["VocabularyPath"];
    this->mTrackImages = this->mData["trackImagesClass"];
    this->mLoadMap = this->mData["loadMap"];
    this->mLoadMapPath = this->mData["loadMapPath"];
    this->mSystem = nullptr;
    if (this->mUseOrbSlam) {
        this->mSystem = new ORB_SLAM2::System(this->mVocPath, this->mConfigPath, ORB_SLAM2::System::MONOCULAR, true, this->mTrackImages,
                                          this->mLoadMap, this->mLoadMapPath, false);
    }

    // Initialize flags for camera movement and scanning
    this->mFollowCamera = true;
    this->mShowPoints = true;
    this->mReset = false;
    this->mMoveLeft = false;
    this->mMoveRight = false;
    this->mMoveDown = false;
    this->mMoveUp = false;
    this->mMoveBackward = false;
    this->mMoveForward = false;
    this->mRotateLeft = false;
    this->mRotateRight = false;
    this->mRotateDown = false;
    this->mRotateUp = false;
    this->mFinishScan = false;

    // Define Camera Render Object for view/scene browsing using Pangolin
    this->mS_cam = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1024, 768, this->mViewpointF, this->mViewpointF, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(this->mViewpointX, this->mViewpointY, this->mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    // Add named OpenGL viewport to the window and provide 3D Handler
    this->mD_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(this->mS_cam));

    // Call the reset function to set initial camera settings
    this->reset();
}

//Build the simulator window
void Simulator::build_window(std::string title) {
    pangolin::CreateWindowAndBind(title, 1024, 768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

//Get the points visible in the current camera view
std::vector<OfflineMapPoint*> Simulator::getPointsFromTcw() {
    // Check settings file
    cv::FileStorage fsSettings(this->mConfigPath, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "Failed to open settings file at: " << this->mConfigPath << std::endl;
        exit(-1);
    }

    // Get camera intrinsic parameters from the settings file
    double fx = fsSettings["Camera.fx"];
    double fy = fsSettings["Camera.fy"];
    double cx = fsSettings["Camera.cx"];
    double cy = fsSettings["Camera.cy"];
    int width = fsSettings["Camera.width"];
    int height = fsSettings["Camera.height"];

    // Define the minimum and maximum valid image coordinates
    double minX = 3.7;
    double maxX = width;
    double minY = 3.7;
    double maxY = height;

    // Extract the camera pose matrix mTcw into an OpenCV cv::Mat Tcw_cv
    cv::Mat Tcw_cv = cv::Mat::eye(4, 4, CV_64FC1);
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
            Tcw_cv.at<double>(i,j) = this->mTcw.m[j * 4 + i];
        }
    }

    // Extract rotation and translation components (from Tcw_cv)
    cv::Mat Rcw = Tcw_cv.rowRange(0, 3).colRange(0, 3);
    cv::Mat Rwc = Rcw.t();
    cv::Mat tcw = Tcw_cv.rowRange(0, 3).col(3);
    cv::Mat mOw = -Rcw.t() * tcw;

    // Save the Twc (inverse camera pose) into an OpenCV cv::Mat Twc_cv
    cv::Mat Twc_cv = cv::Mat::eye(4, 4, CV_64FC1);
    Rwc.copyTo(Twc_cv.rowRange(0,3).colRange(0,3));
    Twc_cv.at<double>(0, 3) = mOw.at<double>(0);
    Twc_cv.at<double>(1, 3) = mOw.at<double>(1);
    Twc_cv.at<double>(2, 3) = mOw.at<double>(2);

    // Update the mTwc transformation matrix with Twc_cv
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            this->mTwc.m[j * 4 + i] = Twc_cv.at<double>(i, j);
        }
    }

    std::vector<OfflineMapPoint*> seen_points;

    // Loop through all map points in the mPoints vector
    for(OfflineMapPoint* point : this->mPoints)
    {
        // Convert the map point's world position to an OpenCV cv::Mat
        cv::Mat worldPos = cv::Mat::zeros(3, 1, CV_64F);
        worldPos.at<double>(0) = point->point.x;
        worldPos.at<double>(1) = point->point.y;
        worldPos.at<double>(2) = point->point.z;

        // Transform the world position to camera coordinates (Pc)
        const cv::Mat Pc = Rcw * worldPos + tcw;
        const double &PcX = Pc.at<double>(0);
        const double &PcY= Pc.at<double>(1);
        const double &PcZ = Pc.at<double>(2);

        // Check if the map point is behind the camera
        if(PcZ < 0.0f)
            continue;

        // Project the point into image coordinates (u, v)
        const double invz = 1.0f / PcZ;
        const double u = fx * PcX * invz + cx;
        const double v = fy * PcY * invz + cy;

        // Check if the projected point is outside the image bounds
        if(u < minX || u > maxX || v < minY || v > maxY)
            continue;

        
        const double minDistance = point->minDistanceInvariance;
        const double maxDistance = point->maxDistanceInvariance;
        const cv::Mat PO = worldPos - mOw;
        const double dist = cv::norm(PO);

        if(dist < minDistance || dist > maxDistance)
            continue;

        // Check if the map point's viewing angle satisfies a minimum threshold
        cv::Mat Pn = cv::Mat(3, 1, CV_64F);
        Pn.at<double>(0) = point->normal.x;
        Pn.at<double>(1) = point->normal.y;
        Pn.at<double>(2) = point->normal.z;

        const double viewCos = PO.dot(Pn) / dist;

        if(viewCos < 0.5)
            continue;

        // If all conditions are satisfied, add the map point to the vector of visible points
        seen_points.push_back(point);
    }

    // Return the vector containing visible map points
    return seen_points;
}

// Function to reset the simulator
void Simulator::reset() {
    // Set initial state flags (For example, The first one is Flag to control if points should be shown)
    this->mShowPoints = true;
    this->mFollow = true;
    this->mFollowCamera = true;
    this->mReset = false;

    // Set current camera position and orientation to the starting position and orientation
    this->mCurrentPosition = this->mStartPosition;
    this->mCurrentYaw = this->mStartYaw;
    this->mCurrentPitch = this->mStartPitch;
    this->mCurrentRoll = this->mStartRoll;

    // Opengl has inversed Y axis
    // Assign yaw, pitch and roll rotations and translation
    Eigen::Matrix4d Tcw_eigen = Eigen::Matrix4d::Identity();
    Tcw_eigen.block<3, 3>(0, 0) = (Eigen::AngleAxisd(this->mCurrentRoll, Eigen::Vector3d::UnitZ()) * 
                            Eigen::AngleAxisd(this->mCurrentYaw, Eigen::Vector3d::UnitY()) *
                            Eigen::AngleAxisd(this->mCurrentPitch, Eigen::Vector3d::UnitX())).toRotationMatrix();
    Tcw_eigen(0, 3) = this->mCurrentPosition.x;
    Tcw_eigen(1, 3) = -this->mCurrentPosition.y;
    Tcw_eigen(2, 3) = this->mCurrentPosition.z;

    // Update the mTcw transformation matrix with the new camera pose Tcw_eigen
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            this->mTcw.m[j * 4 + i] = Tcw_eigen(i,j);
        }
    }

    this->mNewPointsSeen = this->getPointsFromTcw();

    // Clear the vector of previously seen points
    this->mPointsSeen = std::vector<OfflineMapPoint*>();
}

//Function to toggle following the camera in the viewer
void Simulator::ToggleFollowCamera() {
    this->mFollowCamera = !this->mFollowCamera;
}

// Function to toggle showing points in the viewer
void Simulator::ToggleShowPoints() {
    this->mShowPoints = !this->mShowPoints;
}
// Function to perform a reset in the simulator (sets the mReset flag to true)
void Simulator::DoReset() {
    this->mReset = true;
}

//Functions are responsible for setting the corresponding movement flags to true
//Each function represents a specific direction of movement for the camera (left, right, down, up, backward, forward)
void Simulator::MoveLeft() {
    this->mMoveLeft = true;
}

void Simulator::MoveRight() {
    this->mMoveRight = true;
}

void Simulator::MoveDown() {
    this->mMoveDown = true;
}

void Simulator::MoveUp() {
    this->mMoveUp = true;
}

void Simulator::MoveBackward() {
    this->mMoveBackward = true;
}

void Simulator::MoveForward() {
    this->mMoveForward = true;
}


// functions are used to set the rotation flags to true
//Each function represents a specific rotation for the camera (left, right, down, up)
void Simulator::RotateLeft() {
    this->mRotateLeft = true;
}

void Simulator::RotateRight() {
    this->mRotateRight = true;
}

void Simulator::RotateDown() {
    this->mRotateDown = true;
}

void Simulator::RotateUp() {
    this->mRotateUp = true;
}

//Function to finish the scan in the simulator (or to indicate that the scanning process is complete )
void Simulator::FinishScan() {
    this->mFinishScan = true;
}

// Functions are used to apply translations and rotations to the camera in the simulator

void Simulator::applyUpToModelCam(double value) {
    // Values are opposite
    this->mTcw.m[3 * 4 + 1] -= value;
}

void Simulator::applyRightToModelCam(double value) {
    // Values are opposite
    this->mTcw.m[3 * 4 + 0] -= value;
}

void Simulator::applyForwardToModelCam(double value) {
    // Values are opposite
    this->mTcw.m[3 * 4 + 2] -= value;
}

void Simulator::applyYawRotationToModelCam(double value) {
    Eigen::Matrix4d Tcw_eigen = pangolin::ToEigen<double>(this->mTcw);

    // Values are opposite
    double rand = -value * (M_PI / 180);
    double c = std::cos(rand);
    double s = std::sin(rand);

    Eigen::Matrix3d R;
    R << c, 0, s,
        0, 1, 0,
        -s, 0, c;

    Eigen::Matrix4d pangolinR = Eigen::Matrix4d::Identity();
    pangolinR.block<3, 3>(0, 0) = R;

    // Left-multiply the rotation
    Tcw_eigen = pangolinR * Tcw_eigen;

    // Convert back to pangolin matrix and set
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            this->mTcw.m[j * 4 + i] = Tcw_eigen(i, j);
        }
    }
}

void Simulator::applyPitchRotationToModelCam(double value) {
    Eigen::Matrix4d Tcw_eigen = pangolin::ToEigen<double>(this->mTcw);

    // Values are opposite
    double rand = -value * (M_PI / 180);
    double c = std::cos(rand);
    double s = std::sin(rand);

    Eigen::Matrix3d R;
    R << 1, 0, 0,
        0, c, -s,
        0, s, c;

    Eigen::Matrix4d pangolinR = Eigen::Matrix4d::Identity();
    pangolinR.block<3, 3>(0, 0) = R;

    // Left-multiply the rotation
    Tcw_eigen = pangolinR * Tcw_eigen;

    // Convert back to pangolin matrix and set
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            this->mTcw.m[j * 4 + i] = Tcw_eigen(i, j);
        }
    }
}

//The function is responsible for visualizing map points in the 3D environment using OpenGL
//It takes care of drawing the points in different colors based on different characteristics
void Simulator::drawMapPoints()
{
    // Remove the points that are present in the current frame from the list of all seen points
    std::vector<OfflineMapPoint*> pointsExceptThisFrame = this->mPointsSeen;
    std::vector<OfflineMapPoint*>::iterator it;
    for (it = pointsExceptThisFrame.begin(); it != pointsExceptThisFrame.end();)
    {
        if (std::find(this->mCurrentFramePoints.begin(), this->mCurrentFramePoints.end(), *it) != this->mCurrentFramePoints.end())
        {
            it = pointsExceptThisFrame.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // Remove the points that are seen in the new frame from the list of points in the previous frame
    std::vector<OfflineMapPoint*> oldPointsFromFrame = this->mCurrentFramePoints;
    for (it = oldPointsFromFrame.begin(); it != oldPointsFromFrame.end();)
    {
        if (std::find(this->mNewPointsSeen.begin(), this->mNewPointsSeen.end(), *it) != this->mNewPointsSeen.end())
        {
            it = oldPointsFromFrame.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // Draw the points that are not present in the current frame (black color)
    glPointSize((GLfloat)this->mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0,0.0,0.0);

    for(auto point : pointsExceptThisFrame)
    {
        glVertex3f((float)point->point.x, (float)point->point.y, (float)point->point.z);
    }
    glEnd();

    // Draw the points that were seen in the previous frame but not in the new frame (red color)
    glPointSize((GLfloat)this->mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);

    for(auto point : oldPointsFromFrame)
    {
        glVertex3f((float)point->point.x, (float)point->point.y, (float)point->point.z);

    }
    glEnd();

    // Draw the points that are seen in the new frame (green color)
    glPointSize((GLfloat)this->mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0,1.0,0.0);

    for(auto point : this->mNewPointsSeen)
    {
        glVertex3f((float)point->point.x, (float)point->point.y, (float)point->point.z);

    }
    glEnd();
}

//This function checks if two pangolin::OpenGlMatrix objects are equal
//It compares each element of the two matrices to see if they are the same
bool areMatricesEqual(const pangolin::OpenGlMatrix& matrix1, const pangolin::OpenGlMatrix& matrix2) {
    for (int i = 0; i < 16; i++) {
        if (matrix1.m[i] != matrix2.m[i])
            return false;
    }
    //If all elements are equal, it returns true
    return true;
}

//Function to save only new points seen by the camera in the simulator
//It means, this function helps in identifying and saving only the new points that are observed in the current frame but have not been seen in previous frames
void Simulator::saveOnlyNewPoints() {
    this->mNewPointsSeen = this->mCurrentFramePoints;
    std::vector<OfflineMapPoint*>::iterator it;
    
    // Iterate through the mNewPointsSeen vector
    for (it = this->mNewPointsSeen.begin(); it != this->mNewPointsSeen.end();)
    {
        // Check if the current point (*it) is present in the mPointsSeen vector
        if (std::find(this->mPointsSeen.begin(), this->mPointsSeen.end(), *it) != this->mPointsSeen.end())
        {
            it = this->mNewPointsSeen.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

//This function copies the values from matrix2 to matrix1
void assignPreviousTwc(pangolin::OpenGlMatrix& matrix1, const pangolin::OpenGlMatrix& matrix2) {
    for (int i = 0; i < 16; i++) {
        matrix1.m[i] = matrix2.m[i];
    }
}

//The function is responsible for tracking the camera motion using the ORB-SLAM2 library
void Simulator::trackOrbSlam() {
    // Create timestamp
    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto value = now_ms.time_since_epoch();
    double timestamp = value.count() / 1000.0;

    // TODO: Create std::vector<cv::KeyPoint> of all projections of the this->mCurrentFramePoints
    // Create an empty vector to store all the 2D key points (projections) of the map points visible in the current camera frame
    std::vector<cv::KeyPoint> keyPoints;
    for (auto point : this->mCurrentFramePoints) {
        for (auto keyPoint : point->keyPoints) {
            // Add each 2D key point (projection) to the keyPoints vector
            keyPoints.push_back(keyPoint.second);
        }
    }

    // TODO: Create cv::Mat of all the descriptors
    // Create an empty cv::Mat to store the descriptors of all the key points in the current frame
    cv::Mat descriptors;
    for (auto point : this->mCurrentFramePoints) {
        for (auto descriptor : point->descriptors) {
            // Append each descriptor to the descriptors matrix
            descriptors.push_back(descriptor);
        }
    }

    this->mSystem->TrackMonocular(descriptors, keyPoints, timestamp);
}

// Function to run the simulator
void Simulator::Run() {
    pangolin::OpenGlMatrix previousTwc;

    while (!this->mFinishScan) {        //loop until finishing scanning
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //manage the camera state 
        if (this->mFollowCamera && this->mFollow) {
            this->mS_cam.Follow(this->mTwc);
        } else if (this->mFollowCamera && !this->mFollow) {
            this->mS_cam.SetModelViewMatrix(
                    pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));
            this->mS_cam.Follow(this->mTwc);
            this->mFollow = true;
        } else if (!this->mFollowCamera && this->mFollow) {
            this->mFollow = false;
        }

        this->mCurrentFramePoints = this->getPointsFromTcw();
        // If running with orb-slam move this points to orb-slam
        if (this->mUseOrbSlam)
            this->trackOrbSlam();

        if (!areMatricesEqual(previousTwc, this->mTwc)) {
            this->saveOnlyNewPoints();
            this->mPointsSeen.insert(this->mPointsSeen.end(), this->mNewPointsSeen.begin(), this->mNewPointsSeen.end());
        }

        this->mD_cam.Activate(this->mS_cam);

        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        if (this->mShowPoints) {
            this->drawMapPoints();
        }

        pangolin::FinishFrame();

        assignPreviousTwc(previousTwc, this->mTwc);

        //check movement and rotation of camera
        if (this->mMoveLeft)
        {
            this->applyRightToModelCam(-this->mMovingScale);
            this->mMoveLeft = false;
        }

        if (this->mMoveRight)
        {
            this->applyRightToModelCam(this->mMovingScale);
            this->mMoveRight = false;
        }

        if (this->mMoveDown)
        {
            // Opengl has inversed Y axis so we pass -value
            this->applyUpToModelCam(this->mMovingScale);
            this->mMoveDown = false;
        }

        if (this->mMoveUp)
        {
            // Opengl has inversed Y axis so we pass -value
            this->applyUpToModelCam(-this->mMovingScale);
            this->mMoveUp = false;
        }

        if (this->mMoveBackward)
        {
            this->applyForwardToModelCam(-this->mMovingScale);
            this->mMoveBackward = false;
        }

        if (this->mMoveForward)
        {
            this->applyForwardToModelCam(this->mMovingScale);
            this->mMoveForward = false;
        }

        if (this->mRotateLeft)
        {
            this->applyYawRotationToModelCam(-this->mRotateScale);
            this->mRotateLeft = false;
        }

        if (this->mRotateRight)
        {
            this->applyYawRotationToModelCam(this->mRotateScale);
            this->mRotateRight = false;
        }

        if (this->mRotateDown)
        {
            this->applyPitchRotationToModelCam(-this->mRotateScale);
            this->mRotateDown = false;
        }

        if (this->mRotateUp)
        {
            this->applyPitchRotationToModelCam(this->mRotateScale);
            this->mRotateUp = false;
        }

        //restart
        if (mReset) {
            this->reset();
        }
    }

    pangolin::DestroyWindow(this->mSimulatorViewerTitle);   //destroy prev window
    this->build_window(this->mResultsWindowTitle);  //build window to show the results

    this->BuildCloudScanned();   // Erased mNewPointsSeen to only new points but not combined yet so insert both
}

std::vector<OfflineMapPoint*> Simulator::GetCloudPoint() {
    return this->mCloudScanned;
}

//The function is responsible for building the mCloudScanned vector, which contains the scanned points seen so far during the simulation.
void Simulator::BuildCloudScanned() {
    // Erased mNewPointsSeen to only new points but not combined yet so insert both
    this->mCloudScanned.insert(this->mCloudScanned.end(), this->mNewPointsSeen.begin(), this->mNewPointsSeen.end());
    this->mCloudScanned.insert(this->mCloudScanned.end(), this->mPointsSeen.begin(), this->mPointsSeen.end());
}

void Simulator::SetResultPoint(const cv::Point3d resultPoint) {
    this->mResultPoint = resultPoint;
}

//The function is responsible for rendering the scanned points in the OpenGL visualization window.
void Simulator::drawResultPoints() {
    // Remove result point and real result point from cloud scanned if exist
    for(int i = 0; i < this->mCloudScanned.size(); i++) {
        if (*this->mCloudScanned[i] == this->mResultPoint) {
            this->mCloudScanned.erase(this->mCloudScanned.begin() + i);
            break;
        }
    }
    for(int i = 0; i < this->mCloudScanned.size(); i++) {
        if (*this->mCloudScanned[i] == this->mRealResultPoint) {
            this->mCloudScanned.erase(this->mCloudScanned.begin() + i);
            break;
        }
    }
    // Render the scanned points in mCloudScanned (color: black)
    glPointSize((GLfloat)this->mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0,0.0,0.0);

    for(auto point : this->mCloudScanned)
    {
        glVertex3f((float)point->point.x, (float)point->point.y, (float)point->point.z);
    }

    glEnd();
    // Render the estimated result point (color: red)
    glPointSize((GLfloat)this->mResultsPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);

    glVertex3f((float)this->mResultPoint.x, (float)this->mResultPoint.y, (float)this->mResultPoint.z);

    glEnd();

    
    // Render the ground truth result point (color: green)
    glPointSize((GLfloat)this->mResultsPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0,1.0,0.0);

    glVertex3f((float)this->mRealResultPoint.x, (float)this->mRealResultPoint.y, (float)this->mRealResultPoint.z);

    glEnd();
}

void Simulator::updateTwcByResultPoint() {
    // TODO: Change Twc to center the result point when I do check results
}

//CheckResults function is responsible for displaying the simulation results and points in the OpenGL viewer
//It also draws the result points, showing the current state of the simulation
void Simulator::CheckResults() {
    this->mCloseResults = false;
    
    // This loop, continuously update the OpenGL viewer with the current simulation results until the results window is closed.
    while (!this->mCloseResults) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        this->updateTwcByResultPoint();

        this->mS_cam.Follow(this->mTwc);
        this->mD_cam.Activate(this->mS_cam);

        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        
        this->drawResultPoints();

        pangolin::FinishFrame();
    }
    // Close the results window when the loop exits.
    pangolin::DestroyWindow(this->mResultsWindowTitle);
}

// Destructor for the Simulator class
Simulator::~Simulator() {
    for (auto ptr : this->mPoints) {
        free(ptr);
    }
    if (this->mSystem != nullptr)
        free(this->mSystem);
}

