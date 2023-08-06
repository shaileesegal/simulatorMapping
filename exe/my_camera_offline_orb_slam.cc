#include <memory>
#include <string>
#include <thread>
#include <iostream>
#include <unistd.h>
#include <unordered_set>
#include <nlohmann/json.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "System.h"
#include "Converter.h"
#include "include/Auxiliary.h"

/************* SIGNAL *************/
// Unique pointer to the SLAM system and the output directory path
std::unique_ptr<ORB_SLAM2::System> SLAM;
std::string simulatorOutputDir;


// Save the SLAM map 
void saveMap(int mapNumber) {
    // Open a file to save the map data
    std::ofstream pointData;

    int i = 0;

    // Open the file to save the map data
    pointData.open(simulatorOutputDir + "cloud" + std::to_string(mapNumber) + ".csv");

    // Loop through all the map points in the SLAM system
    for (auto &p : SLAM->GetMap()->GetAllMapPoints()) {
        // Check if the map point is valid and not marked as bad
        if (p != nullptr && !p->isBad()) {
            // Get the world position of the map point
            auto point = p->GetWorldPos();
            Eigen::Matrix<double, 3, 1> vector = ORB_SLAM2::Converter::toVector3d(point);
            cv::Mat worldPos = cv::Mat::zeros(3, 1, CV_64F);
            worldPos.at<double>(0) = vector.x();
            worldPos.at<double>(1) = vector.y();
            worldPos.at<double>(2) = vector.z();

            // Update the normal and depth of the map point
            p->UpdateNormalAndDepth();
            cv::Mat Pn = p->GetNormal();
            Pn.convertTo(Pn, CV_64F);

            // save the map point data (to the file)
            pointData << i << ",";
            //save 3D world position
            pointData << worldPos.at<double>(0) << "," << worldPos.at<double>(1) << "," << worldPos.at<double>(2);
            //save maximum and minimum distance invariances, and the positions
            pointData << "," << p->GetMinDistanceInvariance() << "," << p->GetMaxDistanceInvariance() << "," << Pn.at<double>(0) << "," << Pn.at<double>(1) << "," << Pn.at<double>(2);

            // Get the observations of the map point in keyframes
            std::map<ORB_SLAM2::KeyFrame*, size_t> observations = p->GetObservations();

            // Open files to save keypoints and descriptors
            std::ofstream keyPointsData;
            std::ofstream descriptorData;
            keyPointsData.open(simulatorOutputDir + "point" + std::to_string(i) + "_keypoints.csv");
            descriptorData.open(simulatorOutputDir + "point" + std::to_string(i) + "_descriptors.csv");

            // Loop through all the keyframes where the map point is observed
            for (auto obs : observations) {
                ORB_SLAM2::KeyFrame *currentFrame = obs.first;

                // Get the current keypoint for the map point in the keyframe
                cv::KeyPoint currentKeyPoint = currentFrame->mvKeys[obs.second];

                // save keypoint data 
                keyPointsData << currentFrame->mnId << "," << currentKeyPoint.pt.x << "," << currentKeyPoint.pt.y <<
                              "," << currentKeyPoint.size << "," << currentKeyPoint.angle << "," <<
                              currentKeyPoint.response << "," << currentKeyPoint.octave << "," <<
                              currentKeyPoint.class_id << std::endl;

                // Get the descriptor of the map point in the keyframe
                cv::Mat current_descriptor = currentFrame->mDescriptors.row(obs.second);

                // save descriptor data 
                for (int j = 0; j < current_descriptor.rows; j++) {
                    descriptorData << static_cast<int>(current_descriptor.at<uchar>(j, 0));
                    for (int k = 1; k < current_descriptor.cols; k++) {
                        descriptorData << "," << static_cast<int>(current_descriptor.at<uchar>(j, k));
                    }
                    descriptorData << std::endl;
                }
            }

            // Close the keypoints and descriptors files
            keyPointsData.close();
            descriptorData.close();

            // Move to the next line in the map data file
            pointData << std::endl;
            i++;
        }
    }

    // Close the map data file
    pointData.close();
    //map saving is complete
    std::cout << "saved map" << std::endl; 
}


// handler function to stop the program gracefully, when a signal is received (s is the type(which) of signal sent)
void stopProgramHandler(int s) {
    // Save the current map and shutdown the SLAM system
    saveMap(std::chrono::steady_clock::now().time_since_epoch().count());
    SLAM->Shutdown();

    // Close all OpenCV windows
    cvDestroyAllWindows();

    // Print a message and exit the program
    std::cout << "stoped program" << std::endl;
    exit(1);
}



int main() {
    // Set up signal handlers to gracefully stop the program on certain signals
    signal(SIGINT, stopProgramHandler);
    signal(SIGTERM, stopProgramHandler);
    signal(SIGABRT, stopProgramHandler);
    signal(SIGSEGV, stopProgramHandler);

    // Read program settings (from a JSON file)
    std::string settingPath = Auxiliary::GetGeneralSettingsPath();
    std::ifstream programData(settingPath);
    nlohmann::json data;
    programData >> data;
    programData.close();

    char time_buf[21];
    time_t now;
    std::time(&now);
    std::strftime(time_buf, 21, "%Y-%m-%d_%H:%S:%MZ", gmtime(&now));
    std::string currentTime(time_buf);

    // Extract necessary paths and settings from the JSON data
    std::string vocPath = data["VocabularyPath"];
    std::string droneYamlPathSlam = data["DroneYamlPathSlam"];
    std::string videoPath = data["offlineVideoTestPath"];
    bool loadMap = data["loadMap"];
    bool isSavingMap = data["saveMap"];
    std::string loadMapPath = data["loadMapPath"];
    std::string simulatorOutputDirPath = data["simulatorOutputDir"];
    simulatorOutputDir = simulatorOutputDirPath + currentTime + "/";
    std::filesystem::create_directory(simulatorOutputDir);

    // Create initialize of the SLAM system
    SLAM = std::make_unique<ORB_SLAM2::System>(vocPath, droneYamlPathSlam, ORB_SLAM2::System::MONOCULAR, true, true, loadMap,
                                               loadMapPath, true);

    int amountOfAttempts = 0; //Possible to delete, may destroy in the future
    //video processing loop (processes the video frames from 'videoPath')
    while (amountOfAttempts++ < 1) {
        // Open the video file for processing
        cv::VideoCapture capture(0);
        //Checking, if we can open the video successfully
        if (!capture.isOpened()) {
            std::cout << "Error opening video stream or file" << std::endl;
            return 0;
        } else {
            std::cout << "Success opening video stream or file" << std::endl;
        }

        // read and process frames from the video
        cv::Mat frame;
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        //skip the first 170 frames(We are not interested in the first frames)
        for (int i = 0; i < 170; ++i) {
            capture >> frame;
        }
        int amount_of_frames = 1;

        for (;;) {
            // Track the monocular frame using the SLAM system
            SLAM->TrackMonocular(frame, capture.get(CV_CAP_PROP_POS_MSEC));

            // read the next frame
            capture >> frame;

            // Break, if there are no more frames to process
            if (frame.empty()) {
                break;
            }
        }

        // Save the map after processing the frames
        saveMap(amountOfAttempts);

        // Calculate and print the time taken for SLAM processing and the total number of frames processed
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<< std::endl;
        std::cout << amount_of_frames << std::endl;

        // Release the video resources
        capture.release();
    }

    // Save the SLAM map, if required
    if (isSavingMap) {
        SLAM->SaveMap(simulatorOutputDir + "simulatorMap.bin");
    }

    // Shut down the SLAM system and close all OpenCV windows
    SLAM->Shutdown();
    cvDestroyAllWindows();

    return 0;
}