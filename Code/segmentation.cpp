#include <iostream>
#include <chrono>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <depthai/depthai.hpp>

using namespace std;
using namespace std::chrono;


int num_of_classes = 4;

cv::Mat decodeDeeplabv3p(const cv::Mat& outputTensor) {
    
    cv::Mat output = cv::Mat(outputTensor).reshape(1, 240);
    output = output * (255 / num_of_classes);
    output.convertTo(output, CV_8U);
    cv::Mat outputColors(output.size(), CV_8UC3);
    cv::applyColorMap(output, outputColors, cv::COLORMAP_JET);
    outputColors.setTo(cv::Scalar(0, 0, 0), output == 0);

    return outputColors;
}


int main(int argc, char** argv) {
    
    dai::Pipeline pipeline;

    auto cam = pipeline.create<dai::node::ColorCamera>();
    cam->setPreviewSize(240, 240);
    cam->setInterleaved(false);
    cam->setFps(35);
    
    cam->setPreviewKeepAspectRatio(false);

    cam->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    cam->setBoardSocket(dai::CameraBoardSocket::RGB);
    cam->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);
    

    auto detectionNN = pipeline.create<dai::node::NeuralNetwork>();
    detectionNN->setNumInferenceThreads(2);
    detectionNN->input.setBlocking(false);

    auto xoutRgb = pipeline.create<dai::node::XLinkOut>();
    auto xoutNN = pipeline.create<dai::node::XLinkOut>();
    
    xoutRgb->setStreamName("cam");
    xoutNN->setStreamName("nn");
    cam->isp.link(xoutRgb->input);

    //dai::OpenVINO::Blob blob(nnPath);
    detectionNN->setBlobPath("/home/adl/adl/depthai/models/adl_pcami_seg.blob");
    
    detectionNN->passthrough.link(xoutRgb->input);
    cam->preview.link(detectionNN->input);
    detectionNN->out.link(xoutNN->input);
    
    dai::Device device(pipeline);

    auto colorQueue = device.getOutputQueue("cam", 4, false);
    auto nnQueue = device.getOutputQueue("nn", 4, false);
    
    cv::Mat frame;
    cv::Mat coloredFrame;

    auto startTime = steady_clock::now();
    int counter = 0;
    float fps = 0;
    auto color2 = cv::Scalar(255, 0, 255);
    
    
    while (true) {

        std::shared_ptr<dai::ImgFrame> colorFrame;
        std::shared_ptr<dai::NNData> nnPacket;

        colorFrame = colorQueue->get<dai::ImgFrame>();
        nnPacket = nnQueue->get<dai::NNData>();

        counter++;
        auto currentTime = steady_clock::now();
        auto elapsed = duration_cast<duration<float>>(currentTime - startTime);
        if(elapsed > seconds(1)) {
            fps = counter / elapsed.count();
            counter = 0;
            startTime = currentTime;
        }

        if(colorFrame) {
            
            frame = colorFrame->getCvFrame();
            cv::resize(frame, frame, cv::Size(640,400));
            std::stringstream fpsStr;
            fpsStr << "NN fps: " << std::fixed << std::setprecision(2) << fps;
            cv::putText(frame, fpsStr.str(), cv::Point(5, 380), cv::FONT_HERSHEY_TRIPLEX, 0.6, color2);
        }

        if(nnPacket) {
            
            auto layer1 = nnPacket->getFirstLayerInt32();
            cv::Mat lay1 = cv::Mat(240, 240, CV_32S, layer1.data()).clone();
            cv::Mat outputColors = decodeDeeplabv3p(lay1);
            cv::resize(outputColors, outputColors, cv::Size(640,400));
            coloredFrame = frame + outputColors * 0.5;
            
        }

        if(!frame.empty()) {
            
            cv::imshow("Segmentation", coloredFrame);
        }

        if (cv::waitKey(1) == 'q') {
            break;
        }
    } 
    return 0;
}
