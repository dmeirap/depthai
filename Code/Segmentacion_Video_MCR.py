#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
from time import monotonic
from datetime import timedelta
import blobconverter

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/home/adl/adl/depthai/videos/segmentacion_2/MCRFOV1_segmentation.mp4', fourcc, 20.0, (1248,936))

blob = dai.OpenVINO.Blob("/home/adl/adl/depthai/models/mcr_seg.blob")
videoPath = "/home/adl/adl/depthai/videos/videoMCRFOV1.mp4"
INPUT_SHAPE = (240,240)
TARGET_SHAPE = (1248,936)
num_of_classes = 3

def decode_deeplabv3p(output_tensor):
    
    output = output_tensor.reshape(*INPUT_SHAPE)

    # scale to [0 ... 2555] and apply colormap
    output = np.array(output) * (255/num_of_classes)
    output = output.astype(np.uint8)
    output_colors = cv2.applyColorMap(output, cv2.COLORMAP_JET)

    # reset the color of 0 class
    output_colors[output == 0] = [0,0,0]

    return output_colors

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.create(dai.node.NeuralNetwork)

xinFrame = pipeline.create(dai.node.XLinkIn)
xout_nn = pipeline.create(dai.node.XLinkOut)

detection_nn.setBlob(blob)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

# NN output linked to XLinkOut

xinFrame.setStreamName("inFrame")
xout_nn.setStreamName("nn")
xinFrame.out.link(detection_nn.input)
detection_nn.out.link(xout_nn.input)

# Pipeline is defined, now we can connect to the device
with dai.Device() as device:
    
    device.startPipeline(pipeline)

    qIn = device.getInputQueue(name="inFrame")
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    frames = {}

    def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()
    
    cap = cv2.VideoCapture(videoPath)

    while cap.isOpened():
        
        read_correctly, frame = cap.read()
        if not read_correctly:
            break
        
        img = dai.ImgFrame()
        img.setData(to_planar(frame, (INPUT_SHAPE[0], INPUT_SHAPE[1])))
        img.setTimestamp(monotonic())
        img.setWidth(INPUT_SHAPE[0])
        img.setHeight(INPUT_SHAPE[1])
        qIn.send(img)

        layer1 = q_nn.get().getFirstLayerInt32()
        lay1 = np.asarray(layer1, dtype=np.int32).reshape((240,240))
        output_colors = decode_deeplabv3p(lay1)
        output_colors = cv2.resize(output_colors, TARGET_SHAPE)

        frames['frame'] = frame
        #cv2.imshow("frame", frames['frame'])
        frame = cv2.addWeighted(frame, 1, output_colors,0.5,0)
        frames['colored_frame'] = frame

        cv2.imshow("Combined frame", frames['colored_frame'])
        out.write(frames['colored_frame'])

        if cv2.waitKey(1) == ord('q'):
            break