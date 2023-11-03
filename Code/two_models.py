#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time
from datetime import timedelta
import blobconverter

# Custom JET colormap with 0 mapped to `black` - better disparity visualization
jet_custom = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_BONE)
jet_custom[0] = [0, 0, 0]

blob_poste = dai.OpenVINO.Blob("/home/adl/adl/depthai/models/poste_seg.blob")
blob_larguero = dai.OpenVINO.Blob("/home/adl/adl/depthai/models/larguero_seg.blob")

INPUT_SHAPE = (240,240)
TARGET_SHAPE = (640,480)
num_of_classes = 2

def decode_deeplabv3p(output_tensor):
    output = output_tensor.reshape(*INPUT_SHAPE)

    # scale to [0 ... 2555] and apply colormap
    output = np.array(output) * (255/num_of_classes)
    output = output.astype(np.uint8)
    output_colors = cv2.applyColorMap(output, cv2.COLORMAP_JET)

    # reset the color of 0 class
    output_colors[output == 0] = [0,0,0]

    return output_colors

def get_multiplier(output_tensor):
    #class_binary = [[0],[1],[2]]
    class_binary = [[0],[1]]
    class_binary = np.asarray(class_binary, dtype=np.uint8)
    output = output_tensor.reshape(*INPUT_SHAPE)
    output_colors = np.take(class_binary, output, axis=0)
    return output_colors

class FPSHandler:
    def __init__(self):
        self.timestamp = time.time()
        self.start = time.time()
        self.frame_cnt = 0
    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1
    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)

class HostSync:
    def __init__(self):
        self.arrays = {}
    def add_msg(self, name, msg):
        if not name in self.arrays:
            self.arrays[name] = []
        # Add msg to array
        self.arrays[name].append({'msg': msg})
        # Try finding synced msgs
        ts = msg.getTimestamp()
        synced = {}
        for name, arr in self.arrays.items():
            for i, obj in enumerate(arr):
                time_diff = abs(obj['msg'].getTimestamp() - ts)
                # 20ms since we add rgb/depth frames at 30FPS => 33ms. If
                # time difference is below 20ms, it's considered as synced
                if time_diff < timedelta(milliseconds=33):
                    synced[name] = obj['msg']
                    # print(f"{name}: {i}/{len(arr)}")
                    break
        # If there are 3 (all) synced msgs, remove all old msgs
        # and return synced msgs
        if len(synced) == 3: # color, depth, nn
            def remove(t1, t2):
                return timedelta(milliseconds=500) < abs(t1 - t2)
            # Remove old msgs
            for name, arr in self.arrays.items():
                for i, obj in enumerate(arr):
                    if remove(obj['msg'].getTimestamp(), ts):
                        arr.remove(obj)
                    else: break
            return synced
        return False

def crop_to_square(frame):
    height = frame.shape[0]
    width  = frame.shape[1]
    delta = int((width-height) / 2)
    # print(height, width, delta)
    return frame[0:height, delta:width-delta]

# Start defining a pipeline
pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.ColorCamera)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setPreviewKeepAspectRatio(False)
#cam.setIspScale(2,3) # To match 400P mono cameras
cam.setBoardSocket(dai.CameraBoardSocket.RGB)


# For deeplabv3
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setPreviewSize(*INPUT_SHAPE)
cam.setInterleaved(False)

# NN output linked to XLinkOut
isp_xout = pipeline.create(dai.node.XLinkOut)
isp_xout.setStreamName("cam")
cam.isp.link(isp_xout.input)

# Define a neural network that will make predictions based on the source frames
detection_nn_poste = pipeline.create(dai.node.NeuralNetwork)
detection_nn_poste.setBlob(blob_poste)
detection_nn_poste.setNumPoolFrames(4)
detection_nn_poste.input.setBlocking(False)
detection_nn_poste.setNumInferenceThreads(2)
cam.preview.link(detection_nn_poste.input)
#cam.setFps(3)

detection_nn_larguero = pipeline.create(dai.node.NeuralNetwork)
detection_nn_larguero.setBlob(blob_larguero)
detection_nn_larguero.setNumPoolFrames(4)
detection_nn_larguero.input.setBlocking(False)
detection_nn_larguero.setNumInferenceThreads(2)
cam.preview.link(detection_nn_larguero.input)
#cam.setFps(3)

# NN output linked to XLinkOut
xout_nn_poste= pipeline.create(dai.node.XLinkOut)
xout_nn_poste.setStreamName("nn_poste")
detection_nn_poste.out.link(xout_nn_poste.input)

xout_nn_larguero= pipeline.create(dai.node.XLinkOut)
xout_nn_larguero.setStreamName("nn_larguero")
detection_nn_larguero.out.link(xout_nn_larguero.input)

# Left mono camera
left = pipeline.create(dai.node.MonoCamera)
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
# Right mono camera
right = pipeline.create(dai.node.MonoCamera)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
left.out.link(stereo.left)
right.out.link(stereo.right)

# Create depth output
xout_disp = pipeline.create(dai.node.XLinkOut)
xout_disp.setStreamName("disparity")
stereo.disparity.link(xout_disp.input)

# Pipeline is defined, now we can connect to the device
with dai.Device() as device:
    cams = device.getConnectedCameras()
    depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
    if not depth_enabled:
        raise RuntimeError("Unable to run this experiment on device without depth capabilities! (Available cameras: {})".format(cams))
    device.startPipeline(pipeline)
    # Output queues will be used to get the outputs from the device
    q_color = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
    q_disp = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
    q_nn_poste = device.getOutputQueue(name="nn_poste", maxSize=4, blocking=False)
    q_nn_larguero = device.getOutputQueue(name="nn_larguero", maxSize=4, blocking=False)

    fps = FPSHandler()
    sync = HostSync()
    disp_frame = None
    disp_multiplier = 255 / stereo.initialConfig.getMaxDisparity()

    frame = None
    depth = None
    depth_weighted = None
    frames = {}

    while True:

        msgs = False
        if q_color.has():
            msgs = msgs or sync.add_msg("color", q_color.get())
        if q_disp.has():
            msgs = msgs or sync.add_msg("depth", q_disp.get())
        if q_nn_poste.has():
            msgs = msgs or sync.add_msg("nn_poste", q_nn_poste.get())
        if q_nn_larguero.has():
            msgs = msgs or sync.add_msg("nn_larguero", q_nn_larguero.get())

        if msgs:
            fps.next_iter()
            layer1_poste = q_nn_poste.get().getFirstLayerInt32()
            layer1_larguero = q_nn_larguero.get().getFirstLayerInt32()
            lay1_poste = np.asarray(layer1_poste, dtype=np.int32).reshape(*INPUT_SHAPE)
            lay1_larguero = np.asarray(layer1_larguero, dtype=np.int32).reshape(*INPUT_SHAPE)
            output_colors_poste = decode_deeplabv3p(lay1_poste)
            output_colors_larguero = decode_deeplabv3p(lay1_larguero)

            output_colors_poste = cv2.resize(output_colors_poste, TARGET_SHAPE)
            output_colors_larguero = cv2.resize(output_colors_larguero, TARGET_SHAPE)

            frame = q_color.get().getCvFrame()
            frame = cv2.resize(frame, TARGET_SHAPE)
            frames['frame'] = frame
            frame_poste = cv2.addWeighted(frame, 1, output_colors_poste,0.5,0)
            cv2.putText(frame_poste, "Fps: {:.2f}".format(fps.fps()), (2, frame_poste.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))
            frames['colored_frame_poste'] = frame_poste
            frame_larguero = cv2.addWeighted(frame, 1, output_colors_larguero,0.5,0)
            cv2.putText(frame_larguero, "Fps: {:.2f}".format(fps.fps()), (2, frame_larguero.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))
            frames['colored_frame_larguero'] = frame_larguero

            disp_frame = q_disp.get().getFrame()
            disp_frame = (disp_frame * disp_multiplier).astype(np.uint8)
            disp_frame = cv2.resize(disp_frame, TARGET_SHAPE)

            frames['depth'] = cv2.applyColorMap(disp_frame, jet_custom)

            multiplier_poste = get_multiplier(lay1_poste)
            multiplier_poste = cv2.resize(multiplier_poste, TARGET_SHAPE)
            depth_overlay_poste = disp_frame * multiplier_poste
            frames['cutout_poste'] = cv2.applyColorMap(depth_overlay_poste, jet_custom)

            multiplier_larguero = get_multiplier(lay1_larguero)
            multiplier_larguero = cv2.resize(multiplier_larguero, TARGET_SHAPE)
            depth_overlay_larguero = disp_frame * multiplier_larguero
            frames['cutout_larguero'] = cv2.applyColorMap(depth_overlay_larguero, jet_custom)
        
            show_larguero = np.concatenate((frames['colored_frame_larguero'],frames['cutout_larguero']), axis=1)
            show_poste = np.concatenate((frames['colored_frame_poste'],frames['cutout_poste']), axis=1)
            cv2.imshow("Segmentacion Larguero", show_larguero)
            cv2.imshow("Segmentacion Poste", show_poste)
           
        if cv2.waitKey(1) == ord('q'):
            break
