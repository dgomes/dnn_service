import os
import time
import logging
import queue
import json

import paho.mqtt.client as mqtt
import requests
import cv2 as cv
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

DBG_LEVEL = os.getenv('LOGGING_LEVEL')
MQTT_BASE_TOPIC = os.getenv('MQTT_BASE_TOPIC', 'dnn_fs_monitor')
MQTT_SERVER = os.getenv('MQTT_SERVER', '192.168.1.100')
HA_PUSH_URL = os.getenv('HA_PUSH_URL', "http://192.168.1.10:8123/api/webhook/last_motion")
DETECT_OBJECTS = os.getenv('DETECT_OBJECTS', '["person"]')

PATH = "/monit"

dbg_lvl = logging.INFO
if DBG_LEVEL and "debug" in DBG_LEVEL.lower():
    dbg_lvl = logging.DEBUG
logging.basicConfig(
    level=dbg_lvl, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

try:
    detect_objs = json.loads(DETECT_OBJECTS)
except:
    logging.error("Not a valid JSON list: %s", DETECT_OBJECTS)
    detect_objs = []



class MotionEyeEventHandler(FileSystemEventHandler):
    def __init__(self, workqueue):
        super().__init__()
        self.workqueue = workqueue

    def on_created(self, event):
        logging.debug("%s - %s", event.src_path, event.event_type)
        if event.src_path.endswith("jpg"): 
            self.workqueue.put(event.src_path)


class DNNService(object):
    def __init__(self, net, classes, outputs_names):
        self.net = net
        self.classes = classes
        self.outputs_names = outputs_names

    def process(self, img, thr_score=0.5, thr_nms=0.4, size=320):
        cap = cv.VideoCapture(img)

        has_frame, frame = cap.read()
        if not has_frame:
            logging.error("no frame")
            return None

        frame_height, frame_width, *frame_info = frame.shape

        blob = cv.dnn.blobFromImage(frame, 1 / 255, (size, size), 0, True, crop=False)

        tstart = time.time()
        self.net.setInput(blob)
        outs = self.net.forward(self.outputs_names)
        tend = time.time()

        logging.debug("Inference time: %.2f s" % (tend - tstart))

        result = self.postprocess(
            self.net,
            self.classes,
            frame,
            outs,
            thr_score,
            thr_nms,
            frame_width,
            frame_height,
        )
        cap = None
        logging.info("%s", result)
        return result

    def postprocess(self, net, classes, frame, outs, thr_score, thr_nms, frame_width, frame_height):
        predictions = []

        layer_names = net.getLayerNames()
        last_layer_id = net.getLayerId(layer_names[-1])
        last_layer = net.getLayer(last_layer_id)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence >= thr_score:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv.dnn.NMSBoxes(boxes, confidences, thr_score, thr_nms)

        for i in indices:
            i = i[0]
            left, top, width, height = boxes[i]

            predictions.append(
                {
                    "class": str(classes[class_ids[i]]),
                    "box": {
                        "x": left,
                        "y": top,
                        "x1": left + width,
                        "y1": top + height,
                    },
                    "confidence": round(float(confidences[i]), 3),
                }
            )

        return predictions

def on_connect(client, properties, flags, result):
    client.publish(MQTT_BASE_TOPIC+"/status","online",retain=True)
    client.subscribe(MQTT_BASE_TOPIC+"/arm", 0)

def on_message(client, userdata, message):
    logging.debug("on_message %s = %s", message.topic, str(message.payload))
    if "true" in str(message.payload).lower():
        userdata['armed'] = True
    else:
        userdata['armed'] = False
    logging.info("System is %s", "Armed" if userdata['armed'] else "Disarmed")

def main():
    properties = {'armed': True}

    net = cv.dnn.readNetFromDarknet("/opt/yolov3.cfg", "/opt/yolov3.weights")
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    layer_names = net.getLayerNames()
    outputs_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    with open("/opt/coco.names", "rt") as class_names:
        classes = class_names.read().rstrip("\n").split("\n")

    logging.info("Starting DNN Service")

    dnn = DNNService(net, classes, outputs_names)

    workqueue = queue.Queue()

    event_handler = MotionEyeEventHandler(workqueue)
    observer = Observer()
    observer.schedule(event_handler, PATH, recursive=True)
    observer.start()
    logging.info("Monitoring %s", PATH)

    mqttc = mqtt.Client(client_id="dnn_fs_monitor", userdata=properties)
    mqttc.will_set(MQTT_BASE_TOPIC+"/status", "offline",retain=True)
    mqttc.on_connect = on_connect
    mqttc.on_message = on_message

    mqttc.connect(MQTT_SERVER)
    mqttc.loop_start()

    try:
        while True:
            image_file = workqueue.get()
            if workqueue.qsize() > 4:
                # Backlog ? lets skip half the images
                for i in range(3):
                    image_file = workqueue.get()
                    logging.info("Backlog! Skipping %s", image_file)
                
            if not properties['armed']:
                logging.info("Disarmed, skipping %s", image_file)
                continue
            logging.info("DNN'ing %s", image_file)
            res = dnn.process(image_file)
            if res:
                for obj in res:
                    if obj["class"] in detect_objs:
                        with open(image_file, "rb") as positive_image:
                            image_path = os.path.basename(image_file)
                            files = {"image": (image_path, positive_image, "image/jpeg")}
                            r = requests.post(HA_PUSH_URL, files=files)
            workqueue.task_done()
    except KeyboardInterrupt:
        observer.stop()
        mqttc.loop_stop()
    observer.join()
    workqueue.join()


if __name__ == "__main__":
    main()
