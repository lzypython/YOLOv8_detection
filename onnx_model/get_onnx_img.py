import numpy as np
import cv2

import logging

CLASSES = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup", "Dog", "Motorbike", "People", "Table"]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


class BannerDetector:
    def __init__(self, model_path=None):
        # logging.info(f"start init model")
        self.model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(model_path)
        self.img_det_size = 640

    def __call__(self, image, typeneed =0,score_threshold=0.9):
        blob = self.img_process(image)
        self.model.setInput(blob)
        outputs = self.model.forward()

        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2], outputs[0][i][3]]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, 0.45, 0.5)
        detections = []
        # logging.info(f"result_boxes is : {result_boxes}")
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            # print(box)
            x1, y1, x2, y2 = round(box[0] * self.scale), round(box[1] * self.scale), round(
                (box[0] + box[2]) * self.scale), round((box[1] + box[3]) * self.scale)
            box = [x1, y1, x2, y2]
            detection = {
                'class_id': class_ids[index],
                'class_name': CLASSES[class_ids[index]],
                'confidence': scores[index],
                'box': box,
                'scale': self.scale}
            detections.append(detection)
            print(detection)
            draw_bounding_box(image, class_ids[index], scores[index], x1, y1, x2, y2)
        if(typeneed==0):
            cv2.imwrite('images/tmp/single_result.jpg', image)
        else:
            cv2.imwrite('images/tmp/single_result_vid.jpg', image)

        return detections

    def img_process(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_image: np.ndarray = img
        [height, width, _] = original_image.shape
        length = max((height, width))
        logging.info(f"img shape:{height, width}")
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image
        self.scale = length / self.img_det_size

        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        return blob


def getimg(image,typeneed):
    # image = cv2.imread("2015_00062.jpg")
    import time, copy
    model = BannerDetector(model_path="./onnx_model/best.onnx")
    # print("yes")
    bboxes = model(image, typeneed =typeneed,score_threshold=0.5)
    cv_image_copy = copy.deepcopy(image)
