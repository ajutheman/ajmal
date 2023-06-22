import cv2
import numpy as np


def detect_objects(image_path, np=None):
    # Load the pre-trained model for object detection
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'model.caffemodel')

    # Load the image
    image = cv2.imread(image_path)

    # Prepare the input blob for the network
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # Set the input for the network
    net.setInput(blob)

    # Perform forward pass through the network
    detections = net.forward()

    # Iterate over the detected objects
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by confidence threshold
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])

            # Get the bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box and label on the image
            label = f"Object {i+1}"
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with detected objects
    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Provide the image path for object detection
image_path = "path/to/your/image.jpg"
detect_objects(image_path)
