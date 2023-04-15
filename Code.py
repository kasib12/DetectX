import os
import cv2
import numpy as np

# Load the YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load the classes
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


# Define the path to the folder where detected objects will be saved
save_folder = 'C://Users//HP-PC//OneDrive//Desktop//detected object'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set the resolution of the webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)

    # Set the input of the network to the blob
    net.setInput(blob)

    # Get the output of the network
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Loop over the outputs
    for output in outputs:
        # Loop over the detections
        for detection in output:
            # Get the class ID and confidence
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Skip detections with low confidence
            if confidence < 0.5:
                continue

            # Get the coordinates of the bounding box
            center_x = int(detection[0] * frame.shape[1])
            center_y = int(detection[1] * frame.shape[0])
            width = int(detection[2] * frame.shape[1])
            height = int(detection[3] * frame.shape[0])
            x = int(center_x - width / 2)
            y = int(center_y - height / 2)

            # Extract the object from the frame
            object_img = frame[y:y + height, x:x + width]

            # Save the object image to disk
            save_path = os.path.join(save_folder, f'{classes[class_id]}_{confidence:.2f}.jpg')
            cv2.imwrite(save_path, object_img)

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, f"{classes[class_id]} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # Exit if the 'q' key is pressed
    if key == ord("q"):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
