import numpy as np
import tensorflow as tf
import cv2
import os

from scripts.plot_bounding_boxes_p6k import plot_bounding_box as plot_bounding_box


base_path = "../../../Data/"
rparis6k_path = os.path.join(base_path, 'datasets', 'rparis6k')

download_path = "/Users/eliainnocenti/Downloads"

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="../models/model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def prepare_image(image_path, input_shape):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_shape[1], input_shape[2]))
    img = img / 255.0  # Normalize # TODO: do I have to normalize?
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)


def detect_objects(image_path):
    input_shape = input_details[0]['shape']
    img = prepare_image(image_path, input_shape)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    # Get outputs
    boxes = interpreter.get_tensor(output_details[0]['index'])
    class_scores = interpreter.get_tensor(output_details[1]['index'])

    return boxes, class_scores


# Example usage with test set images
#image_path = os.path.join(rparis6k_path, 'images', 'paris_defense_000041.jpg')     # defense 1        # 0.6
#image_path = os.path.join(rparis6k_path, 'images', 'paris_eiffel_000170.jpg')      # eiffel 2         # 0.3
#image_path = os.path.join(rparis6k_path, 'images', 'paris_invalides_000090.jpg')   # invalides 3      # 0.2
#image_path = os.path.join(rparis6k_path, 'images', 'paris_louvre_000307.jpg')      # louvre 4         # 0.7
#image_path = os.path.join(rparis6k_path, 'images', 'paris_moulinrouge_000844.jpg') # moulinrouge 5    # 0.3
#image_path = os.path.join(rparis6k_path, 'images', 'paris_museedorsay_000802.jpg') # museedorsay 6    # 0.3
#image_path = os.path.join(rparis6k_path, 'images', 'paris_notredame_000779.jpg')   # notredame 7      # 0.2
#image_path = os.path.join(rparis6k_path, 'images', 'paris_pantheon_000982.jpg')    # pantheon 8       # 0.6
#image_path = os.path.join(rparis6k_path, 'images', 'paris_pompidou_000143.jpg')    # pompidou 9       # 0.4
#image_path = os.path.join(rparis6k_path, 'images', 'paris_sacrecoeur_000279.jpg')  # sacrecoeur 10    # 0.3 - 0.4
#image_path = os.path.join(rparis6k_path, 'images', 'paris_triomphe_000348.jpg')    # triomphe 11      # 0.4 - 0.5

# Examples usage with personal images
image_path = os.path.join(download_path, 'eiffel.jpg')
#image_path = os.path.join(download_path, 'louvre.jpg')
#image_path = os.path.join(download_path, 'pantheon.jpg')

boxes, class_scores = detect_objects(image_path)

# Post-processing
confidence_threshold = 0.2
max_boxes = 10

# Get the class with highest score for each box
class_ids = np.argmax(class_scores[0], axis=1)
confidences = np.max(class_scores[0], axis=1)

# Filter based on confidence threshold
mask = confidences > confidence_threshold
filtered_boxes = boxes[0][mask]
filtered_class_ids = class_ids[mask]
filtered_confidences = confidences[mask]

# Apply non-max suppression (you might need to implement this)
# For simplicity, let's just take the top max_boxes
top_indices = np.argsort(filtered_confidences)[-max_boxes:]

# Load the original image
original_image = cv2.imread(image_path)
height, width = original_image.shape[:2]

# Print results and draw boxes
for i in top_indices:
    # Convert normalized coordinates to pixel coordinates
    # FIXME: which format do I have? [xmin, ymin, xmax, ymax] or [xmin, ymin, width, height]?

    '''
    xmin, ymin, xmax, ymax = filtered_boxes[i]
    xmin = int((xmin + 1) * width / 2)
    xmax = int((xmax + 1) * width / 2)
    ymin = int((ymin + 1) * height / 2)
    ymax = int((ymax + 1) * height / 2)
    '''

    xmin, ymin, b_width, b_height = filtered_boxes[i]
    xmin = int((xmin + 1) * width / 2)
    ymin = int((ymin + 1) * width / 2)
    b_width = int((b_width + 1) * height / 2)
    b_height = int((b_height + 1) * height / 2)

    #plot_bounding_box(image_path, xmin, ymin, xmin + b_width, ymin + b_height)

    print(f"Detection {i + 1}:")
    print(f"  Class: {filtered_class_ids[i]}")
    print(f"  Confidence: {filtered_confidences[i]:.2f}")
    #print(f"  Bounding box (pixel coordinates): [{xmin}, {ymin}, {xmax}, {ymax}]")
    print(f"  Bounding box (pixel coordinates): [{xmin}, {ymin}, {b_width}, {b_height}]")

    # Draw rectangle on the image
    #cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.rectangle(original_image, (xmin, ymin), (b_width, b_height), (0, 255, 0), 2)

    # Put class label and confidence score
    label = f"Class {filtered_class_ids[i]}: {filtered_confidences[i]:.2f}"
    cv2.putText(original_image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Save the image with bounding boxes
#cv2.imwrite('output_image.jpg', original_image)
#print("Output image saved as 'output_image.jpg'")
