import cv2
import numpy as np
import torch
import os
import random

from torchvision.models.mobilenetv3 import _mobilenet_v3_conf
from torchvision import models
from torchvision import transforms
from library_buatan.mask_rcnn_mobilenetv3s import model_maskrcnnn
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

target_predict = {0:'Apple_scab',
                1:'Black_rot',
                2:'Cedar_apple_rust',
                3:'healthy'}
target_detection = [
    "Disease", "Disease", "Disease", "Leaf"
]

inverted_residual_setting, last_channel = _mobilenet_v3_conf('mobilenet_v3_small')
img_predict = models.MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                    last_channel=last_channel,
                    num_classes=4)

obj_detection = model_maskrcnnn()

img_predict.load_state_dict( torch.load(
    os.path.join(os.getcwd(), "hasil/mobilenetv3small/apple_leaf_disease_20_mobilenetv3small_Adam_default_outchan1280_16batch/apple_leaf_disease_20_mobilenetv3small_Adam_default_outchan1280_16batch.pth")
    ) )
obj_detection.load_state_dict(torch.load(
    os.path.join(os.getcwd(), f"model_epoch_mask_rcnn50_normalize.pth")
    ) )

img_predict.to(device)
obj_detection.to(device)

img_predict.eval()
obj_detection.eval()

path = 'apple_leaf_dataset/rcnn/valid_rcnn'
target_folders = os.listdir(path)

combined_image = 0
for idx, target_folder in enumerate(target_folders):
    files = os.listdir( os.path.join( path, target_folder) )
    length = len(files)
    file = files[ random.randint(0, length-1)]

    # obj_detection.eval()
    img_path = os.path.join(path, target_folder, file)

    image_bgr = cv2.imread(img_path)
    # image_bgr = cv2.resize(image_bgr, dsize=(224,224))

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    # Transform image to tensor and add batch dimension
    transform = transforms.Compose([
        # transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize( (.5,.5,.5), (.5,.5,.5) ),
        ])

    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    # image_tensor = torch.from_numpy(image_rgb).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        predictions = obj_detection(image_tensor)
        outputs = img_predict(image_tensor)

    # Classification
    _ , classification = outputs.max(1)
    classification = classification.view(-1).cpu()

    # Extract masks, boxes, labels, and scores
    masks = predictions[0]['masks']       # [N, 1, H, W]
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    threshold = 0.4  # Confidence threshold

    # Use overlay for blending masks over image
    overlay = image_bgr.copy()

    counting = 0
    for i in range(len(masks)):
        if scores[i] >= threshold:
            # Convert mask to uint8 numpy array (H,W)
            mask = masks[i, 0].mul(255).byte().cpu().numpy()
            mask_bool = mask > 127  # binary mask for indexing
            box = boxes[i].cpu().numpy().astype(int)
            class_name = target_detection[labels[i]]
            score = scores[i].item()

            if class_name == "Disease":
                counting += 1
            else:
                continue

            # Generate random color (BGR)
            color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()

            # Create colored mask with the random color
            colored_mask = np.zeros_like(image_bgr, dtype=np.uint8)
            for c in range(3):
                colored_mask[:, :, c] = mask_bool * color[c]

            # Alpha blend the colored mask onto the overlay
            alpha = 0.4
            overlay = np.where(mask_bool[:, :, None],
                            ((1 - alpha) * overlay + alpha * colored_mask).astype(np.uint8),
                            overlay)

            # Draw bounding box and label text on overlay
            # x1, y1, x2, y2 = box
            # cv2.putText(overlay, f"{class_name}: {score:.2f}", (x1, y1 - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, lineType=cv2.LINE_AA)
            # cv2.putText(overlay, f"{class_name}: {score:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.4, color, 1, lineType=cv2.LINE_AA)

    cv2.putText(overlay, f"Disease: {counting}", (0, overlay.shape[1]-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255,0,255), 1, lineType=cv2.LINE_AA)
    cv2.putText(overlay, f"Prediction: {target_predict[int(classification)]}", (0, 0+10), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255,0,255), 1, lineType=cv2.LINE_AA)
    cv2.putText(overlay, f"Label: {target_folder}", (0, 0+20), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255,0,255), 1, lineType=cv2.LINE_AA)

    # Show the result using matplotlib (convert BGR -> RGB)
    result_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    if idx == 0:
        combined_image = result_rgb
    else:
        try:
            combined_image = np.hstack((combined_image, result_rgb))         
        except:
            continue

# Concatenate images horizontally
# combined_image = np.hstack((img1, img2)) 

# Display the combined image in a single window
cv2.imshow('Combined Images', combined_image)

# Wait for a key press (misal pencet q)
cv2.waitKey(0)

# Destroy all windows
cv2.destroyAllWindows()