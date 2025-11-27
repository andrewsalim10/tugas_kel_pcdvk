import cv2
import numpy as np
import os
import glob
import random
import matplotlib.pyplot as plt

output_color_path = os.path.join(base_path, 'color_segmented_rcnn')
print(f"Saved to : {output_color_path}")

lower_hsv = np.array([10, 40, 40])   
upper_hsv = np.array([90, 255, 255]) 

for split, folders in source_dirs.items():
    for folder_path in folders:
        if not os.path.exists(folder_path): continue

        class_name = os.path.basename(folder_path)
        output_folder = os.path.join(output_color_path, split, class_name)
        os.makedirs(output_folder, exist_ok=True)

        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))

        if not image_files: continue

        sample_file = random.choice(image_files)
        img_sample = cv2.imread(sample_file)
        if img_sample is not None:
        
            hsv_sample = cv2.cvtColor(img_sample, cv2.COLOR_BGR2HSV)
            mask_sample = cv2.inRange(hsv_sample, lower_hsv, upper_hsv)
            masked_img_sample = cv2.bitwise_and(img_sample, img_sample, mask=mask_sample)
            gray_masked_sample = cv2.cvtColor(masked_img_sample, cv2.COLOR_BGR2GRAY)

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(img_sample, cv2.COLOR_BGR2RGB)); plt.title("Original")
            plt.subplot(1, 3, 2); plt.imshow(mask_sample, cmap='gray'); plt.title("Color Mask")
            plt.subplot(1, 3, 3); plt.imshow(gray_masked_sample, cmap='gray'); plt.title("Final Clean Grayscale")
            plt.axis('off')
            plt.show()
   

        for file_path in image_files:
            try:
                img = cv2.imread(file_path)
                if img is None: continue
          
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
                masked_img = cv2.bitwise_and(img, img, mask=mask)
                final_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

                cv2.imwrite(os.path.join(output_folder, os.path.basename(file_path)), final_gray)
            except Exception as e: print(e)
