from PIL import Image, ImageDraw
import numpy as np

image_path = "merah_hijau.jpg"   
image = Image.open(image_path).convert("RGB")

w, h = image.size

mask = Image.new("L", (w, h), 0)  
draw = ImageDraw.Draw(mask)

box_w, box_h = w // 3, h // 3  
x1, y1 = (w - box_w) // 2, (h - box_h) // 2
x2, y2 = x1 + box_w, y1 + box_h
draw.rectangle([x1, y1, x2, y2], fill=255)

masked_array = np.array(image)
masked_array[y1:y2, x1:x2] = [0, 0, 0]  
masked_image = Image.fromarray(masked_array)

mask.save("empty_mask.png")          
masked_image.save("masked_center.png")  

