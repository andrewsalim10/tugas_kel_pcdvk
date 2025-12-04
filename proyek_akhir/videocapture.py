import cv2

import time
from threading import Thread

# time.sleep(5) # ga bisa buat delay

# Open the default camera
cam = cv2.VideoCapture("http://stream.cctv.malangkota.go.id/WebRTCApp/streams/636589114401733445781917.m3u8?token=null")
# cam = cv2.VideoCapture("http://stream.cctv.malangkota.go.id/WebRTCApp/streams/385150101635081489243344.m3u8")
# cam = cv2.VideoCapture("http://stream.cctv.malangkota.go.id/WebRTCApp/play.html?name=385150101635081489243344")
# cam = cv2.VideoCapture("https://streams.videolan.org/samples/MPEG-4/video.mp4")

# cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Get the default frame width and height
# frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2

print(frame_width)
print(frame_height)


# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))


# fps
fps = 1/60
fps_ms = int(fps * 1_000)

# def update():
#      if cam.isOpened():
          

while True:
    ret, frame = cam.read()
    # time.sleep(fps)

    if ret:
        # Write the frame to the output file
        # out.write(frame)
        cv2.imwrite("lalu_lintas.png", frame)

        # resize
        frame = cv2.resize(frame, dsize=(frame_width, frame_height))

        # crop center
        # frame = center_crop(frame, (224, 224))
        
        # Display the captured frame
        cv2.imshow('Camera', frame)
        time.sleep(fps)

    # Press 'q' to exit the loop
    # if cv2.waitKey(1) == ord('q'):
    if cv2.waitKey(fps_ms) & 0xFF == ord('q'):
        break

# Release the capture and writer objects
cam.release()
# out.release()
cv2.destroyAllWindows()

### ### ###

# import cv2
# import numpy as np
# import requests

# # url = r'https://i.imgur.com/DrjBucJ.png'
# url = r'https://i.pinimg.com/236x/bf/1a/03/bf1a031f0f65e284e2d500009ab78263.jpg'

# resp = requests.get(url, stream=True).raw
# image = np.asarray(bytearray(resp.read()), dtype="uint8")
# image = cv2.imdecode(image, cv2.IMREAD_COLOR)

# # while True:
#     # for testing
# cv2.imshow('image',image)

# cv2.waitKey(0)
#     # if cv2.waitKey(1) == ord('q'):
#     #     break

# cv2.destroyAllWindows()
