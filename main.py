import cv2
from ultralytics import YOLO
import pandas as pd
import time

model = YOLO("yolov8x-pose")

video_path = "video-class.mp4"
cap = cv2.VideoCapture(video_path)

frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)

seconds = round(frames/fps)

frame_total = 500 # You can change it to the number of frame you want to take.
i = 0
a = 0

all_data = []

# Create window display
cv2.namedWindow('YOLOv8 Inference', cv2.WINDOW_NORMAL)

# Resize window
cv2.resizeWindow('YOLOv8 Inference', 800, 600) 

while (cap.isOpened()):
  cap.set(cv2.CAP_PROP_POS_MSEC, (i * ((seconds/frame_total)*1000)))
  flag, frame = cap.read()

  if flag == False:
    break

  image_path = f'results_img/img_{i}.jpg'
  cv2.imwrite(image_path, frame)
  
  start_time = time.time()
  # YOLOv8 Will detect your video frame
  results = model(frame, verbose=False)
  
  # Visualize the results on the frame
  annotated_frame = results[0].plot()
  end_time = time.time()

  fps = 1 / (end_time - start_time)
  print("FPS :", fps)

  cv2.putText(annotated_frame, "FPS :"+str(int(fps)), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 255), 1, cv2.LINE_AA)
  
  # Display the annotated frame
  cv2.imshow("YOLOv8 Inference", annotated_frame)
  
  for r in results:
    bound_box = r.boxes.xyxy  # get the bounding box on the frame
    conf = r.boxes.conf.tolist() # get the confident it is a human from a frame
    keypoints = r.keypoints.xyn.tolist() # get the every human keypoint from a frame

    # this code for save every human that detected from 1 image, so if 1 image have 10 people, we will save 10 human picture.

    for index, box in enumerate(bound_box):
      if conf[index] > 0.5: # we do it for reduce blurry human image.
        x1, y1, x2, y2 = box.tolist()
        pict = annotated_frame[int(y1):int(y2), int(x1):int(x2)]
        output_path = f'human_picture/person_{a}.jpg'

        # we save the person image file name to csv for labelling the csv file.
        data = {'image_name': f'person_{a}.jpg'}

        # Initialize the x and y lists for each possible key
        for j in range(len(keypoints[index])):
            data[f'x{j}'] = keypoints[index][j][0]
            data[f'y{j}'] = keypoints[index][j][1]

       # we save the human keypoint that detected by yolo model to csv file to train our Machine learning model later.

        all_data.append(data)
        cv2.imwrite(output_path, pict)
        a += 1
  i += 1
  # Break the loop if 'q' is pressed
  if cv2.waitKey(1) & 0xFF ==ord('q'):
    break
  

print(i-1, a-1)
cap.release()
cv2.destroyAllWindows()

# Combine all data dictionaries into a single DataFrame
df = pd.DataFrame(all_data)

# Save the DataFrame to a CSV file
csv_file_path = 'results_csv/keypoints.csv'
df.to_csv(csv_file_path, index=False)