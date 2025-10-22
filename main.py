import os
import boto3
import cv2

output_dir = './data'
output_dir_imgs = os.path.join(output_dir, 'imgs')
output_dir_annotations = os.path.join(output_dir, 'annotations')

# Create the AWS Rekognition client
reko_client = boto3.client('rekognition')

# Set the target class 
target_class = "Zebra"

# Load and process the video
video = cv2.VideoCapture('./media/animals.mp4')
frame_no = -1 # keep track of which frame is currently being processed

ret = True
while ret:
    ret, frame = video.read()

    if ret:
        frame_no += 1
        H, W, _ = frame.shape # parse the height and width of each frame

        # Convert the current frame to a jpg image
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()

        # Detect objects using Amazon Rekognition
        response = reko_client.detect_labels(Image={'Bytes': image_bytes},
                                        MinConfidence=70)
        
        with open(os.path.join(output_dir_annotations, 'frame_{}.txt'.format(str(frame_no).zfill(6))), 'w') as f:
            for label in response['Labels']:
                if label['Name'] == target_class:
                    for instance_no in range(len(label['Instances'])):
                        bounding_box = label['Instances'][instance_no]['BoundingBox']

                        # The 'Left' and 'Top' of the bounding box returns absolute coordinates which range from 0-1
                        x1, y1 = bounding_box['Left'], bounding_box['Top']
                        width, height = bounding_box['Width'], bounding_box['Height']

                        # Write the detections in YOLO format
                        f.write('{} {} {} {} {}'.format(0, (x1 + width / 2), (y1 + height / 2), width, height))

                        # Add the bounding boxes to the images
                        x1, y1 = int(bounding_box['Left'] * W), int(bounding_box['Top'] * H)
                        width, height = int(bounding_box['Width'] * W), int(bounding_box['Height'] * H)
                        cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 3)

            f.close()
        
        # Save all the images 
        cv2.imwrite(os.path.join(output_dir_imgs, 'frame_{}.jpg'.format(str(frame_no).zfill(6))), frame)
        
