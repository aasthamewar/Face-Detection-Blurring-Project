import os
import argparse
import cv2
import mediapipe as mp



def process_img(img, face_detection):
    H, W, _ = img.shape  # Get image dimensions
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)  # Corrected from W to H
            w = int(w * W)
            h = int(h * H)  #  Corrected from W to H

            # Added padding to fully cover the face
            padding = 20
            x1_p = max(x1 - padding, 0)
            y1_p = max(y1 - padding, 0)
            x2_p = min(x1 + w + padding, W)
            y2_p = min(y1 + h + padding, H)

            # Apply blur to the padded face region
            img[y1_p:y2_p, x1_p:x2_p, :] = cv2.blur(img[y1_p:y2_p, x1_p:x2_p, :], (30, 30))

    return img


args = argparse.ArgumentParser()

args.add_argument("--mode", default='webcam')  #  Set default to 'video'
args.add_argument("--filePath", default=None)  # Default video path

args = args.parse_args()

output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:

    if args.mode in ["image"]:
        # Read image from path
        img = cv2.imread(args.filePath)
        img = process_img(img, face_detection)

        # Save processed image
        cv2.imwrite(os.path.join(output_dir, 'output.png'), img)

    elif args.mode in ['video']:
        # Open video file
        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        # Create VideoWriter with same resolution as input
        output_video = cv2.VideoWriter(
            os.path.join(output_dir, 'output.mp4'),
            cv2.VideoWriter_fourcc(*'MP4V'),
            25,
            (frame.shape[1], frame.shape[0])
        )

        while ret:
            # Process each frame
            frame = process_img(frame, face_detection)
            output_video.write(frame)
            ret, frame = cap.read()

        # Release resources
        cap.release()
        output_video.release()


    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(0) #using 1 webcam
        ret, frame = cap.read()

        while ret:
            frame = process_img(frame, face_detection)
            cv2.imshow('frame',frame)
            cv2.waitKey(25)
            ret, frame = cap.read()

        cap.release()