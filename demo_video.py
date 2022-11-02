import os
import cv2
import argparse
import time
from benchmark_single_img import edgeSR
from utils import scale_image
parser = argparse.ArgumentParser()
parser.add_argument(
    "--tflite", type=str, default="./tflite/edgeSR_120_360.tflite"
)
parser.add_argument(
    "--video", type=int, default=0
)


if __name__ == "__main__":
    opt = parser.parse_args()
    model = edgeSR(tflite_path=opt.tflite)
    
    started = time.time()
    last_logged = time.time()
    frame_count = 0
    sample_video_path = os.path.join("./sample_videos", f"sample_video_{opt.video}.mp4")
    print(f"# input video file name = {sample_video_path}")
    cap = cv2.VideoCapture(sample_video_path)

    while True:
        ret, image = cap.read()
        
        if not ret:
            break
        preprocessed = model.preprocess(image)
        sr = model.forward(preprocessed)
        cv2.imshow("lr", image)
        cv2.imshow("sr", sr)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()