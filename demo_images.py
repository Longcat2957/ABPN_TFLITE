import os
import cv2
import argparse
import time
from benchmark_single_img import edgeSR
from utils import scale_image
parser = argparse.ArgumentParser()
parser.add_argument(
    "--tflite", type=str, default="./tflite/edgeSR_mslug.tflite"
)
parser.add_argument(
    "--img", type=str, default="input_1.jpg"
)


if __name__ == "__main__":
    opt = parser.parse_args()
    img_path = os.path.join("./sample_images", opt.img)
    model = edgeSR(tflite_path=opt.tflite)
    
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    preprocessed = model.preprocess(img)
    model.saveSRImg(preprocessed, "./output/mslug.jpg")
    bicubic = scale_image(img, 3.0)
    bicubic_path = os.path.join("./output", f"bicubic_output_mslug.jpg")
    cv2.imwrite(bicubic_path, bicubic)