import pprint
import time
import os
import numpy as np
import tensorflow as tf
import cv2

class edgeSR(object):
    def __init__(self, tflite_path:str):
        if not os.path.exists(tflite_path):
            raise FileNotFoundError(tflite_path)
        try:
            self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
            print(f"# success to load tflite model({tflite_path})")
        except:
            raise ValueError(f"! Failed to load tflite file({tflite_path}) ...")
        self.details = {"input_details" : self.interpreter.get_input_details(), \
            "output_details" : self.interpreter.get_output_details()}

    def get_details(self):
        pprint.pprint(self.details)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x:np.ndarray):
        """
        x:np.ndarray -> np.ndarray
        """
        self.interpreter.allocate_tensors()
        batched_input = np.expand_dims(x, axis=0)
        self.interpreter.set_tensor(self.details["input_details"][0]['index'], batched_input)
        self.interpreter.invoke()
        batched_output = self.interpreter.get_tensor(self.details["output_details"][0]['index'])
        output = np.squeeze(batched_output, axis=0)
        return output
    
    def benchmark(self, x:np.ndarray, step:int=10):
        elapsed = 0.0
        for i in range(step):
            start = time.time()
            self.forward(x)
            elapsed += time.time() - start
            time.sleep(0.001)
        print(f"# BENCHMARK RESULTS >> {step / elapsed:.3f} frame/s")
            
    def saveSRImg(self, x:np.ndarray, name:str="output.jpg"):
        output = self.forward(x)
        try:
            cv2.imwrite(name, output)
            print(f"# success to save img !!!")
        except:
            print(f"! failed to save img ...")    
        

if __name__ == "__main__":
    sr_engine = edgeSR("base7_D4C28_bs16ps64_lr1e-3_qat_time.tflite")
    sr_engine.get_details()
    x = cv2.imread("input_1.jpg", cv2.IMREAD_COLOR)
    sr_engine.saveSRImg(x)
    sr_engine.benchmark(x)
    