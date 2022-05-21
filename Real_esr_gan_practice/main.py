# from PIL import Image
import os
from inference_realesrgan import *

if __name__ == '__main__':
    
    # 경로 찾기
    absolute_path = os.path.abspath(__file__)
    print(os.path.abspath(__file__))
    dir_path = os.path.dirname(absolute_path)
    dir_image_path = os.path.join(dir_path,'image')
    print(dir_image_path)
    
    real_esrgan(input=dir_image_path,outscale=3.5, suffix='')
   
    # dir_image = "./image"
    # dir_path = os.path.join()