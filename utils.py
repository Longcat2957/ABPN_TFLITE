import cv2
import numpy as np

def center_crop(img:np.ndarray, dim:tuple):
    """Returns center cropped image

    Args:
        img (np.ndarray): image to be center cropped
        dim (tuple): dimensions (width, height) to be cropped
    """
    flag = None
    # if batched input ...
    if len(img.shape) == 4:
        img = np.squeeze(img, axis=0)
        flag = True
    else:
        flag = False
    
    width, height = img.shape[1], img.shape[0]      # cv2.imread -> np.ndarray (H, W, C)
    if dim[0] <= img.shape[1]:
        crop_width = dim[0]
    else:
        raise ValueError(f"cropped_width is bigger than original size")
    if dim[1] <= img.shape[0]:
        crop_height = dim[1]
    else:
        ValueError(f"cropped_height is bigger than original size")
    mid_x, mid_y = int(width//2), int(height//2)
    cw2, ch2 = int(crop_width//2), int(crop_height//2)
    cropped_img = img[mid_y-ch2:mid_y+ch2+crop_height%2, mid_x-cw2:mid_x+cw2+crop_width%2]
    if flag:
        cropped_img = np.expand_dims(cropped_img, axis=0)
    return cropped_img

def scale_image(img:np.ndarray, factor:float=1.0):
    """Returns resize image by scale factor

    Args:
        img (np.ndarray): image to be scaled
        factor (float, optional): scale factor to resize. Defaults to 1.0.
    """
    flag = None
    if len(img.shape) == 4:
        img = np.squeeze(img, axis=0)
        flag = True
    else:
        flag = False

    if flag:
        cropped_img = np.expand_dims(cropped_img, axis=0)
    return cv2.resize(img,(int(img.shape[1]*factor), int(img.shape[0]*factor)))

if __name__ == "__main__":
    # Test my cv2 utilities
    orig_img = cv2.imread("./sample_images/original_img.png", cv2.IMREAD_COLOR)
    center_cropped = center_crop(orig_img, (300, 300))
    print(center_cropped.shape)