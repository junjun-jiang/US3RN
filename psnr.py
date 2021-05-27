import numpy as np
import math


def MPSNR(img1, img2):
    ch = np.size(img1,2)
    if ch == 1:
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        s = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        return s
    else:
        sum = 0
        for i in range(ch):
            mse = np.mean((img1[:,:,i] - img2[:,:,i]) ** 2)
            if mse == 0:
                return 100
            PIXEL_MAX = 1.0
            s = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
            sum = sum + s
        s = sum / ch
        return s



