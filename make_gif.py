import numpy as np
import glob
import re
import imageio

def img_file_to_gif(img_file_lst, output_file_name):
    imgs_array = [np.array(imageio.imread(img_file)) for img_file in img_file_lst]
    imageio.mimsave(output_file_name, imgs_array, duration=0.5)

if __name__ == '__main__':
    result_file = sorted(glob.glob('./result/*.*'), key = (lambda x : int(re.findall('\d+',x)[0])))
    variance_file = sorted(glob.glob('./variance/*.*'), key = (lambda x : int(re.findall('\d+',x)[0])))

    img_file_to_gif(result_file, "./result/result.gif")
    img_file_to_gif(variance_file, "./variance/variance.gif")