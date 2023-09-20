import tkinter as tk
from tkinter import filedialog
import os
import glob
import cv2
from PIL import Image, ImageTk
import numpy as np
import albumentations as A
import torch


def get_band(img, ch):
    band = img.read(ch)
    band = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return band

height = 1024
width = 1024

root = tk.Tk()
root.title('demo interface')


origin_image_frame = tk.LabelFrame(root, padx=10, pady=10)
predict_image_frame = tk.LabelFrame(root, padx=10, pady=10)
info_frame = tk.LabelFrame(root, padx=10, pady=10)

origin_image_frame.grid(row=0, column=0)
predict_image_frame.grid(row=0, column=1)
info_frame.grid(row=0, column=2)

# origin_image_frame

canvas = tk.Canvas(origin_image_frame, width=512, height=512)
# open_button = tk.Button(origin_image_frame, text='open file', command=open_dir).grid(row=0, column=0)
dir = filedialog.askdirectory(initialdir=os.getcwd())
image_list = glob.glob(os.path.join(dir, '*.png'))
img = cv2.imread(image_list[0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
aug = A.Resize(height, width, interpolation=cv2.INTER_NEAREST)(image=img)
img = aug['image']
PIL_img = Image.fromarray(aug['image'])
tk_img = ImageTk.PhotoImage(PIL_img)
canvas.create_image(0, 0, image=tk_img)
canvas.pack()


# predict_iamge_frame

def multiClassToVisualize(image):
    output = np.array(image.shape)
    output = np.argmax(image, 0)
    return output

model = torch.load(os.path.join('model', 'best_model.pth'), map_location=torch.device('cpu'))
model.eval()
canvas = tk.Canvas(predict_image_frame, width=512, height=512)
img = img.transpose([2, 0, 1]).astype(np.double)
img = torch.from_numpy(img).to('cpu').unsqueeze(0).to(torch.float32)
result = model.predict(img).cpu().numpy()
# result = result.reshape(img.size[:2])
result = (result.squeeze().round())
result = multiClassToVisualize(result)
result[result == 0] = 0
result[result == 1] = 122
result[result == 2] = 255

# result[:,:33]=3   #將不屬於原圖的padding刪除（用於正射影像）
# result[:,990:]=3
PIL_result = Image.fromarray(np.uint8(result))
tk_result = ImageTk.PhotoImage(PIL_result)
canvas.create_image(0, 0, image=tk_result)
canvas.pack()




# info frame

def mosaic_statis(img, resolution, RGB=False):
    # img: grayscal img - numpy array
    # resolution: 10 -> 10x10 resolution
    img = A.Resize(1000, 1000, interpolation=cv2.INTER_NEAREST)(image=img)['image']
    print(img.shape)
    img[img>0] = 1
    h, w = img.shape
    grid_height = h // resolution
    grid_width = w // resolution
    # fig, ax = plt.subplots(10, 10, figsize=(10, 10))
    image_reshaped = img.reshape(resolution, grid_height, resolution, grid_width)
    perc = np.count_nonzero(image_reshaped==1, axis=(1, 3))/grid_height/grid_width
    perc[perc>0]=1
    level = int(perc.sum()*10/resolution/resolution)

    # plt.tight_layout()
    # plt.show()
    return level


def quality():
    if predict_level.get() == '0':
        show_quality = tk.Label(info_frame, text='無雲(b)').grid(row=1, column=1, columnspan=6)
    else:
        show_quality = tk.Label(info_frame, text='有雲(a)').grid(row=1, column=1, columnspan=6)

level = mosaic_statis(result, 10)
predict_level = tk.StringVar(value=level)
img_name = tk.Label(info_frame, text=os.path.basename(image_list[0]).split(',')[0]).grid(row=0, column=0, columnspan=8)

image_quality_text = tk.Label(info_frame, text='Quality').grid(row=1, column=0)
show_quality = tk.Label(info_frame, text='有雲(a)' if predict_level.get() != 0 else '無雲(b)').grid(row=1, column=1, columnspan=6)

predict_level_text = tk.Label(info_frame, text='predict level').grid(row=2, column=0)
show_level = tk.Label(info_frame, textvariable=predict_level).grid(row=2, column=1, columnspan=6)

for i in range(5):
    tk.Radiobutton(info_frame, text=str(i), variable=predict_level, value=str(i), command=quality).grid(row=3, column=i+1)
tk.Radiobutton(info_frame, text='9', variable=predict_level, value='9', command=quality).grid(row=3, column=7)


comfirm_button = tk.Button(info_frame, text='comfirm').grid(row=4, column=0, columnspan=6)

# next & previous button

previous_button = tk.Button(root, text="Previous").grid(row=2, column=0)
next_button = tk.Button(root, text='Next').grid(row=2, column=1)


root.mainloop()