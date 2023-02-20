import os
import shutil as sh
import cv2
import pandas as pd
from PIL import Image


def cv2_PIL(original_img):
        img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return img


def df_prediction(bbox_text, image_folder):
    df_prediction = pd.read_csv(bbox_text, sep=' ', header=None)
    df_prediction.columns = ['classes', 'x_center', 'y_center', 'w', 'h', 'img_id']

    img_w = []
    img_h = []
    for i in df_prediction.img_id:
        img_path = os.path.join(image_folder, i) + '.png'
        img = Image.open(img_path)
        width, height = img.size
        img_w.append(width)
        img_h.append(height)

    df_prediction['img_w'] = img_w
    df_prediction['img_h'] = img_h
    df_prediction['xmin'] = ((df_prediction['x_center'] - df_prediction['w'] / 2) * df_prediction.img_w).astype(int)
    df_prediction['xmax'] = ((df_prediction['w'] / 2 + df_prediction['x_center']) * df_prediction.img_w).astype(int)
    df_prediction['ymin'] = ((df_prediction['y_center'] - df_prediction['h'] / 2) * df_prediction.img_h).astype(int)
    df_prediction['ymax'] = ((df_prediction['h'] / 2 + df_prediction['y_center']) * df_prediction.img_h).astype(int)

    return df_prediction


def draw_bbox(bbox_text, image_folder, bbox_save_path, df_prediction):
    # sh.rmtree(img_ext_path)  # delete this folder
    # if file don't exist, then create
    try:
        os.makedirs(bbox_save_path)
    except:
        print ("Directory already exist.")

    # color of each class
    class_dic = {'title':0, 'bar_bbox':1, 'x_tick_bbox':2, 'x_label_bbox':3, 'y_tick_bbox':4, 'y_label_bbox':5}
    class_color = {'title':(255,0,0), 'bar_bbox':(0,255,0), 'x_tick_bbox':(255,255,0), \
            'x_label_bbox':(0,255,255), 'y_tick_bbox':(0,255,255), 'y_label_bbox':(255,0,255)}
    # Red	#FF0000	(255,0,0)
    # Lime	#00FF00	(0,255,0)
    # Blue	#0000FF	(0,0,255)
    # Yellow	#FFFF00	(255,255,0)
    # Cyan / Aqua	#00FFFF	(0,255,255)
    # Magenta / Fuchsia	#FF00FF	(255,0,255)

    # draw
    bbox_thick = 1
    # get each image
    for i in set(df_prediction.img_id.values):
        img = cv2.imread(os.path.join(image_folder, i + '.png'))
        height, width, channels = img.shape
        # thick = int((height + width) // 900)
        row_id = df_prediction[df_prediction.img_id == i]
        
        # get each class in one image 
        for c in set(row_id.classes.values):
            label = [k for k, v in class_dic.items() if v == c]
            label = ''.join(label)
            row_class = row_id[row_id.classes == c]

            # daw bbox in one class
            for j, row in row_class.iterrows():
                start_point = (int(row.xmin), int(row.ymin))
                end_point = (int(row.xmax), int(row.ymax))

                # RGB --> BGR
                color = class_color[label]
                lst = list(color)
                lst[0], lst[-1] = lst[-1], lst[0]
                color = tuple(lst)

                # draw rectangle
                im = cv2.rectangle(img, start_point, end_point, color, thickness=bbox_thick)
                cv2.putText(img, label, (int(row.xmin)-bbox_thick-1, int(row.ymin)-bbox_thick-1), fontFace=0, \
                            fontScale=0.001*height, color=color, thickness=bbox_thick)
                # plt.imshow(cv2_PIL(im))
    
        cv2.imwrite(os.path.join(bbox_save_path, i + '.png'), im)