import os
import io 
from google.cloud import vision
from icecream import ic
import pandas as pd


def prepare_image_local(image_path):
    try:
        # Loads the image into memory
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        return image
    except Exception as e:
        print(e)
        return


def get_text(file_name):
    image = prepare_image_local(file_name)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/schnappiiii/Downloads/vision_api.json'
    client = vision.ImageAnnotatorClient()
    response_text = client.text_detection(image=image)

    text = []
    ymin = []
    xmin = []
    ymax = []
    xmax = []
    height = []
    width = []

    for r in response_text.text_annotations[1:]:
        t = r.bounding_poly.vertices[0].y
        l = r.bounding_poly.vertices[0].x
        h = r.bounding_poly.vertices[2].y - t
        w = r.bounding_poly.vertices[2].x - l
        t1 = t + h
        l1 = l + w
        ymin.append(t)
        xmin.append(l)
        ymax.append(t1)
        xmax.append(l1)
        height.append(h)
        width.append(w)
        text.append(r.description)

    description = response_text.text_annotations[0].description.split('\n')
    dic = {'text': text, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'width': width, 'height': height}
    df = pd.DataFrame(dic)

    return description, df