"""
This script for extracting table images from PDF and make the annotations like PubTabNet (FinTabNet)
namly
"""
# importing prerequisites
import json
import numpy as np
from PIL import Image
from PIL import ImageFont, ImageDraw
from pdf2image import convert_from_path
from PyPDF2 import PdfFileReader
import copy
import shutil
import os

# Define color code
colors = [(255, 0, 0),(0, 255, 0)]
categories = ["table", "cell"]

# Function to viz the annotation
def markup(image, annotations, pdf_height):
    ''' Draws the segmentation, bounding box, and label of each annotation
    '''
    draw = ImageDraw.Draw(image, 'RGBA')
    for annotation in annotations:
        # Draw bbox
        orig_annotation = copy.copy(annotation['bbox'])
        annotation['bbox'][3] = pdf_height-orig_annotation[1]
        annotation['bbox'][1] = pdf_height-orig_annotation[3]
        draw.rectangle(
            (annotation['bbox'][0],
             annotation['bbox'][1],
             annotation['bbox'][2],
             annotation['bbox'][3]),
            outline=colors[annotation['category_id'] - 1] + (255,),
            width=2
        )
        # Draw label
        # w, h = draw.textsize(text=categories[annotation['category_id'] - 1])
        # if annotation['bbox'][3] < h:
        #     draw.rectangle(
        #         (annotation['bbox'][2],
        #          annotation['bbox'][1],
        #          annotation['bbox'][2] + w,
        #          annotation['bbox'][1] + h),
        #         fill=(64, 64, 64, 255)
        #     )
        #     draw.text(
        #         (annotation['bbox'][2],
        #          annotation['bbox'][1]),
        #         text=categories[annotation['category_id'] - 1],
        #         fill=(255, 255, 255, 255)
        #     )
        # else:
        #     draw.rectangle(
        #         (annotation['bbox'][0]-w,
        #          annotation['bbox'][1]-h,
        #          annotation['bbox'][0],
        #          annotation['bbox'][1]),
        #         fill=(64, 64, 64, 255)
        #     )
        #     draw.text(
        #         (annotation['bbox'][0]-w,
        #          annotation['bbox'][1]-h),
        #         text=categories[annotation['category_id'] - 1],
        #         fill=(255, 255, 255, 255)
        #     )
    return np.array(image)

split_ = 'test'

base_path = 'fintabnet/img_tables/'
save_img_path = base_path + split_ + '/'
os.makedirs(save_img_path, exist_ok=True)

json_line = open(base_path + 'FinTabNet_1.0.0_table_' + split_ + '.jsonl', 'w')

table_idx_ = 1

# Parse the JSON file and read all the images and labels
with open('fintabnet/FinTabNet_1.0.0_cell_' + split_ + '.jsonl', 'r') as fp:
    images = {}
    for line in fp:
        sample = json.loads(line)
        # print(sample)
        # Index images
        pdf_path = 'fintabnet/pdf/' + sample['filename']

        pdf_page = PdfFileReader(open(pdf_path, 'rb')).getPage(0)
        pdf_shape = pdf_page.mediaBox
        pdf_height = pdf_shape[3]-pdf_shape[1]
        pdf_width = pdf_shape[2]-pdf_shape[0]
        converted_images = convert_from_path(pdf_path, size=(pdf_width, pdf_height))
        img_pdf = converted_images[0]

        # Draw bbox
        orig_annotation = copy.copy(sample["bbox"])
        sample['bbox'][3] = float(pdf_height) - float(orig_annotation[1])
        sample['bbox'][1] = float(pdf_height) - float(orig_annotation[3])

        img_table = img_pdf.crop((sample['bbox'][0],
                                  sample['bbox'][1],
                                  sample['bbox'][2],
                                  sample['bbox'][3]))

        annotations = []
        negative_ = False
        for t, token in enumerate(sample["html"]["cells"]):
            if "bbox" in token:
                # print(token["bbox"])

                token["bbox"][0] = np.around(np.around(token["bbox"][0], 1) - np.around(orig_annotation[0], 1))
                token["bbox"][1] = np.around(np.around(token["bbox"][1], 1) - np.around(orig_annotation[1], 1))
                token["bbox"][2] = np.around(np.around(token["bbox"][2], 1) - np.around(orig_annotation[0], 1))
                token["bbox"][3] = np.around(np.around(token["bbox"][3], 1) - np.around(orig_annotation[1], 1))

                # print(orig_annotation)
                if token["bbox"][0] < 0 or token["bbox"][1] < 0 or token["bbox"][2] < 0 or token["bbox"][3] < 0:
                    print(token["bbox"])
                    negative_ = True

                annotations.append({"category_id":2, "bbox": token["bbox"]})

        img_file_name = str(sample['table_id']) + '_' + str(table_idx_) + '.png'

        table_sample = {'filename': img_file_name,
                        'split': sample['split'],
                        'imgid': table_idx_,
                        'html': sample["html"]}

        if not negative_:
            json.dump(table_sample, json_line)
            json_line.write('\n')
            img_table.save(save_img_path + img_file_name)
            print(table_idx_)
        # else:
        #     with open(base_path + 'errors_test/' + str(sample['table_id']) + '_' + str(table_idx_) + '.txt', 'w') as f_e:
        #         f_e.write(json.dumps(table_sample))
        #         f_e.write(json.dumps(sample))
        #     img_table_vis = Image.fromarray(markup(img_table, annotations, sample['bbox'][3] - sample['bbox'][1]))
        #     img_table_vis.save(base_path + 'errors_test/' + img_file_name)
        #     shutil.copy(pdf_path, base_path + 'errors_test/')

        table_idx_ = table_idx_ + 1

json_line.close()
