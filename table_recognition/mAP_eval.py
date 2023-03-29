"""
Evaluate the cell bounding box decoder by mAP
author: namly
"""
import numpy as np
from mean_average_precision import MetricBuilder
import json
import json_lines
from tqdm import tqdm
import sys
import pickle
import os
import glob


def pickle_load(path, prefix='end2end'):
    if os.path.isfile(path):
        data = pickle.load(open(path, 'rb'))
    elif os.path.isdir(path):
        data = dict()
        search_path = os.path.join(path, '{}_*.pkl'.format(prefix))
        pkls = glob.glob(search_path)
        for pkl in pkls:
            this_data = pickle.load(open(pkl, 'rb'))
            data.update(this_data)
    else:
        raise ValueError
    return data


if __name__ == "__main__":

    # create metric_fn
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)

    # calculate array ground truth
    jsonFile = '/disks/vaskar/exodus/icdar-2021-competition/pubtabnet/PubTabNet_2.0.0.jsonl'
    gt_dict = dict()
    with open(jsonFile, 'rb') as f:
        for item in tqdm(json_lines.reader(f)):
            """
                item's keys : ['filename', 'split', 'imgid', 'html']
                        item['html']'s keys : ['cells', 'structure']
                        item['html']['cell'] : list of dict
                            eg. [
                                {"tokens": ["<b>", "V", "a", "r", "i", "a", "b", "l", "e", "</b>"], "bbox": [1, 4, 27, 13]},
                                {"tokens": ["<b>", "H", "a", "z", "a", "r", "d", " ", "r", "a", "t", "i", "o", "</b>"], "bbox": [219, 4, 260, 13]},
                            ]
                        item['html']['structure']'s ['tokens']
                            eg. "structure": {"tokens": ["<thead>", "<tr>", "<td>", "</td>", ... ,"</tbody>"}
            """
            if item['split'] != 'val':
                continue

            filename = item['filename']
            gt_sample = []

            # text
            cells = item['html']['cells']
            textList = []
            for cell in cells:
                if 'bbox' not in cell.keys() or len(cell['tokens']) == 0:
                    # empty bbox
                    continue
                else:
                    cell_box = []
                    cell_box.extend(cell['bbox'])
                    cell_box.append(0)
                    cell_box.append(0)
                    cell_box.append(0)
                    gt_sample.append(cell_box)

            gtSample = {
                'type': 'complex' if '>' in item['html']['structure']['tokens'] else 'simple',
                'bbox': np.array(gt_sample),
                'filename': filename
            }

            gt_dict[filename] = gtSample

    # calculate array predict
    epoch_id = int(sys.argv[1])

    predFile = '/home2/nam/nam_data/work_dir/1114_TableMASTER_structure_seq500_cell150_batch4/step_12_17/structure_val_result_epoch_' + str(epoch_id)

    predDict = pickle_load(predFile, prefix='structure')

    for idx, (file_name, context) in enumerate(predDict.items()):
        # loading html of prediction
        bboxes = context['bbox']

        # x,y,w,h to x,y,x,y
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[..., 0] = bboxes[..., 0] - bboxes[..., 2] / 2
        new_bboxes[..., 1] = bboxes[..., 1] - bboxes[..., 3] / 2
        new_bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2] / 2
        new_bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3] / 2

        bboxes_ = []
        for bbox in new_bboxes:
            if bbox[0] == 0 and bbox[1] == 0 and bbox[0] == 0 and bbox[0] == 0:
                continue

            bboxes_.append(np.append(bbox, [0, 1]))

        bboxes_ = np.array(bboxes_)

        # check file_name in gt
        if file_name not in gt_dict:
            continue
        gt_context = gt_dict[file_name]

        metric_fn.add(bboxes_, gt_context['bbox'])

    print("Calculate VOC PASCAL mAP")

    # compute PASCAL VOC metric
    print(f"VOC PASCAL mAP: {metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}")

    # compute PASCAL VOC metric at the all points
    print(f"VOC PASCAL mAP in all points: {metric_fn.value(iou_thresholds=0.5)['mAP']}")



