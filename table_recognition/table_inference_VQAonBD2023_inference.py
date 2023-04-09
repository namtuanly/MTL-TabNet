
import torch
from mmcv.image import imread, imwrite, gray2bgr

from mmdet.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.datasets import build_dataset  # noqa: F401
from mmocr.models import build_detector  # noqa: F401

from tqdm import tqdm
import glob
import re
import copy


import cv2
import numpy as np
from PIL import Image
import os
import sys

IMAGE_EXT = "jpg"


def find_edge(img_path: str):
    img = cv2.imread(img_path, 0)
    blur = cv2.blur(img, (5, 5))
    edges = cv2.Canny(blur, 100, 200)
    return edges


def find_target(edges):
    results = np.where(edges == 255)
    top = np.min(results[0])
    bottom = np.max(results[0]) - 1
    left = np.min(results[1])
    right = np.max(results[1]) - 1
    return (left, top, right, bottom)


def to_RGB(image: Image):
    if image.mode == 'RGB': return image
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    background.format = image.format
    return background


def get_crop_img(img_path: str):
    edges = find_edge(img_path)
    left, top, right, bottom = find_target(edges)
    rgb_img = to_RGB(Image.open(img_path))
    trim_img = rgb_img.crop((left - 1, top - 1, right + 1, bottom + 1))

    return trim_img


def deal_isolate_span(thead_part):
    """
    Deal with isolate span cases in this function.
    It causes by wrong prediction in structure recognition model.
    eg. predict <td rowspan="2"></td> to <td></td> rowspan="2"></b></td>.
    :param thead_part:
    :return:
    """
    # 1. find out isolate span tokens.
    isolate_pattern = "<td></td> rowspan=\"(\d)+\" colspan=\"(\d)+\"></b></td>|" \
                      "<td></td> colspan=\"(\d)+\" rowspan=\"(\d)+\"></b></td>|" \
                      "<td></td> rowspan=\"(\d)+\"></b></td>|" \
                      "<td></td> colspan=\"(\d)+\"></b></td>"
    isolate_iter = re.finditer(isolate_pattern, thead_part)
    isolate_list = [i.group() for i in isolate_iter]

    # 2. find out span number, by step 1 results.
    span_pattern = " rowspan=\"(\d)+\" colspan=\"(\d)+\"|" \
                   " colspan=\"(\d)+\" rowspan=\"(\d)+\"|" \
                   " rowspan=\"(\d)+\"|" \
                   " colspan=\"(\d)+\""
    corrected_list = []
    for isolate_item in isolate_list:
        span_part = re.search(span_pattern, isolate_item)
        spanStr_in_isolateItem = span_part.group()
        # 3. merge the span number into the span token format string.
        if spanStr_in_isolateItem is not None:
            corrected_item = '<td{}></td>'.format(spanStr_in_isolateItem)
            corrected_list.append(corrected_item)
        else:
            corrected_list.append(None)

    # 4. replace original isolated token.
    for corrected_item, isolate_item in zip(corrected_list,isolate_list):
        if corrected_item is not None:
            thead_part = thead_part.replace(isolate_item, corrected_item)
        else:
            pass
    return thead_part


def deal_duplicate_bb(thead_part):
    """
    Deal duplicate <b> or </b> after replace.
    Keep one <b></b> in a <td></td> token.
    :param thead_part:
    :return:
    """
    # 1. find out <td></td> in <thead></thead>.
    td_pattern = "<td rowspan=\"(\d)+\" colspan=\"(\d)+\">(.+?)</td>|" \
                 "<td colspan=\"(\d)+\" rowspan=\"(\d)+\">(.+?)</td>|" \
                 "<td rowspan=\"(\d)+\">(.+?)</td>|" \
                 "<td colspan=\"(\d)+\">(.+?)</td>|" \
                 "<td>(.*?)</td>"
    td_iter = re.finditer(td_pattern, thead_part)
    td_list = [t.group() for t in td_iter]

    # 2. is multiply <b></b> in <td></td> or not?
    new_td_list = []
    for td_item in td_list:
        if td_item.count('<b>') > 1 or td_item.count('</b>') > 1:
            # multiply <b></b> in <td></td> case.
            # 1. remove all <b></b>
            td_item = td_item.replace('<b>','').replace('</b>','')
            # 2. replace <tb> -> <tb><b>, </tb> -> </b></tb>.
            td_item = td_item.replace('<td>', '<td><b>').replace('</td>', '</b></td>')
            new_td_list.append(td_item)
        else:
            new_td_list.append(td_item)

    # 3. replace original thead part.
    for td_item, new_td_item in zip(td_list, new_td_list):
        thead_part = thead_part.replace(td_item, new_td_item)
    return thead_part


def deal_bb(result_token):
    """
    In our opinion, <b></b> always occurs in <thead></thead> text's context.
    This function will find out all tokens in <thead></thead> and insert <b></b> by manual.
    :param result_token:
    :return:
    """
    # find out <thead></thead> parts.
    thead_pattern = '<thead>(.*?)</thead>'
    if re.search(thead_pattern, result_token) is None:
        return result_token
    thead_part = re.search(thead_pattern, result_token).group()
    origin_thead_part = copy.deepcopy(thead_part)

    # check "rowspan" or "colspan" occur in <thead></thead> parts or not .
    span_pattern = "<td rowspan=\"(\d)+\" colspan=\"(\d)+\">|<td colspan=\"(\d)+\" rowspan=\"(\d)+\">|<td rowspan=\"(\d)+\">|<td colspan=\"(\d)+\">"
    span_iter = re.finditer(span_pattern, thead_part)
    span_list = [s.group() for s in span_iter]
    has_span_in_head = True if len(span_list) > 0 else False

    if not has_span_in_head:
        # <thead></thead> not include "rowspan" or "colspan" branch 1.
        # 1. replace <td> to <td><b>, and </td> to </b></td>
        # 2. it is possible to predict text include <b> or </b> by Text-line recognition,
        #    so we replace <b><b> to <b>, and </b></b> to </b>
        thead_part = thead_part.replace('<td>', '<td><b>')\
            .replace('</td>', '</b></td>')\
            .replace('<b><b>', '<b>')\
            .replace('</b></b>', '</b>')
    else:
        # <thead></thead> include "rowspan" or "colspan" branch 2.
        # Firstly, we deal rowspan or colspan cases.
        # 1. replace > to ><b>
        # 2. replace </td> to </b></td>
        # 3. it is possible to predict text include <b> or </b> by Text-line recognition,
        #    so we replace <b><b> to <b>, and </b><b> to </b>

        # Secondly, deal ordinary cases like branch 1

        # replace ">" to "<b>"
        replaced_span_list = []
        for sp in span_list:
            replaced_span_list.append(sp.replace('>', '><b>'))
        for sp, rsp in zip(span_list, replaced_span_list):
            thead_part = thead_part.replace(sp, rsp)

        # replace "</td>" to "</b></td>"
        thead_part = thead_part.replace('</td>', '</b></td>')

        # remove duplicated <b> by re.sub
        mb_pattern = "(<b>)+"
        single_b_string = "<b>"
        thead_part = re.sub(mb_pattern, single_b_string, thead_part)

        mgb_pattern = "(</b>)+"
        single_gb_string = "</b>"
        thead_part = re.sub(mgb_pattern, single_gb_string, thead_part)

        # ordinary cases like branch 1
        thead_part = thead_part.replace('<td>', '<td><b>').replace('<b><b>', '<b>')


    # convert <tb><b></b></tb> back to <tb></tb>, empty cell has no <b></b>.
    # but space cell(<tb> </tb>)  is suitable for <td><b> </b></td>
    thead_part = thead_part.replace('<td><b></b></td>', '<td></td>')
    # deal with duplicated <b></b>
    thead_part = deal_duplicate_bb(thead_part)
    # deal with isolate span tokens, which causes by wrong predict by structure prediction.
    # eg.PMC5994107_011_00.png
    thead_part = deal_isolate_span(thead_part)
    # replace original result with new thead part.
    result_token = result_token.replace(origin_thead_part, thead_part)
    return result_token


def merge_span_token(master_token_list):
    """
    Merge the span style token (row span or col span).
    :param master_token_list:
    :return:
    """
    new_master_token_list = []
    pointer = 0
    if master_token_list[-1] != '</table>':
        master_token_list.append('</table>')
    while pointer < len(master_token_list) and master_token_list[pointer] != '</table>':
        try:
            if master_token_list[pointer] == '<td':
                if (pointer + 5) <= len(master_token_list) and (master_token_list[pointer+2].startswith(' colspan=') or
                                                                master_token_list[pointer+2].startswith(' rowspan=')):
                    """
                    example:
                    pattern <td rowspan="2" colspan="3">
                    '<td' + 'rowspan=" "' + 'colspan=" "' + '>' + '</td>'
                    """
                    # tmp = master_token_list[pointer] + master_token_list[pointer+1] + \
                    #       master_token_list[pointer+2] + master_token_list[pointer+3] + master_token_list[pointer+4]
                    tmp = ''.join(master_token_list[pointer:pointer+4+1])
                    pointer += 5
                    new_master_token_list.append(tmp)

                elif (pointer + 4) <= len(master_token_list) and \
                        (master_token_list[pointer+1].startswith(' colspan=') or
                         master_token_list[pointer+1].startswith(' rowspan=')):
                    """
                    example:
                    pattern <td colspan="3">
                    '<td' + 'colspan=" "' + '>' + '</td>'
                    """
                    # tmp = master_token_list[pointer] + master_token_list[pointer+1] + master_token_list[pointer+2] + \
                    #       master_token_list[pointer+3]
                    tmp = ''.join(master_token_list[pointer:pointer+3+1])
                    pointer += 4
                    new_master_token_list.append(tmp)

                else:
                    new_master_token_list.append(master_token_list[pointer])
                    pointer += 1
            else:
                new_master_token_list.append(master_token_list[pointer])
                pointer += 1
        except:
            print("Break in merge...")
            break
    new_master_token_list.append('</table>')

    return new_master_token_list


def deal_eb_token(master_token):
    """
    post process with <eb></eb>, <eb1></eb1>, ...
    emptyBboxTokenDict = {
        "[]": '<eb></eb>',
        ...
    }
    :param master_token:
    :return:
    """
    master_token = master_token.replace('<eb></eb>', '<td></td>')
    return master_token


def insert_text_to_token(master_token_list, cell_content_list):
    """
    Insert OCR text result to structure token.
    :param master_token_list:
    :param cell_content_list:
    :return:
    """
    master_token_list = merge_span_token(master_token_list)
    merged_result_list = []
    text_count = 0
    for master_token in master_token_list:
        if master_token.startswith('<td'):
            if text_count > len(cell_content_list)-1:
                text_count += 1
                continue
            else:
                master_token = master_token.replace('><', '>{}<'.format(cell_content_list[text_count]))
                text_count += 1
        master_token = deal_eb_token(master_token)
        merged_result_list.append(master_token)

    return ''.join(merged_result_list)


def text_to_list(master_token):
    # insert virtual master token
    master_token_list = master_token.split(',')

    if master_token_list[-1] == '<td></td>':
        master_token_list.append('</tr>')
        master_token_list.append('</table>')
    elif master_token_list[-1] != '</table>':
        master_token_list.append('</table>')

    if master_token_list[-2] != '</tr>':
        master_token_list.insert(-1, '</tr>')

    return master_token_list


def pp_table_recognition(result_dict, path_html):
    file_name, context = result_dict
    pred_text = context['text']
    pred_cells = context['cell']
    pred_html = insert_text_to_token(text_to_list(pred_text), pred_cells)
    with open(os.path.join(path_html, file_name.replace(IMAGE_EXT, 'txt')), 'w') as f_html:
        f_html.write(pred_html + '\n')


def build_model(config_file, checkpoint_file):
    device = 'cpu'
    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)

    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][
            0].pipeline

    return model


class Inference:
    def __init__(self, config_file, checkpoint_file, device=None):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.model = build_model(config_file, checkpoint_file)

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            # Specify GPU device
            device = torch.device("cuda:{}".format(device))

        self.model.to(device)

    def result_format(self, pred, file_path):
        raise NotImplementedError

    def predict_single_file(self, file_path):
        pass

    def predict_batch(self, imgs):
        pass


class Structure_Recognition(Inference):
    def __init__(self, config_file, checkpoint_file, samples_per_gpu=4):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        super().__init__(config_file, checkpoint_file)
        self.samples_per_gpu = samples_per_gpu

    def result_format(self, pred, file_path=None):
        pred = pred[0]
        return pred

    def predict_single_file(self, file_path):

        new_path = file_path[:-3] + 'png'
        try:
            dst = get_crop_img(file_path)
            dst.save(new_path)

            # numpy inference
            img = cv2.imread(new_path)

            h, w, _ = img.shape
            w = int(w / 4)
            h = int(h / 4)

            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(new_path, img)
        except:
            print(file_path)
            return

        # numpy inference
        img = imread(new_path)
        file_name = os.path.basename(file_path)
        result = model_inference(self.model, [img], batch_mode=True)
        result = self.result_format(result, file_path)
        result_dict = (file_name, result)
        return result, result_dict


class Runner:
    def __init__(self, cfg):
        self.structure_master_config = cfg['structure_master_config']
        self.structure_master_ckpt = cfg['structure_master_ckpt']
        self.structure_master_result_folder = cfg['structure_master_result_folder']

        test_folder = cfg['test_folder']
        chunks_nums = cfg['chunks_nums']
        self.chunks_nums = chunks_nums
        self.chunks = self.get_file_chunks(test_folder, chunks_nums=chunks_nums)

    def init_structure_master(self):
        self.master_structure_inference = \
            Structure_Recognition(self.structure_master_config, self.structure_master_ckpt)

    def release_structure_master(self):
        torch.cuda.empty_cache()
        del self.master_structure_inference


    def do_structure_predict(self, path, is_save=True, gpu_idx=None):
        if isinstance(path, str):
            if os.path.isfile(path):
                print('Single file in structure master prediction ...')
                _, result_dict = self.master_structure_inference.predict_single_file(path)
                pp_table_recognition(result_dict, self.structure_master_result_folder)

            elif os.path.isdir(path):
                print('Folder files in structure master prediction ...')
                search_path = os.path.join(path, '*.' + IMAGE_EXT)
                files = glob.glob(search_path)
                for file in tqdm(files):
                    _, result_dict = self.master_structure_inference.predict_single_file(file)
                    pp_table_recognition(result_dict, self.structure_master_result_folder)

            else:
                raise ValueError

        elif isinstance(path, list):
            print('Chunks files in structure master prediction ...')
            for i, p in enumerate(path):
                try:
                    _, result_dict = self.master_structure_inference.predict_single_file(p)
                    pp_table_recognition(result_dict, self.structure_master_result_folder)
                    if gpu_idx is not None:
                        print("[GPU_{} : {} / {}] {} file structure inference. ".format(gpu_idx, i+1, len(path), p))
                    else:
                        print("{} file structure inference. ".format(p))
                except:
                    continue

        else:
            raise ValueError


    def get_file_chunks(self, folder, chunks_nums=8):
        """
        Divide files in folder to different chunks, before inference in multiply gpu devices.
        :param folder:
        :return:
        """
        print("Divide files to chunks for multiply gpu device inference.")
        file_paths = glob.glob(folder + '*.' + IMAGE_EXT)
        counts = len(file_paths)
        nums_per_chunk = counts // chunks_nums
        img_chunks = []
        for n in range(chunks_nums):
            if n == chunks_nums - 1:
                s = n * nums_per_chunk
                img_chunks.append(file_paths[s:])
            else:
                s = n * nums_per_chunk
                e = (n + 1) * nums_per_chunk
                img_chunks.append(file_paths[s:e])
        return img_chunks


    def run_structure_single_chunk(self, chunk_id):
        # list of path
        paths = self.chunks[chunk_id]

        # structure master
        self.init_structure_master()
        self.do_structure_predict(paths, is_save=True, gpu_idx=chunk_id)
        self.release_structure_master()




if __name__ == '__main__':
    # Runner
    chunk_nums = int(sys.argv[1])
    chunk_id = int(sys.argv[2])

    cfg = {
        # config file
        'structure_master_config': './configs/textrecog/master/table_master_local_attn_new_decoder_FinTabNet_full_img520_win300_0_tag600_cell150_batch4.py',        # structure

        # check point file
        'structure_master_ckpt': '/home2/nam/nam_data/work_dir/1114_TableMASTER_local_attn_new_decoder_FinTabNet_full_img520_win300_0_tag600_cell150_batch4/epoch_20.pth',              # structure

        # folder that store predicted table HTML
        'structure_master_result_folder': '/disks/strg16-176/nam/VQAonBD2023/test/test_html_infer/',

        # input image folder
        'test_folder': '/disks/strg16-176/nam/VQAonBD2023/test/VQAonBD_testset/test_images/',

        'chunks_nums': chunk_nums
    }

    print(cfg)
    if not os.path.exists(cfg['structure_master_result_folder']):
        os.makedirs(cfg['structure_master_result_folder'], exist_ok=True)

    runner = Runner(cfg)
    # structure task
    runner.run_structure_single_chunk(chunk_id=chunk_id)
