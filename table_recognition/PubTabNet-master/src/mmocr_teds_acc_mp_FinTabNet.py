import os
import json
import time
import pickle
from metric import TEDS
from multiprocessing import Pool
import glob
import re
import copy
import shutil
import sys


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


def htmlPostProcess(text):
    text = '<html><body>' + text + '</body></html>'
    return text


def singleEvaluation(teds, file_name, context, gt_context):
    # save problem log
    # save_folder = ''

    # html format process
    htmlContext = htmlPostProcess(context)
    htmlGtContext = htmlPostProcess(gt_context)
    # Evaluate
    score = teds.evaluate(htmlContext, htmlGtContext)

    print("FILENAME : {}".format(file_name))
    print("SCORE    : {}".format(score))
    return score


def visual_pred_bboxes(bboxes, filename, dir_path, dir_img):
    """
    visual after normalized bbox in results.
    :param results:
    :return:
    """
    import cv2
    import numpy as np

    os.makedirs(dir_path, exist_ok=True)

    img = cv2.imread(dir_img + filename)
    save_path = dir_path + '{}_pred_bbox.png'. \
        format(filename.split('.')[0])

    # x,y,w,h to x,y,x,y
    new_bboxes = np.empty_like(bboxes)
    new_bboxes[..., 0] = bboxes[..., 0] - bboxes[..., 2] / 2
    new_bboxes[..., 1] = bboxes[..., 1] - bboxes[..., 3] / 2
    new_bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2] / 2
    new_bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3] / 2

    table_height = img.shape[0]

    # draw
    for new_bbox in new_bboxes:
        orig_annotation = copy.copy(new_bbox)
        new_bbox[3] = table_height-orig_annotation[1]
        new_bbox[1] = table_height-orig_annotation[3]
        img = cv2.rectangle(img, (int(new_bbox[0]), int(new_bbox[1])),
                            (int(new_bbox[2]), int(new_bbox[3])), (0, 255, 0), thickness=1)
    cv2.imwrite(save_path, img)


if __name__ == "__main__":

    epoch_id = int(sys.argv[1])

    val_test = 'val'

    t_start = time.time()
    pool = Pool(64)
    start_time = time.time()
    predFile = '/home2/nam/nam_data/work_dir/1114_TableMASTER_FinTabNet_seq500_cell150_batch4/_structure_val_result_epoch_' + str(epoch_id)

    gtJsonFile = '/disks/strg16-176/nam/data/fintabnet/img_tables/gtVal_FinTabNet_val.json'

    fintabnet_dir = '/disks/strg16-176/nam/data/fintabnet/img_tables/' + val_test + '/'

    # Initialize TEDS object
    teds = TEDS(structure_only=False, n_jobs=1) #, ignore_nodes='b')

    predDict = pickle_load(predFile, prefix='structure')

    with open(gtJsonFile, 'r') as f:
        gtValDict = json.load(f)

    scores_simple = []
    scores_complex = []
    caches = dict()
    for idx, (file_name, context) in enumerate(predDict.items()):
        # loading html of prediction
        pred_text = context['text']
        pred_cells = context['cell']

        pred_html = insert_text_to_token(text_to_list(pred_text), pred_cells)
        # pred_html = deal_bb(pred_html)

        # file_name = os.path.basename(file_path)
        # check file_name in gt
        if file_name not in gtValDict:
            continue
        gt_context = gtValDict[file_name]
        # print(file_name)
        score = pool.apply_async(func=singleEvaluation, args=(teds, file_name, pred_html, gt_context['html'],))
        if gt_context['type'] == 'simple':
            scores_simple.append(score)
        else:
            scores_complex.append(score)
        tmp = {'score':score, 'gt':gt_context['html'], 'pred':pred_html}
        caches.setdefault(file_name, tmp)

        # # visualize bboxes
        # visual_pred_bboxes(context['bbox'], file_name, predFile + '/visual_pred_bboxes/', fintabnet_dir)

    pool.close()
    pool.join() # 进程池中进程执行完毕后再关闭，如果注释，那么程序直接关闭。
    pool.terminate()

    # get score from scores
    cal_scores = []
    cal_scores_simple = []
    for score in scores_simple:
        cal_scores.append(score.get())
        cal_scores_simple.append(score.get())

    cal_scores_complex = []
    for score in scores_complex:
        cal_scores.append(score.get())
        cal_scores_complex.append(score.get())

    print('AVG TEDS score: {}'.format(sum(cal_scores)/len(cal_scores)))
    print('AVG TEDS Simple score: {}'.format(sum(cal_scores_simple)/len(cal_scores_simple)))
    print('AVG TEDS Complex score: {}'.format(sum(cal_scores_complex)/len(cal_scores_complex)))
    print('TEDS cost time: {}s'.format(time.time()-start_time))
    print('Number sample: {}'.format(len(cal_scores)))

    with open(predFile + '/cfg.txt', 'a') as f:
        f.write('AVG TEDS score: {}'.format(sum(cal_scores)/len(cal_scores)) + '\n')
        f.write('AVG TEDS Simple score: {}'.format(sum(cal_scores_simple)/len(cal_scores_simple)) + '\n')
        f.write('AVG TEDS Complex score: {}'.format(sum(cal_scores_complex)/len(cal_scores_complex)) + '\n')
        f.write('TEDS cost time: {}s'.format(time.time()-start_time) + '\n')
        f.write('Number sample: {}'.format(len(cal_scores)) + '\n')






