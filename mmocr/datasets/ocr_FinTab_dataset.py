from mmdet.datasets.builder import DATASETS
from mmocr.core.evaluation.ocr_metric import eval_ocr_metric
from mmocr.datasets.base_dataset import BaseDataset
from apted import APTED, Config
from apted.helpers import Tree
from collections import deque
import distance
from lxml import html
import numpy as np
import time


class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == 'td':
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % \
                     (self.tag, self.colspan, self.rowspan, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value
        """
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1
        """
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.
        if node1.tag == 'td':
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.


def tokenize(node):
    ''' Tokenizes table cells
    '''
    global __tokens__
    __tokens__.append('<%s>' % node.tag)
    if node.text is not None:
        __tokens__ += list(node.text)
    for n in node.getchildren():
        tokenize(n)
    if node.tag != 'unk':
        __tokens__.append('</%s>' % node.tag)
    if node.tag != 'td' and node.tail is not None:
            __tokens__ += list(node.tail)


def tree_convert_html(node, convert_cell=False, parent=None):
    ''' Converts HTML tree to the format required by apted
    '''
    global __tokens__
    if node.tag == 'td':
        if convert_cell:
            __tokens__ = []
            tokenize(node)
            cell = __tokens__[1:-1].copy()
        else:
            cell = []
        new_node = TableTree(node.tag,
                             int(node.attrib.get('colspan', '1')),
                             int(node.attrib.get('rowspan', '1')),
                             cell, *deque())
    else:
        new_node = TableTree(node.tag, None, None, None, *deque())
    if parent is not None:
        parent.children.append(new_node)
    if node.tag != 'td':
        for n in node.getchildren():
            tree_convert_html(n, convert_cell, new_node)
    if parent is None:
        return new_node


def similarity_eval_html(pred, true, structure_only=False):
    ''' Computes TEDS score between the prediction and the ground truth of a
        given samples
    '''
    if pred.xpath('body/table') and true.xpath('body/table'):
        pred = pred.xpath('body/table')[0]
        true = true.xpath('body/table')[0]
        n_nodes_pred = len(pred.xpath(".//*"))
        n_nodes_true = len(true.xpath(".//*"))
        tree_pred = tree_convert_html(pred, convert_cell=not structure_only)
        tree_true = tree_convert_html(true, convert_cell=not structure_only)
        n_nodes = max(n_nodes_pred, n_nodes_true)
        distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
        return 1.0 - (float(distance) / n_nodes)
    else:
        return 0.0


def merge_span_token(master_token_list):
    """
    Merge the span style token (row span or col span).
    :param master_token_list:
    :return:
    """
    new_master_token_list = []
    pointer = 0
    if master_token_list[-1] != '</tbody>':
        master_token_list.append('</tbody>')
    while pointer < len(master_token_list) and master_token_list[pointer] != '</tbody>':
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
    new_master_token_list.append('</tbody>')
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

    HTML = ''.join(merged_result_list)

    HTML = '''<html>
    <head>
    <meta charset="UTF-8">
    <style>
    table, th, td {
      border: 1px solid black;
      font-size: 10px;
    }
    </style>
    </head>
    <body>
    <table frame="hsides" rules="groups" width="100%%">
    %s
    </table>
    </body>
    </html>''' % ''.join(HTML)
    return HTML


def text_to_list(master_token):
    # insert virtual master token
    master_token_list = master_token.split(',')

    if master_token_list[-1] == '<td></td>':
        master_token_list.append('</tr>')
        master_token_list.append('</tbody>')
    elif master_token_list[-1] != '</tbody>':
        master_token_list.append('</tbody>')

    return master_token_list


@DATASETS.register_module()
class OCRFinTabDataset(BaseDataset):
    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['img_info']['ann_file'] = self.ann_file
        results['text'] = results['img_info']['text']

    def evaluate(self, results, metric='acc', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str: float]
        """
        gt_texts = []
        pred_texts = []

        teds_total = []
        # starting time
        start = time.time()

        for i in range(len(self)):
            item_info = self.data_infos[i]
            text = item_info['text']
            gt_texts.append(text)
            pred_texts.append(results[i]['text'])

            gt_cells = ["".join(item) for item in item_info['cell_content']]
            pred_text = results[i]['text']
            pred_cells = results[i]['cell']

            gt_html = insert_text_to_token(text_to_list(text), gt_cells)
            pred_html = insert_text_to_token(text_to_list(pred_text), pred_cells)

            teds_total.append(similarity_eval_html(html.fromstring(pred_html), html.fromstring(gt_html),
                                                   structure_only=False))

        eval_results = eval_ocr_metric(pred_texts, gt_texts)
        eval_results['TEDS'] = np.mean(teds_total)

        # end time
        end = time.time()
        # total time taken
        # print_nam
        print("Time: " + str(end - start))
        eval_results['Time eval'] = end - start

        return eval_results
