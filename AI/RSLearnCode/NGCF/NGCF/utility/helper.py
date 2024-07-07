'''
Created on Aug 19, 2016
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
__author__ = "xiangwang"
import os
import re


def txt2list(file_src):
    """读取指定文件中的所有行，并将每一行作为字符串存储在一个列表中，然后返回这个列表"""
    orig_file = open(file_src, "r")
    lines = orig_file.readlines()
    return lines


def ensureDir(dir_path):
    """确保指定目录路径存在。如果目录不存在，则创建这个目录及其所有必要的父目录"""
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


def uni2str(unicode_str):
    """将给定的Unicode字符串转换为ASCII编码的字符串，并且忽略非ASCII字符。然后移除字符串两端的空格和换行符。"""
    return str(unicode_str.encode('ascii', 'ignore')).replace('\n', '').strip()


def hasNumbers(inputString):
    """检查输入字符串中是否包含数字。如果找到数字，则返回True"""
    return bool(re.search(r'\d', inputString))


def delMultiChar(inputString, chars):
    """删除输入字符串中的多个指定字符"""
    for ch in chars:
        inputString = inputString.replace(ch, '')
    return inputString


def merge_two_dicts(x, y):
    """将两个字典x和y合并成一个新的字典z。如果y中的键在x中已存在，它们的值将被y中的值所覆盖。"""
    z = x.copy()
    z.update(y)
    return z


def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    """
    功能：早停策略
    参数：
        log_value：当前的评估指标值，用来与best_value比较
        best_value：最佳评估指标值，会根据log_value的表现来更新
        stopping_steop：当前的停止步数，用于计算何时触发早停
        expected_order：期望的指标顺序，acc升序，dec降序
        flag_step：达到这个步数后触发早停
    """
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']
    # 判断当前的log_value是否优于best_value
    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:  # 若未优于最佳记录，则增加stopping_step
        stopping_step += 1
    # 若stopping_step大于预设的flag_step，则打印早停信息，并返回should_stop=True
    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop
