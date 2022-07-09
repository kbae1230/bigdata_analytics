# 일회용성
# class 구현 불필요
import os
from pathlib import Path
from collections import defaultdict
import subprocess
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

EXT = ['.TIF', '.JP2','.LABEL']


def make_src_dict(dirname : str) -> dict:
    src_dict = defaultdict(list)
    src_list = find_label_scene(dirname)
    
    for src in src_list:
        airport = src.split(dirname)[-1].split('/')[1]
        file_name = os.path.basename(src).split('.')[0]
        src_dict[airport] += [file_name]
    return src_dict


def find_label_scene(src_path : str) -> list:
    filepath_list = []
    for root, _, files in os.walk(src_path):
        for file in files:
            _, ext = get_filename_ext(file)
            if ext in EXT:
                filepath_list.append(os.path.join(root, file))
    return filepath_list


def get_filename_ext(file : str) -> str:
    filename = os.path.splitext(file)[0].upper()
    ext = os.path.splitext(file)[-1].upper()
    return filename, ext


def check_airport(src_dict : str, filename : str) -> str:
    for k, v in src_dict.items():
        if filename in v:
            return k


def make_dir(dirpath : str) -> None:
    if not Path(dirpath).exists() :
        os.mkdir(dirpath)
        logging.info(f'{dirpath} created')


def move_file(filepath : str, dst_path : str) -> int:
    mv_level = subprocess.call(f'mv {filepath} {dst_path}' , shell=True)
    return mv_level


def define_destination(src, folder):
    # total 로 위치 변경
    total_path = src.replace('scenes', 'total')
    dst_path = os.path.join(total_path, folder)
    return dst_path


def main():
    dirname = "/nas/Dataset/RSI_CL_KAP22/2022-07-05/220705_해외항공기지_12AOI"
    src_path = '/nas/Dataset/RSI_CL_KAP22/aircraft/inf/2022-07-05/scenes'
    src_dict = make_src_dict(dirname)
    filepath_list = find_label_scene(src_path)

    cnt = 0
    for filepath in filepath_list:
        cnt += 1
        file_name = os.path.basename(filepath)
        filename, _ = get_filename_ext(file_name)
        airport = check_airport(src_dict, filename)
        dst_path = define_destination(src_path, airport)
        
        make_dir(dst_path)
        mv_level = move_file(filepath, dst_path)
        if mv_level != 0:
            raise('Failed to move')
    print('옮겨진 파일 갯 수 :', cnt)


if __name__ == '__main__':
    main()