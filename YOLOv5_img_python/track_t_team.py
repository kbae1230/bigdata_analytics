import pyspark
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import *
import sys
sys.path.insert(0, './yolov5')
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
# 다각형 내 객체 카운트용도
from shapely.geometry import Point, Polygon
# 영역 표시용 xml 파일을 읽어서 영역 저장
import xml.etree.ElementTree as et
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
# 영상에 이동하는 점 저장용 리스트 추가 / 박스 중앙점 그리는 데 사용


def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = xyxy[0].item() # 좌측 x값
    bbox_top = xyxy[1].item() # 상단 y값
    bbox_w = xyxy[2].item() - xyxy[0].item() # bounding box 너비
    bbox_h = xyxy[3].item() - xyxy[1].item() # bounding box 높이
    x_c = (bbox_left + bbox_w / 2) # 중앙점 x좌표
    y_c = (bbox_top + bbox_h / 2) # 중앙점 y좌표
    w = bbox_w # 너비
    h = bbox_h # 높이
    return x_c, y_c, w, h
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette] # 객체별 컬러 지정
    return tuple(color)
# (프레임(함수가 적용될 이미지), bounding box 좌표, 객체번호, 오프셋)
def draw_boxes(img, bbox, identities=None, offset=(0, 0)): 
    for i, box in enumerate(bbox): 
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0 # identities(객체번호) 부여
        color = compute_color_for_labels(id) # 컬러부여
        label = '{}{:d}'.format("", id) # id번호
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0] # label 텍스트 사이즈 지정
        
        if alarm == 'Loitering' or alarm == 'Intrusion' :
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 3)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3) # bounding box 그리기 (0,0,255)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1) # label텍스트 박스 그리기
        cv2.putText(img, label, (x1, y1 +
                                t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2) # label 텍스트 출력
    return img # draw_boxes함수가 적용된 프레임(이미지)
def detect(opt, save_img=False): # 객체 탐지하기

    memory = [(0,0)]

    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt') # 실행시 --로 지정하는 옵션 확인
    # initialize deepsort
    cfg = get_config() # /deep_sort_pytorch/utils/parser.py의 get_config함수 실행
    cfg.merge_from_file(opt.config_deepsort) # 실행시 옵션변경 가능 default="deep_sort_pytorch/configs/deep_sort.yaml -> ckpt.t7 로 지정되어있음
    
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True) #/deep_sort_pytorch/deep_sort/deep_sort.py의 class Deepsort(object) 실행 및 변수값 지정
    
    # Initialize
    device = select_device(opt.device) # /yolov5/utils/torch_utils.py 에서 select_device실행 device = 'cpu' or cuda('0' or '0,1,2,3')
    
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model / 우리경우, weights-> yolov5s.pt
    model = torch.load(weights, map_location=device)[
        'model'].float()  # load to FP32 
    model.to(device).eval()
    
    if half:
        model.half()  # to FP16
    # Set Dataloader
    vid_path, vid_writer = None, None
    
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        # view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names 
    
    # Run inference
    t0 = time.time() # 실행 시작 시간
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img 
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt' # 결과물 저장경로 설정 /yolov5/inference/output/
    
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset): # 데이터 셋에서 각 프레임 정보 받아오기
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized() # nms 전 시간
        pred = model(img, augment=opt.augment)[0] # --augment로 augmentation가능/ prediction할 모델 지정
        '''
        Apply NMS = Non Maxmimum Suppressions
        NMS는 여러개의 바운딩 박스가 겹쳐있을때,
        어떤 것을 선택하고 어떤 것을 버릴지 판단하는 알고리즘
        가장 높은 confidence score 를 가진 박스를 선택하고
        해당 박스와의 IOU가 threshold 이상이면 제외(suppresion) 해준다.
        Returns:
        detections with shape: nx6 (x1, y1, x2, y2, conf, cls)'''
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms) # nms 변수 설정(모델,conf-thres,iou-thres,class,agnostic)
        t2 = time_synchronized() # nms 후 시간
        global p
        # Process detections
        for i, det in enumerate(pred):  # detections per image /pred의 결과 값들을 det으로 뽑아냄
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s # p,s,im0 변수 지정
            # testmiddle=tuple(((det[0]+det[2])/2,det[3]))
            # cv2.circle(im0, testmiddle, 3, (0, 255, 0), -1)
            # 객체가 이동하는 아래중앙점 남기기
            for i in range(len(memory)):
                t_memory = tuple(memory[i])
                cv2.circle(im0, t_memory, 2, (255, 255, 0), -1)
            # 영역표시 (Loitering, Intrusion)
            pathh = filepath.split('_')[0] + '.map'
            global mapxy_1, mapxy_2, t_mapxy, alarm
            alarm = 'none'
            mapxy_1 = []
            mapxy_2 = []
            # 해당 폴더에 map 파일이 존재하면 아래를 진행
            if os.path.isfile(pathh) == True:
                tree = et.parse(path.split('_')[0] + '.map')
                root = tree.getroot()
                # '배회'면 아래를 진행
                if bool(root.find('Loitering')) == True:
                    Loitering = root.find('Loitering')
                    for point in Loitering.iter('Point'):
                        
                        points = point.text.split(',')
                        xy = (int(points[0]),int(points[1]))
                        mapxy_1.append(xy)
                    for i in range(1, len(mapxy_1)):
                        cv2.line(im0, mapxy_1[i-1], mapxy_1[i], (0, 255, 255), 5)
                        cv2.line(im0, mapxy_1[0], mapxy_1[len(mapxy_1)-1], (0, 255, 255), 5)
                # '침입'이면 아래를 진행
                if bool(root.find('Intrusion')) == True:
                    Intrusion = root.find('Intrusion')
                    for point in Intrusion.iter('Point'):
                
                        points = point.text.split(',')
                        xy = (int(points[0]),int(points[1]))
                        mapxy_2.append(xy)
                        
                    for i in range(1, len(mapxy_2)):
                        cv2.line(im0, mapxy_2[i-1], mapxy_2[i], (0, 255, 255), 5)
                        cv2.line(im0, mapxy_2[0], mapxy_2[len(mapxy_2)-1], (0, 255, 255), 5)
            # Frame, Time, Count 표시기능 추가
            frame = 30
            cv2.putText(im0, "Time: %f"% (int(frame_idx+1)*(1/frame)), (int(20), int(80)),0, 5e-3 * 200, (255,0,0),3) # frame_idx이용 각 프레임 시간 구하기
            cv2.putText(im0, "FrameNo: %i"%(int(frame_idx+1)),(int(20), int(120)),0, 5e-3 * 200, (255,0,0),3)
            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size / det[:, :4] => (x1, y1, x2, y2)
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class / c=class번호, n=class번호당 갯수
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string / 각 class의 이름과 n값 출력
                    #a = names[int(c)] #객체 class명 추출 위한 변수설정
                bbox_xywh = []
                confs = []
                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det: # det에서 (x1, y1, x2, y2)과 conf값, class 추출
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy) # 위에서 정의한 bbox_rel함수 이용 중앙값 좌표와 bounding box 너비와 높이 구하기
                    obj = [x_c, y_c, bbox_w, bbox_h] # bounding box 정보를 obj 라는 변수로 선언
                    bbox_xywh.append(obj)
                    confs.append([conf.item()]) # 위에서 선언한 bbox_xywh, confs리스트에 obj(bbox 정보),와 conf값 추가하기
                    # 객체의 중앙점, 각 꼭짓점에 점 찍는 것 추가
                    radius = 3
                    undermiddle=(int((xyxy[0]+xyxy[2])/2),int(xyxy[3]))
                    cv2.circle(im0, undermiddle, radius, (255, 255, 0), -1)
                    #print(undermiddle)
                    # memory 라는 리스트에 중앙좌표(x, y) 넣기
                    memory.append(undermiddle)
                    ''' 객체가 침입영역이나 배회영역안에 들어오면 polygon,within 메쏘드를 사용하여 판별 후 알람을 부여
                    원래 전의 코드에서는 영역지정할때 같이 넣었으나 그렇게 하면 undermiddle말고 memory[-1]을 사용해야함
                    그러면 다수의 객체에서 객체간의 구분이 안되고 객체가 없을때도 영역이나 배회라고 인식함
                    '''
                    #객체이름이 fire일때를 찾아내기 위한 변수설정
                    cls_name = names[int(cls)]
                    points = undermiddle
                    pointss = Point(points)
                    poly_lot = Polygon(mapxy_1)
                    loitering = pointss.within(poly_lot)
                    if loitering == True:
                        alarm = 'Loitering'
                    poly_int = Polygon(mapxy_2)
                    intrusion = pointss.within(poly_int)
                    if intrusion == True:
                        alarm = 'Intrusion'
                    if cls_name == 'fire':
                        alarm = 'FireDetection'
            
                if alarm != 'Loitering' and alarm != 'Intrusion' and alarm != 'FireDetection' :
                    alarm = 'none' 
                print(alarm)
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                outputs = deepsort.update(xywhs, confss, im0)
                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4] # output의 x1, y1, x2, y2
                    identities = outputs[:, -1] # output의 track_id(객체 번호)
                    draw_boxes(im0, bbox_xyxy, identities) # output의 결과로 박스그리기 위에서 선언한 draw_boxes 실행
                    
                # Write MOT compliant results to file
                if len(outputs) != 0: # 실행시 --save-txt 입력되면 실행
                    for j, output in enumerate(outputs): # output값을 이용해서 좌측 x값, 상단 y값, 너비, 높이, 객체번호 추출
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        # if intrusion == True  :
                        #     alarm = 'Intrusion'
                        # else :
                        #     alarm = 'none'
                        # /inference/output/results.txt 결과 삽입 내용 (총 10개 가능)
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '%s' + '\n') % (frame_idx, identity, bbox_left,
                                                        bbox_h, bbox_w, bbox_top, int((bbox_left+bbox_w)/2), 
                                                        int((bbox_top+bbox_h)/2), 
                                                        int(frame_idx+1)*(1/frame), 
                                                        -1, alarm
                                                        ))  # label format                        
            else:
                deepsort.increment_ages()
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1)) # 이미지 사이즈, 실행시간 출력
            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
            # Save results (image with detections)
            if save_img:
                # print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    # print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h)) # fourcc : default='mp4v' / 비디오 코덱
                            # save_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h)) # fourcc : default='mp4v' / 비디오 코덱
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)
    
    # 한장의 이미지를 분석하는데 걸린 시간
    print('Done. (%.3fs)' % (time.time() - t0))

# 딥러닝 시작
if __name__ == '__main__':
    # Kafka의 mysql_kk 토픽의 제일 마지막 행 시작하도록 변경
    while(True):


        # SparkSession 열기
        spark = SparkSession \
            .builder \
            .appName("test") \
            .getOrCreate()

        # 카프카 토픽 열기
        df = spark \
            .read \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "kafka:29092") \
            .option("subscribe", "mysql_kk") \
            .option("failONDataLoss", "false") \
            .load()
        
        # 카프카 토픽 열기
        df_pre = spark \
            .read \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "kafka:29092") \
            .option("subscribe", "flow") \
            .option("failONDataLoss", "false") \
            .load()

        # 데이터프레임 value 값 추출
        df1 = df.withColumn("value", df["value"].cast(StringType()))
        df_pre = df_pre.withColumn("value", df_pre["value"].cast(StringType()))
        df = df1.select(get_json_object(col("value"), "$.@timestamp").alias("datetime"),\
                get_json_object(col("value"), "$.user_id").alias("user_id"),\
                get_json_object(col("value"), "$.video_file").alias("path"))
        df_pre = df_pre.select(get_json_object(col("value"), "$.path").alias("path_pre"))
        
        # value 내의 파일경로부분을 'filepath' 변수로 추출
        filepath = df.tail(1)[0][2]
        path_pre = df_pre.tail(1)[0][0]

        # filepath 추출
        print("filepath: ",filepath)
        print("path_pre: ",path_pre)

        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str,
                            default='yolov5/weights/chanLeeV8.pt', help='model.pt path')
        
        # file/folder, 0 for webcam
        parser.add_argument('--source', type=str,
                            default=filepath, help='source')
        
        # 디폴트 저장위치 변경 !!
        parser.add_argument('--output', type=str, default='videos/result/',
                            help='output folder')  # output folder
        parser.add_argument('--img-size', type=int, default=640,
                            help='inference size (pixels)')
        
        # 객체 인식 확률
        parser.add_argument('--conf-thres', type=float,
                            default=0.4, help='object confidence threshold')
        
        # 트래킹 인식 확률 (객체 추적에서 IOU 개념 중요)
        parser.add_argument('--iou-thres', type=float,
                            default=0.5, help='IOU threshold for NMS')

        # 비디오 코덱 설정
        parser.add_argument('--fourcc', type=str, default='mp4v',
                            help='output video codec (verify ffmpeg support)')
        
        # GPU 사용?
        parser.add_argument('--device', default='',
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true',
                            help='display results')
        parser.add_argument('--save-txt', action='store_true',
                            help='save results to *.txt')
        
        # class 0 is person
        # 사람에 대한 객체만 인식하지만, 모든 객체 인식하도록 변경 가능
        parser.add_argument('--classes', nargs='+', type=int,
                            # help='filter by class')
                            help='filter by class')
        parser.add_argument('--agnostic-nms', action='store_true',
                            help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true',
                            help='augmented inference')
        parser.add_argument("--config_deepsort", type=str,
                            default="deep_sort_pytorch/configs/deep_sort.yaml")
        
        args = parser.parse_args()
        args.img_size = check_img_size(args.img_size)
        print(args)
        
        if filepath != path_pre:
            with torch.no_grad():
                detect(args)
        else:
            continue

        memory = []
        
        struct = df.drop('datetime')
        
        # 처리된 파일경로를 kafka에 전달
        query = df \
                .selectExpr("CAST(datetime AS STRING) AS key","to_json(struct(*)) AS value") \
                .write \
                .format("kafka") \
                .option("kafka.bootstrap.servers", "kafka:29092") \
                .option("topic","flow") \
                .option("checkpointLocation", "/spark/cpl") \
                .save()
        
        # 딜레이 시간 추가