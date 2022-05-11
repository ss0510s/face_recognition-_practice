# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 


from __future__ import division
import collections
import numpy as np
import glob
import os
import os.path as osp
from numpy.linalg import norm
from .face_align import *
from .model_zoo import *

__all__ = ['FaceAnalysis', 'Face']

# 결과 저장
Face = collections.namedtuple('Face', [
    'bbox', 'kps', 'det_score', 'embedding',
    'embedding_norm', 'normed_embedding',
    'landmark'
])

Face.__new__.__defaults__ = (None, ) * len(Face._fields)


class FaceAnalysis:
    # 생성자 함수
    # 모델 경로 ~/.insightface/models/name 폴더 내에 위치
    def __init__(self, name, root='./face_model'):
        self.models = {}
        root = os.path.abspath(root) # 모델 위치
        print(root)
        onnx_files = glob.glob(osp.join(root, name, '*.onnx')) # 모델 파일
        onnx_files = sorted(onnx_files) # 모델 파일 정렬
        for onnx_file in onnx_files:
            if onnx_file.find('_selfgen_')>0:
                #print('ignore:', onnx_file)
                continue
            # model_zoo에서 get_model로 모델을 얻음
            model = get_model(onnx_file)
            # models에 model 이름이 없는 경우
            if model.taskname not in self.models:
                # 모델 저장
                print('find model:', onnx_file, model.taskname)
                self.models[model.taskname] = model
            # 그렇지 않은 경우,
            else:
                # 모델 삭제
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models # 에러검출
        self.det_model = self.models['detection'] # 모델 저장

    # 모델 준비
    # threshold, detection size
    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        # 모델 리스트
        for taskname, model in self.models.items():
            # detection 인 경우
            # 모델 prepare
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size)
            else:
                model.prepare(ctx_id)

    # 모델 적용
    def get(self, img, max_num=0):
        # 모델 bbox, kpss
        # face detection
        bboxes, kpss = self.det_model.detect(img,
                                             threshold=self.det_thresh,
                                             max_num=max_num,
                                             metric='default')
        # bbox가 0이면 0 리턴
        if bboxes.shape[0] == 0:
            return []
        ret = []
        # 각각의 box에 대한 결과 추출
        for i in range(bboxes.shape[0]):
            # bbox 추출
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            embedding = None
            normed_embedding = None
            embedding_norm = None

            # face recognition
            if 'recognition' in self.models:
                assert kps is not None
                rec_model = self.models['recognition']
                aimg = norm_crop(img, landmark=kps)
                embedding = None
                embedding_norm = None
                normed_embedding = None
                # 임베딩 값 추출
                embedding = rec_model.get_feat(aimg).flatten()
                embedding_norm = norm(embedding)
                normed_embedding = embedding / embedding_norm

            # face에 결과를 담음
            face = Face(bbox=bbox,
                        kps=kps,
                        det_score=det_score,
                        embedding=embedding,
                        normed_embedding=normed_embedding,
                        embedding_norm=embedding_norm)
            ret.append(face)
        return ret

    # 사각형 그리기
    def draw_on(self, img, faces):
        import cv2
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(np.int)
            color = (0, 0, 255)
            # 사각형 그리기
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(np.int)
                #print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(img, (kps[l][0], kps[l][1]), 1, color,
                               2)
        return img

