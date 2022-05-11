from sklearn.neighbors import NearestNeighbors
import numpy as np
from PIL import Image
from .face_analysis import *
import cv2
import os
from typing import List
from tqdm import tqdm

'''
    모델 불러오는 함수
        인자 : model_path = "모델 파일이 있는 폴더 이름"
'''
def model_prepare(model_path):
    app = FaceAnalysis(name = model_path)
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

'''
    임베딩값 처리
    - 비어있는 값 제외
    - 임베딩 값만 추출
'''
def filter_empty_embs(img_set: List, img_labels: List[str]):
    # filtering where insightface could not generate an embedding
    good_idx = [i for i,x in enumerate(img_set) if x]
    
    if len(good_idx) == len(img_set):
        clean_embs = [e[0].embedding for e in img_set]
        clean_labels = img_labels
        
    else:
        # filtering eval set and labels based on good idx
        clean_labels = np.array(img_labels)[good_idx]
        clean_set = np.array(img_set, dtype=object)[good_idx]
        
        # generating embs for good idx
        clean_embs = [e[0].embedding for e in clean_set]
    
    return clean_embs, clean_labels

'''
    임베딩 생성 함수
        인자: dir_path = "이미지 폴더 경로"
            app = "FaceAnalysis 객체"
'''
def embs_result(dir_path : str,  app : FaceAnalysis):
    files = os.listdir(dir_path)[0:] # 이미지 파일 리스트
    files.sort()

    eval_set = list()
    eval_labels = list()
    IMAGES_PER_IDENTITY = 11
    
    # 이미지 파일 내에서 각각의 이미지에 대하여 추출
    for i in tqdm(range(1, len(files), IMAGES_PER_IDENTITY), unit_divisor=True):
        # store eval embs and labels
        # eval_set_t, eval_labels_t = generate_embs(app,img_path)
        
        eval_set_t = list()
        eval_labels_t = list()
        
        # 각각의 이미지에 대한 임베딩값 추출
        for img_fpath in files:
                    
            # read grayscale img
            img = Image.open(os.path.join(dir_path, img_fpath)) 
            img_arr = np.asarray(img)  
            
            # convert grayscale to rgb
            im = Image.fromarray((img_arr * 255).astype(np.uint8))
            rgb_arr = np.asarray(im.convert('RGB'))       
        
            # generate Insightface embedding
            res = app.get(rgb_arr)          
            # append emb to the eval set
            eval_set_t.append(res)          
            # append label to eval_label set
            eval_labels_t.append(img_fpath.split("_")[0])  
        
        eval_set.extend(eval_set_t)
        eval_labels.extend(eval_labels_t)
    
    evaluation_embs, evaluation_labels = filter_empty_embs(eval_set, eval_labels)
    return evaluation_embs, evaluation_labels

'''
    비교 함수
        : img_fpath 이미지에 대해 evaluation_embs에 저장된 값들에 있는 지 비교
'''
def print_ID_results(evaluation_embs:list, app : FaceAnalysis, img_fpath: str, evaluation_labels: np.ndarray, verbose: bool = False):      
    
    nn = NearestNeighbors(n_neighbors=3, metric="cosine")
    nn.fit(X=evaluation_embs)

    img_set = list()
    # read grayscale img
    img = Image.open(img_fpath) 
    img_arr = np.asarray(img)  
    
    # convert grayscale to rgb
    im = Image.fromarray((img_arr * 255).astype(np.uint8))
    rgb_arr = np.asarray(im.convert('RGB'))       

    # generate Insightface embedding
    res = app.get(rgb_arr)   
    
    for i in range (len(res)):
        img_emb = res[i].embedding
        img_set.append(res)  

        # get pred from KNN
        dists, inds = nn.kneighbors(X=img_emb.reshape(1,-1), n_neighbors=3, return_distance=True)

        # get labels of the neighbours
        pred_labels = [evaluation_labels[i] for i in inds[0]]

        # check if any dist is greater than 0.5, and if so, print the results
        no_of_matching_faces = np.sum([1 if d <=0.5 else 0 for d in dists[0]])
        if no_of_matching_faces > 0:
            print(f"Matching face(s) {i} found in database! dist : {dists}")
            verbose = True
        else: 
            print(f"No matching face(s) not found in database! dist : {dists}")