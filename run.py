import os
import cv2
import argparse
import torch
import json
import numpy as np
import pandas as pd
from PIL import Image
import re
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import matplotlib
from google.cloud import storage
from concurrent import futures
from modules import Preprocess, Detection, OCR, Retrieval, Correction
from tool.config import Config 
from tool.utils import natural_keys, visualize, find_highest_score_each_class
import time

parser = argparse.ArgumentParser("Document Extraction")
parser.add_argument('--L', type=str, help='List of Video')
parser.add_argument('--V', type=str, default='', help='Video Index')
parser.add_argument("--input", help="Path to single image to be scanned")
parser.add_argument("--output", default="./results", help="Path to output folder")
parser.add_argument("--debug", action="store_true", help="Save every steps for debugging")
parser.add_argument("--do_retrieve", action="store_true", help="Whether to retrive information")
parser.add_argument("--find_best_rotation", action="store_true", help="Whether to find rotation of document in the image")
args = parser.parse_args()

def download_public_file(bucket_name, source_blob_name, destination_file_name):
    """Downloads a public blob from the bucket."""

    storage_client = storage.Client.create_anonymous_client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded public blob {} from bucket {} to {}.".format(
            source_blob_name, bucket.name, destination_file_name
        )
    )


def list_blobs(bucket_name, folder):
    """Lists all the blobs in the bucket."""

    children = []
    storage_client = storage.Client.create_anonymous_client()
    blobs = storage_client.list_blobs(bucket_name, prefix=folder)
    for blob in blobs: 
        children.append(blob.name)
    return children

class Pipeline:
    def __init__(self, args, config):
        self.output = args.output
        self.debug = args.debug
        self.do_retrieve = args.do_retrieve
        self.find_best_rotation = args.find_best_rotation
        self.load_config(config)
        self.make_cache_folder()
        self.init_modules()
        

    def load_config(self, config):
        self.det_weight = config.det_weight
        self.ocr_weight = config.ocr_weight
        self.det_config = config.det_config
        self.ocr_config = config.ocr_config
        self.bert_weight = config.bert_weight
        self.class_mapping = {k:v for v,k in enumerate(config.retr_classes)}
        self.idx_mapping = {v:k for k,v in self.class_mapping.items()}
        self.dictionary_path = config.dictionary_csv
        self.retr_mode = config.retr_mode
        self.correction_mode = config.correction_mode

    def make_cache_folder(self):
        self.cache_folder = os.path.join(args.output, 'cache')
        os.makedirs(self.cache_folder,exist_ok=True)
        self.preprocess_cache = os.path.join(self.cache_folder, "preprocessed.jpg")
        self.detection_cache = os.path.join(self.cache_folder, "detected.jpg")
        self.crop_cache = os.path.join(self.cache_folder, 'crops')
        self.final_output = os.path.join(self.output, 'result.jpg')
        self.retr_output = os.path.join(self.output, 'result.txt')

    def init_modules(self):
        self.det_model = Detection(
            config_path=self.det_config,
            weight_path=self.det_weight)
        self.ocr_model = OCR(
            config_path=self.ocr_config,
            weight_path=self.ocr_weight)
        self.preproc = Preprocess(
            det_model=self.det_model,
            ocr_model=self.ocr_model,
            find_best_rotation=self.find_best_rotation)
  
        if self.dictionary_path is not None:
            self.dictionary = {}
            df = pd.read_csv(self.dictionary_path)
            for id, row in df.iterrows():
                self.dictionary[row.text.lower()] = row.lbl
        else:
            self.dictionary=None

        self.correction = Correction(
            dictionary=self.dictionary,
            mode=self.correction_mode)

        if self.do_retrieve:
            self.retrieval = Retrieval(
                self.class_mapping,
                dictionary=self.dictionary,
                mode = self.retr_mode,
                bert_weight=self.bert_weight)

    def start(self, img):
        # Document extraction
        img1 = self.preproc(img)

        if self.debug:
            saved_img = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.preprocess_cache, saved_img)

            boxes, img2  = self.det_model(
                img1,
                crop_region=True,
                return_result=True,
                output_path=self.cache_folder)
            saved_img = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.detection_cache, saved_img)
        else:
            boxes = self.det_model(
                img1,
                crop_region=True,
                return_result=False,
                output_path=self.cache_folder)

        img_paths=os.listdir(self.crop_cache)
        img_paths.sort(key=natural_keys)
        img_paths = [os.path.join(self.crop_cache, i) for i in img_paths]
        
        texts = self.ocr_model.predict_folder(img_paths, return_probs=False)
        texts = self.correction(texts, return_score=False)

        return texts
        print(texts)
        
        if self.do_retrieve:
            preds, probs = self.retrieval(texts)
        else:
            preds, probs = None, None

        visualize(
          img1, boxes, texts, 
          img_name = self.final_output, 
          class_mapping = self.class_mapping,
          labels = preds, probs = probs, 
          visualize_best=self.do_retrieve)

        if self.do_retrieve:
            best_score_idx = find_highest_score_each_class(preds, probs, self.class_mapping)
            with open(self.retr_output, 'w') as f:
                for cls, idx in enumerate(best_score_idx):
                    f.write(f"{self.idx_mapping[cls]} : {texts[idx]}\n")


if __name__ == '__main__':
    config = Config('./tool/config/configs.yaml')
    pipeline = Pipeline(args, config)

    paths = "../TransNet_Database"
    
    video_paths = sorted(glob.glob(f"{paths}/Keyframes_L{args.L}/*/"))
    video_paths = ['/'.join(i.split('/')[:-1]) for i in video_paths]

    # os.makedirs(des_path, exist_ok=True)

    for vd_path in video_paths:
        ocr_video = {}
        #vd, fr = vd_path.split('/')[-1].split('_')
        # if int(vd[1:]) == 19 and int(fr[1:]) < 46:
        #     continue 
        print(vd_path)
        # check_file = int(vd_path.split('/')[-1].replace('C02_V',''))
        # if check_file <= 349:
        #   print(f"Skip: {vd_path}")
        #   continue

        re_feats = []
        keyframe_paths = glob.glob(f'{vd_path}/*.jpg')
        keyframe_paths = sorted(keyframe_paths, key=lambda x : x.split('/')[-1].replace('.jpg',''))

        for keyframe_path in tqdm(keyframe_paths):

            start_time = time.time()
            img = cv2.imread(keyframe_path)
            ocr_video[keyframe_path.split('/')[-1].split('.')[0]] = pipeline.start(img)
            end_time = time.time()

            print(f"Executed {keyframe_path} in {end_time - start_time} s")

        json_ocr = json.dumps(ocr_video)
        with open("./%s.json"%(vd_path), "w") as outfile:
            outfile.write(json_ocr)



