import os
import json
from tqdm import tqdm
import random
import shutil

jsondir = r"WoodScape/annotations/gtLabels"
jpgdir = r"WoodScape/images"

jpgtraindir = r"WoodScape/splitData/images/train"
jpgvaldir = r"WoodScape/splitData/images/val"

jsontraindir = r"WoodScape/splitData/gtLabels/train"
jsonvaldir = r"WoodScape/splitData/gtLabels/val"

os.makedirs(jpgtraindir, exist_ok=True)
os.makedirs(jpgvaldir, exist_ok=True)
os.makedirs(jsontraindir, exist_ok=True)
os.makedirs(jsonvaldir, exist_ok=True)


jsonlist = os.listdir(jsondir)
random.seed(42)
random.shuffle(jsonlist)


for jsonname in jsonlist[:-1000]:
    name = jsonname.split('.')[0]
    imgname = name + '.png'
    imgsrc = os.path.join(jpgdir, imgname)
    imgdst = os.path.join(jpgtraindir, imgname)
    jsonsrc = os.path.join(jsondir, jsonname)
    jsondst = os.path.join(jsontraindir, jsonname)
    

    if os.path.exists(imgsrc) and os.path.exists(jsonsrc):
        shutil.copyfile(imgsrc, imgdst)
        shutil.copyfile(jsonsrc, jsondst)
    else:
        print(f"Warning: Source file not found: {imgsrc} or {jsonsrc}")


for jsonname in jsonlist[-1000:]:
    name = jsonname.split('.')[0]
    imgname = name + '.png' 
    imgsrc = os.path.join(jpgdir, imgname)
    imgdst = os.path.join(jpgvaldir, imgname)
    jsonsrc = os.path.join(jsondir, jsonname)
    jsondst = os.path.join(jsonvaldir, jsonname)
    
    if os.path.exists(imgsrc) and os.path.exists(jsonsrc):
        shutil.copyfile(imgsrc, imgdst)
        shutil.copyfile(jsonsrc, jsondst)
    else:
        print(f"Warning: Source file not found: {imgsrc} or {jsonsrc}")
