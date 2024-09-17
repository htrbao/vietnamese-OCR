# **An OCR Toolbox for Vietnamese Documents**

## Configurations
<p align="left">
 <a href=""><img src="https://img.shields.io/badge/python-3.9-aff.svg"></a>
</p>

### Run locally
- Create conda environment, note that python version should be <span style="color:#9BB8ED;">Python 3.9</span>
```
conda create --name ocr-tool-box python=3.9
conda activate ocr-tool-box
```

- Install required packages

```
pip install -r requirements.txt --no-cache-dir
```

## **Pipeline**

<div align="center"> Main Pipeline</div>

![Alt Text](demo/pipeline1.png)

<div align="center"> Process Flow Block</div>

![Alt Text](demo/pipeline2.png)

There are two stages (can also run in second stage only):
  - The first stage is to detect and rectify document in the image, then forward through the "process flow" to find the best orientation of the document.
  - The second stage is to forward the rotated image through the entire "process flow" normally to retrieve information


- Full pipeline:
```
python run.py
```
- **Extra Parameters**
  - ***--L***: to specific the List of Videos you want.
  - ***--V***: to specific the Video you want.
  ```
  python run.py --L 01 --V 001
  ```
  This will work for only on Video `L01_V001`.


  ```
  python run.py --L 01
  ```
  This will work for all Video with format `L01_Vxxx`.

## References
- https://github.com/WenmuZhou/PAN.pytorch
- https://github.com/andrewdcampbell/OpenCV-Document-Scanner
- https://github.com/pbcquoc/vietocr
- https://github.com/kaylode/vietnamese-ocr-toolbox
