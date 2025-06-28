# DCHF-Bloc de fuziune adaptivă folosind volume multiple și filtre tridimensionale pentru estimarea adâncimii din imagini

Link repository gitlab:  
[Link Gitlab proiect](https://gitlab.upt.ro/fabian.dogaru/DCHF-Bloc-de-fuziune-pentru-estimarea-adancimii-din-imagini/-/tree/main)

## Predicții Colab
Pentru a rula predicții rapid se poate utiliza notebook-ul atașat acestui repository, cu mențiunea că trebuie descărcat și modelul antrenat.  
[Link Drive](https://drive.google.com/drive/folders/1g5A43tpyG4hiGFsApZQOF8U-Dw60Xb6P?usp=sharing)

## Setup
#### Detalii model
Setul de date pe care a fost antrenat și evaluat modelul este KITTI.  
Ca limbaj de programare, a fost folosit Python împreună cu biblioteca de învățare automată PyTorch.  

#### Mediu virtual
Pentru a antrena sau a face predicții local, se recomandă crearea unui mediu virtual cu CUDA 11.0, Python 3.7.11, și Pytorch 1.7.0.  
Acestea pot fi făcute cu comenzile:
```
conda env create -f environment.yml
conda activate pytorch170cu11
```
În cazul în care se dorește folosirea venv, pachetele pot fi instalate folosind requirements.txt
#### Predicții local
Pentru a face predicții local se folosește script-ul de mai jos, dar înainte trebuie descărcat fișierul .pth de la link-ul de mai sus.
```
python predict.py\
 --image_path <calea către o imagine sau un director de imagini >\
 --exp_opts <calea către opțiunea metodei de antrenare a modelului>\
 --model_path <calea către model (fișierul .pth)>
```
#### KITTI
Pentru antrenarea modelului trebie să fie descărcat setul de date KITTI. Pentru acest lucru se poate descărca întregul set (aproximativ 175GB) cu comanda de mai jos:
```
wget -i ./datasets/kitti_archives_to_download.txt -P <calea unde se va salva>
```
Apoi se face dezarhivare cu:
```
cd <calea unde a fost salvat>
unzip "*.zip"
```

Pentru evaluarea metodelor pe KITTI trebuie generate fișierele ce conțin estimările reale (așa cum a fost făcut și în [Monodepth2](https://github.com/nianticlabs/monodepth2)):

```
python datasets/utils/export_kitti_gt_depth.py --data_path <root of KITTI> --split raw
```

#### Antrenare locală
Mai întâi se descarcă modelul Swin-Transformer(Tiny Size) din [repo oficial](https://github.com/microsoft/Swin-Transformer) și se setează călea către acesta în `path_my.py`.
După toate aceste se poate utiliza script-ul

`options/TiO-Depth/train/train_scripts.sh`.
