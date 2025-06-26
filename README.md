# DCHF-Bloc de fuziune adaptivă folosind volume multiple și filtre tridimensionale pentru estimarea adâncimii din imagini

 Acest repository reprezintă lucrarea finală de diplomă unde am încercat să îmbunătățesc rezultatele estimării adâncimii din imagini.

## Arhitectura modelului
Modelul folosește o arhitectură siameză, cu două sub-rețele pentru estimarea adâncimii monoculare.

![This is an alt text.](/ImagesReadME/TiO-Depth.png "This is a sample image.")

## Blocul MFM 
Acesta este cel care se ocupă cu comunicarea celor două rețele pentru generarea informațiilor stereo

În varianta clasică acest bloc nu este suficient de expresiv, reprezentând o simplă atenție încrucișată

![This is an alt text.](/ImagesReadME/MFM "This is a sample image.")

Mai jos este varianta propusă care folosește volume multiple, filtre tridimensionale și fuziune adaptivă.

![This is an alt text.](/ImagesReadME/DCHF "This is a sample image.")

![This is an alt text.](/ImagesReadME/Hourglass "This is a sample image.")

## Predicții
Pentru a rula predicții se poate utiliza notebook-ul atașat acestui repository, cu mențiunea că trebuie descărcat și modelul antrenat.
https://drive.google.com/drive/folders/1g5A43tpyG4hiGFsApZQOF8U-Dw60Xb6P?usp=sharing

## Diagrame de clasă
![This is an alt text.](/ImagesReadME/DiagramaClaseModel "This is a sample image.") 

![This is an alt text.](/ImagesReadME/DiagramaClaseDecodificator "This is a sample image.")

## Setup
Pentru antrenarea modelului a fost folosită o placă NVIDIA GeForce RTX 4070 Ti.  
Setul de date pe care a fost antrenat și evaluat modelul este KITTI.  
Ca limbaj de programare, a fost folosit Python împreună cu biblioteca de învățare automată PyTorch.  

Pentru viitoare studii se recomandă crearea unui repo cu CUDA 11.0, Python 3.7.11, și Pytorch 1.7.0.  
Acestea pot fi făcute cu comenzile:
```
conda env create -f environment.yml
conda activate pytorch170cu11
```

## Recunoaștere
[TiO-Depth](https://github.com/ZM-Zhou/TiO-Depth_pytorch)  
[Mmcv](https://github.com/open-mmlab/mmcv)  
[Monodepth2](https://github.com/nianticlabs/monodepth2)  
[KITTI Dataset](http://www.cvlibs.net/datasets/kitti/index.php)  
