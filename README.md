# Segmentation Convolutional Neural Networks for Automatic Crater Detection on Mars [TSYP CS Challenge 2022]
# Preparation
To begin, you will need to clone this GitHub repository :
```bash 
git clone github.com/Soulaimene/CS_Challenge_2022.git

```
To obtain the data, you will need to access the [drive](https://drive.google.com/drive/folders/1W23o71RVu513O8AT20DxYGukyeAhguD6?usp=share_link) we have created. In this drive, you will find a zipped dataset called "Input_data.zip." Download this file and unzip it in the cloned folder.

Before training, if you want to use the dataset to train a model for semantic segmentation of filled data, you will need to : 
* Access the "filled" folder in the drive and download the zipped dataset "output filled data." 
* Unzip the downloaded dataset and save it in the cloned folder. 

Similarly, if you want the thick or thin versions, you will need to access the respective folders called "thin" or "thick" and download the corresponding zipped datasets.
# How to train
```bash 
python main.py --n_epoch 50 --n_bs 32  --n_unet 0
```
> Note : If you did not save the datasets in the cloned folder, you will need to specify the path to your input data and masks data, as well as the path to all other relevant data.
```bash 
python main.py --n_epoch 50 --n_bs 32  --n_unet 0 --data_p "/data_path/" --Input_p "/Input_data_path/" --mask_p "/mask_data_path/"
```
You need to ensure that you have a compatible version of TensorFlow installed, as well as the appropriate versions of CUDA and cuDNN. These tools allow TensorFlow to use the GPU for acceleration, which can significantly improve the performance of machine learning tasks. 
