import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from PIL import Image
from glob import glob
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import models
from PIL import Image



class Utils(object):
    def __init__(self ,data_path,input_data,output_filled_data):
        self.data_path=data_path
        self.input_data=input_data
        self.output_filled_data=output_filled_data
        self.Name=[]
        self.test=[]
        self.load_img_data=load_img_path(self.input_data)
        self.output_filled=load_img_path(self.output_filled_data)
        
        
    def fix_data_filled(self):
        
        for i in range(len(self.output_filled)):
            if self.output_filled[i][self.output_filled[i].index('N')+1] == '-' :
                nom=self.output_filled[i][self.output_filled[i].index('N'):self.output_filled[i].index('N')+8]
            else :            
                nom=self.output_filled[i][self.output_filled[i].index('N'):self.output_filled[i].index('N')+7]
            id_image_last_caracter=self.output_filled[i].index('.png')
            id_image_first_caracter=len(self.output_filled[i])-self.output_filled[i][::-1].index('_')
            ch=str(self.input_data)+'/thm_dir_'+nom+'_'+str(self.output_filled[i][id_image_first_caracter:id_image_last_caracter])+'.png'
            self.Name.append(nom)
            self.test.append(ch)
        return self.Name, self.test
    def DataToDataframe(self):
        data=[]
        for i in range(4950) :
            data.append([self.test[i],self.output_filled[i]])
        df=pd.DataFrame(data,columns=['input_image','output'])
        return df
    def visualize_data(self,N):
        df=self.DataToDataframe()
        plt.figure(figsize=(10,10))
        plt.subplot(1,3,1)
        img=cv2.imread(df.input_image.iloc[N])
        plt.imshow(img)
        plt.subplot(1,3,2)
        msk=cv2.imread(df.output.iloc[N])
        plt.imshow(msk)
        plt.subplot(1,3,3)
        plt.imshow(img)
        plt.imshow(msk,alpha=0.5)
    def data_loader(self):
        df=self.DataToDataframe()
            
        data_train,data_test=train_test_split(df,test_size=0.1)
        data_train,data_val=train_test_split(data_train,test_size=0.2)
        datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,zoom_range=0.2)
        datagenM = ImageDataGenerator(rescale=1./255,horizontal_flip=True,vertical_flip=True,zoom_range=0.2)



        image_train=datagen.flow_from_dataframe(data_train,  
                                target_size=(256,256), 
                                color_mode='grayscale',
                                shuffle=True,
                                seed=42,
                                x_col ="input_image", 
                                batch_size=32,
                                class_mode=None
                                
                                )
        mask_train=datagenM.flow_from_dataframe(data_train, 
                                target_size=(256,256), 
                                color_mode='grayscale',
                                shuffle=True,
                                seed=42,
                                x_col ="output", 
                                batch_size=32,
                                class_mode=None
                                )
        image_validation=datagen.flow_from_dataframe(data_val,  
                                target_size=(256,256), 
                                color_mode='grayscale',
                                shuffle=True,
                                seed=42,
                                x_col ="input_image", 
                                batch_size=32,
                                class_mode=None
                                )

        mask_validation=datagenM.flow_from_dataframe(data_val, 
                                target_size=(256,256), 
                                color_mode='grayscale',
                                shuffle=True,
                                seed=42,
                                x_col ="output", 
                                batch_size=32,
                                class_mode=None
                                )
        self.train_gen=zip(image_train,mask_train)
        self.valid_gen=zip(image_validation,mask_validation)
def load_img_path(data_path):
    L=[]
    for filename in os.listdir(data_path) :
        L.append(os.path.join(data_path,filename))
    return L

