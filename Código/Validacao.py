import random
from deap import base
from deap import creator
from deap import tools
import copy
from shutil import copyfile
import os.path
from time import gmtime, strftime
import pickle
from ImagemCSV import *
import os
import random
import json
import numpy as np
from math import sqrt
from deap import algorithms
import PIL
import copy
from itertools import chain
from random import randint
from sklearn.metrics import accuracy_score, precision_score, recall_score, cohen_kappa_score, f1_score, average_precision_score
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from random import randint
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy import stats
from sklearn import preprocessing
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

spy_colors = np.array([[0, 0, 0],
                          [255, 0, 0],
                          [0, 255, 0],
                          [0, 0, 255],
                          [255, 255, 0],
                          [255, 0, 255],
                          [0, 255, 255],
                          [200, 100, 0],
                          [0, 200, 100],
                          [100, 0, 200],
                          [200, 0, 100],
                          [100, 200, 0],
                          [0, 100, 200],
                          [150, 75, 75],
                          [75, 150, 75],
                          [75, 75, 150],
                          [255, 100, 100],
                          [100, 255, 100],
                          [100, 100, 255],
                          [255, 150, 75],
                          [75, 255, 150],
                          [150, 75, 255],
                          [50, 50, 50],
                          [100, 100, 100],
                          [150, 150, 150],
                          [200, 200, 200],
                          [250, 250, 250],
                          [100, 0, 0],
                          [200, 0, 0],
                          [0, 100, 0],
                          [0, 200, 0],
                          [0, 0, 100],
                          [0, 0, 200],
                          [100, 100, 0],
                          [200, 200, 0],
                          [100, 0, 100],
                          [200, 0, 200],
                          [0, 100, 100],
                          [0, 200, 200]], np.int)
 
def mapColors(im,d={}):
    newimdata = []
    im=Image.fromarray(im, 'L')
    if not d:
        classes=set(np.array(im).ravel())- set((0,))
        d = {}
        d[0]=(0,0,0)
        for i in classes:
            d[i]=(spy_colors[i][0],spy_colors[i][1],spy_colors[i][2])
 
    for color in im.getdata():
        newimdata.append(d[color])
    newim = Image.new('RGB',im.size)
    newim.putdata(newimdata)
    return newim,d
 
def imgNova(individuo, imageAllBands, imageAllBands2, img2):
     
             
        df=pd.read_csv(img2, sep=';')
        
        df=df[df.columns[-3:]]
         
        gt_validacao=[]
        for i, row in df.iterrows():
            gt_validacao.append(row['GT'])
             
        gmlc = KNeighborsClassifier(9)

        novoIndividuo=copy.copy(imageAllBands)
        novoIndividuo.RedefinirBandas((np.nonzero(individuo)[0]))        
 
        grupoTreinamento=[]
        grupoTreinamentoGT=[]
        grupoTreinamento.extend(novoIndividuo.amostra_grupo1)
        grupoTreinamentoGT.extend(novoIndividuo.grupo1_GT)
        grupoTreinamento.extend(novoIndividuo.amostra_grupo2)
        grupoTreinamentoGT.extend(novoIndividuo.grupo2_GT)
        grupoTreinamento.extend(novoIndividuo.amostra_grupo3)
        grupoTreinamentoGT.extend(novoIndividuo.grupo3_GT)
        scaler = StandardScaler()
        grupoTreinamento=scaler.fit_transform(grupoTreinamento)
        gmlc.fit(grupoTreinamento,grupoTreinamentoGT)
         
        '''classifica imagem'''
        novoIndividuo2=copy.copy(imageAllBands2)
        novoIndividuo2.RedefinirBandas((np.nonzero(individuo)[0]))  
        pred1=gmlc.predict(scaler.fit_transform(list(novoIndividuo2._vet_img)))
                 
         
        acc=accuracy_score(gt_validacao, pred1, normalize=True)

        return acc, 0, 0
 
 
def main():
    Img="Indian_Pines"
    
    Img_csv_Path=Img+"/DadosDoc4"
    Pasta="Final Geral/IRMOBS/"+Img
    ext_gt="_gt.png"
     
    Algoritmo="\IRMOBS_Geral_"
    list_resultsacc_IDMMoBBS=[]
    list_resultsrec_IDMMoBBS=[]
    list_resultskap_IDMMoBBS=[]
    list_resultsbands_IDMMoBBS=[]

    #imagem1
    imgPath='/home/furlan/Desktop/Resultados_Finais/'+Pasta+Algoritmo+Img+'1/'
    
    imgName=imgPath+Img+'A.png'

    img='/home/furlan/Desktop/Resultados_Finais/'+Img_csv_Path+'/selecao90A.csv'    

    img2='/home/furlan/Desktop/Resultados_Finais/'+Img_csv_Path+'/validacao10A.csv'
    
    img_ref=ImagemCSV(img, "")
    img_ref2=ImagemCSV(img2, "")

    ###
    individuo=[0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
    ###

    acc,rec, kap =imgNova(individuo,img_ref,img_ref2,img2)
    list_resultsacc_IDMMoBBS.append(100*acc)
    list_resultsrec_IDMMoBBS.append(100*rec)
    list_resultskap_IDMMoBBS.append(100*kap)
    list_resultsbands_IDMMoBBS.append(list(map(int, individuo)).count(1))
    print(Algoritmo,acc,rec, kap, list(map(int, individuo)).count(1))    
 
    print("====================================== Results====================================== \n")
    
    print("======IDMMOBS========================================= \n")
    print("======OA====== \n")
    print("Valores",list_resultsacc_IDMMoBBS)
    print("Media", np.mean(list_resultsacc_IDMMoBBS))
    print("Desvio Padrão", np.std(list_resultsacc_IDMMoBBS))
    print("======BANDAS====== \n")
    print("Media Bandas", np.mean(list_resultsbands_IDMMoBBS))
    
if __name__ == "__main__":
    main()