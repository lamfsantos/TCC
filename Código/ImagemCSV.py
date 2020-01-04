import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, cohen_kappa_score, f1_score, average_precision_score
from math import *
from decimal import *
import pandas as pd

from sklearn.model_selection import cross_val_score, StratifiedKFold
BLA = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import preprocessing
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier

class ImagemCSV(object): 
    def __init__(self, caminho, objReferencia):
        super(ImagemCSV, self).__init__()         
        if objReferencia=="":
            super(ImagemCSV, self).__init__()
            self._caminho = caminho

            self.df=pd.read_csv(self._caminho, sep=';')
            self.df = self.df[self.df.GT != 0]
            self.class_indices = set(self.df['GT'].values.ravel())- set((0,))

            self.GT=self.df['GT'].values
            self.Abrir()
        else: 
            self._caminho = objReferencia._caminho
            self._objReferencia=objReferencia
            self.GT=objReferencia.GT
            self.AbrircomReferencia()
        self.classificado=False

    def NumeroBandas(self):        
        return self._nbands
    
    def Abrir(self):        
        self._img = self.df.iloc[:, :-3]
        self.posicoes=[]
        self._nbands=len(self._img.columns)
        self.class_indices = set(self.GT.ravel())        
        self._vet_GT2=np.ravel(self.GT)
        self.PrepararAmostras()
        self.CarregarAmostras()

    def AbrircomReferencia(self):        
        self._img=self._objReferencia.df
        self._nbands=len(self._img.columns)

        self.posicoes=self._objReferencia.posicoes
        self.class_indices = self._objReferencia.class_indices      
        self._vet_GT2=self._objReferencia._vet_GT2
        self.PrepararAmostras()

        self.indice_grupo1=self._objReferencia.indice_grupo1
        self.indice_grupo2=self._objReferencia.indice_grupo2
        self.indice_grupo3=self._objReferencia.indice_grupo3
        
        self.grupo1_GT=self._objReferencia.grupo1_GT
        self.grupo2_GT=self._objReferencia.grupo2_GT
        self.grupo3_GT=self._objReferencia.grupo3_GT
        
        self.CarregarAmostras()
        
    def RedefinirBandas(self,vet_bandas):
        self._img=self.df[list(map(str, vet_bandas+1))]
        self._nbands=len(self._img.columns)
        self.CarregarAmostras()
 
    def Teste(self):
        '''
            Por que recall médio? 
            Porque essa métrica é calculada com base nos 
            acertos de cada classe individualmente e depois 
            é feito a média. Se alguma classe não apresenta 
            pixels o recall é penalizado e não posso eliminar
            nenhuma classe da minha imagem.
            (2) A acurácia pode considerar muitos acertos de 
            uma dada classe, aumentando o seu valor, mas pode 
            ter uma classe com poucos representantes que pode 
            sumir no processo e o algoritmo não ser penalizado.
        '''
        
        #grupo 3 e grupo 2
        clf1 = MLClassifier()
        #clf1.defclasses_indices(self.class_indices)
        grupoTreinamento=[]
        grupoTreinamentoGT=[]
        
        grupoTreinamento.extend(self.amostra_grupo2)
        grupoTreinamento.extend(self.amostra_grupo3)
        grupoTreinamentoGT.extend(self.grupo2_GT)
        grupoTreinamentoGT.extend(self.grupo3_GT)
        
        clf1.fit(grupoTreinamento,grupoTreinamentoGT)
        
        pred1=clf1.predict(self.amostra_grupo1)
        
        s1=recall_score(self.grupo1_GT,pred1 , average='macro')
        p1=precision_score(self.grupo1_GT, pred1, average='macro')
        a1=accuracy_score(self.grupo1_GT, pred1, normalize=True)
        k1=cohen_kappa_score(self.grupo1_GT, pred1)
        f11=f1_score(self.grupo1_GT, pred1, average='macro')
        #grupo 1 e grupo 3
        clf2 = MLClassifier()
        #clf2.defclasses_indices(self.class_indices)
        grupoTreinamento=[]
        grupoTreinamentoGT=[]
        grupoTreinamento.extend(self.amostra_grupo1)
        grupoTreinamento.extend(self.amostra_grupo3)
        grupoTreinamentoGT.extend(self.grupo1_GT)
        grupoTreinamentoGT.extend(self.grupo3_GT)
        clf2.fit(grupoTreinamento,grupoTreinamentoGT)
        pred2=clf2.predict(self.amostra_grupo2)
        s2=recall_score(self.grupo2_GT,pred2 , average='macro')
        p2=precision_score(self.grupo2_GT, pred2, average='macro')
        a2=accuracy_score(self.grupo2_GT, pred2, normalize=True)
        k2=cohen_kappa_score(self.grupo2_GT, pred2)
        f12=f1_score(self.grupo2_GT, pred2, average='macro')
       
        #grupo 1 e grupo 2
        clf3 = MLClassifier()
        #clf3.defclasses_indices(self.class_indices)
        grupoTreinamento=[]
        grupoTreinamentoGT=[]
        grupoTreinamento.extend(self.amostra_grupo1)
        grupoTreinamento.extend(self.amostra_grupo2)
        grupoTreinamentoGT.extend(self.grupo1_GT)
        grupoTreinamentoGT.extend(self.grupo2_GT)
        clf3.fit(grupoTreinamento,grupoTreinamentoGT)
        pred3=clf3.predict(self.amostra_grupo3)
        s3=recall_score(self.grupo3_GT,pred3 , average='macro')
        p3=precision_score(self.grupo3_GT,pred3, average='macro')        
        a3=accuracy_score(self.grupo3_GT, pred3, normalize=True)
        k3=cohen_kappa_score(self.grupo3_GT, pred3)
        f13=f1_score(self.grupo3_GT, pred3, average='macro')
        
        
        return (s1+s2+s3)/3,(p1+p2+p3)/3,(a1+a2+a3)/3,(k1+k2+k3)/3,(f11+f12+f13)/3

    def TesteKNN(self):
        '''
            Por que recall médio?
            Porque essa métrica é calculada com base nos
            acertos de cada classe individualmente e depois
            é feito a média. Se alguma classe não apresenta
            pixels o recall é penalizado e não posso eliminar
            nenhuma classe da minha imagem.
            (2) A acurácia pode considerar muitos acertos de
            uma dada classe, aumentando o seu valor, mas pode
            ter uma classe com poucos representantes que pode
            sumir no processo e o algoritmo não ser penalizado.
        '''

        #grupo 3 e grupo 2
        clf1 = KNeighborsClassifier()
        #clf1.defclasses_indices(self.class_indices)
        grupoTreinamento=[]
        grupoTreinamentoGT=[]

        grupoTreinamento.extend(self.amostra_grupo2)
        grupoTreinamento.extend(self.amostra_grupo3)
        grupoTreinamentoGT.extend(self.grupo2_GT)
        grupoTreinamentoGT.extend(self.grupo3_GT)

        clf1.fit(grupoTreinamento,grupoTreinamentoGT)

        pred1=clf1.predict(self.amostra_grupo1)

        s1=recall_score(self.grupo1_GT,pred1 , average='macro')
        p1=precision_score(self.grupo1_GT, pred1, average='macro')
        a1=accuracy_score(self.grupo1_GT, pred1, normalize=True)
        k1=cohen_kappa_score(self.grupo1_GT, pred1)
        f11=f1_score(self.grupo1_GT, pred1, average='macro')
        #grupo 1 e grupo 3
        clf2 = KNeighborsClassifier()
        #clf2.defclasses_indices(self.class_indices)
        grupoTreinamento=[]
        grupoTreinamentoGT=[]
        grupoTreinamento.extend(self.amostra_grupo1)
        grupoTreinamento.extend(self.amostra_grupo3)
        grupoTreinamentoGT.extend(self.grupo1_GT)
        grupoTreinamentoGT.extend(self.grupo3_GT)
        clf2.fit(grupoTreinamento,grupoTreinamentoGT)
        pred2=clf2.predict(self.amostra_grupo2)
        s2=recall_score(self.grupo2_GT,pred2 , average='macro')
        p2=precision_score(self.grupo2_GT, pred2, average='macro')
        a2=accuracy_score(self.grupo2_GT, pred2, normalize=True)
        k2=cohen_kappa_score(self.grupo2_GT, pred2)
        f12=f1_score(self.grupo2_GT, pred2, average='macro')

        #grupo 1 e grupo 2
        clf3 = KNeighborsClassifier()
        #clf3.defclasses_indices(self.class_indices)
        grupoTreinamento=[]
        grupoTreinamentoGT=[]
        grupoTreinamento.extend(self.amostra_grupo1)
        grupoTreinamento.extend(self.amostra_grupo2)
        grupoTreinamentoGT.extend(self.grupo1_GT)
        grupoTreinamentoGT.extend(self.grupo2_GT)
        clf3.fit(grupoTreinamento,grupoTreinamentoGT)
        pred3=clf3.predict(self.amostra_grupo3)
        s3=recall_score(self.grupo3_GT,pred3 , average='macro')
        p3=precision_score(self.grupo3_GT,pred3, average='macro')
        a3=accuracy_score(self.grupo3_GT, pred3, normalize=True)
        k3=cohen_kappa_score(self.grupo3_GT, pred3)
        f13=f1_score(self.grupo3_GT, pred3, average='macro')


        return (s1+s2+s3)/3,(p1+p2+p3)/3,(a1+a2+a3)/3,(k1+k2+k3)/3,(f11+f12+f13)/3
    
    def PrepararAmostras(self):
        self.indice_grupo1=[]
        self.indice_grupo2=[]
        self.indice_grupo3=[]
        
        self.indice_amostras_teste=[]
        self.grupo1_GT=[]
        self.grupo2_GT=[]
        self.grupo3_GT=[]
        
        self.classes_teste=[]
        self.numero_pixels_validos=0
        for i in self.class_indices:
            self.posicoes=[]
            indice_i_grupo1=[]
            indice_i_grupo2=[]
            indice_i_grupo3=[]

            grupo1_GTp=[]
            grupo2_GTp=[]
            grupo3_GTp=[]


            self.posicoes=np.nonzero(np.equal(self._vet_GT2, i)==1)[0]
            total_pixels=len(self.posicoes)
            self.numero_pixels_validos=self.numero_pixels_validos+total_pixels
            ngrupo=0
            for x in range(0, total_pixels):
                ngrupo=ngrupo+1
                
                if ngrupo==1:           
                    indice_i_grupo1.append(self.posicoes[x])
                    grupo1_GTp.append(i)
                elif ngrupo==2:
                    indice_i_grupo2.append(self.posicoes[x])
                    grupo2_GTp.append(i)
                else:
                    indice_i_grupo3.append(self.posicoes[x])
                    grupo3_GTp.append(i)
                    ngrupo=0

            #Compartilha nos grupos
            self.indice_grupo1.extend(indice_i_grupo1)
            self.grupo1_GT.extend(grupo1_GTp)
                       
            self.indice_grupo2.extend(indice_i_grupo2)
            self.grupo2_GT.extend(grupo2_GTp)  
            
            self.indice_grupo3.extend(indice_i_grupo3)
            self.grupo3_GT.extend(grupo3_GTp)
       
    def CarregarAmostras(self):
        self.amostra_grupo1=[]
        self.amostra_grupo2=[]
        self.amostra_grupo3=[]
        self._vet_img= self._img.values
        for x in self.indice_grupo1:
            self.amostra_grupo1.append(self._vet_img[x][:])   
        for x in self.indice_grupo2:
            self.amostra_grupo2.append(self._vet_img[x][:]) 
        for x in self.indice_grupo3:
            self.amostra_grupo3.append(self._vet_img[x][:])
        
    def Sensitivity(self):       
        if self.classificado==False:
            return 0
        else:
            return recall_score(self.classes_teste,self.imageSegmentada, average='macro') 
    
    def Numero_Pixels_Validos(self):
        return self.numero_pixels_validos
    

    def TreinamentoGMLC(self):
        image=np.array([self.amostras_treinamento])
        class_mask=np.array([self.classes_treinamento])
        class_indices = set(class_mask.ravel()) - set((0,))
        classes = TrainingClassSetCSV()
        classes.nbands = image.shape[-1]
        for i in class_indices:
            cl = TrainingClassCSV(image, class_mask, i)
            classes.add_class(cl)
        self.classes=classes

class TrainingClass:

    def __init__(self, X, y, index=0):
        self.n_features = X.shape[-1],  
        self.n_samples = np.sum(np.equal(y, index).ravel())
        y=np.array(y)
        X=np.array(X)
        self.index=index
        X = X.reshape(-1, X.shape[-1]).T  
        y = y.ravel()
        ii = np.argwhere(y == index)
        self.X = np.take(X, ii.squeeze(), axis=1)
        self.cov_m = np.cov ( self.X )
        self._inv_cov = np.linalg.inv (self.cov_m)
        self.m = np.average(self.X, axis=1 )
        self._log_det_cov = self.log_det_cov() 
        
    def pooled_covariance(self, classes):        
        sup=0
        inf=0
        for c in classes:
            sup=sup+(c.n_samples-1)*c.cov_m
            inf=inf+c.n_samples
        
        n_cov=sup/(inf-len(classes))
        self.cov_m = n_cov
        self._inv_cov = np.linalg.inv (self.cov_m)
        self._log_det_cov = self.log_det_cov()

    def log_det_cov(self):
        (evals, evecs) = np.linalg.eigh(self.cov_m)
        return np.sum(np.log([v for v in evals if v > 0]))

         
class MLClassifier ( object ):
    """
    A simple ML classifier
    """
    
    def __init__(self):
        self.classes = {}
        self.imageSegmentada = []        
              
    def fit (self, X, y):
        y=np.array(y)
        X=np.array(X)
        self.class_indices=set(y.ravel())- set((0,))
        self.classes = []
        for i in self.class_indices:
            cl = TrainingClass(X, y, i)
            #if cl.n_samples > cl.n_features:
            self.classes.append(cl)
    
    def predict(self, X, y=None):
        clmap = self.classify_image(np.array(X))
        return np.array(clmap)  
                
    def classify_image(self, X):
        shape = X.shape
        X = X.reshape(-1, shape[-1])        
        scores = np.empty((X.shape[0], len(self.classes)), np.float64)
        delta = np.empty_like(X, dtype=np.float64)
        Y = np.empty_like(delta)
        for (i, c) in enumerate(self.classes):
            if c.n_samples <= c.n_features:
                c.pooled_covariance(self.classes)
            scalar = 1 - 0.5 * c._log_det_cov
            delta = np.subtract(X, c.m, out=delta)
            Y = delta.dot(-0.5 * c._inv_cov, out=Y)
            scores[:, i] = np.einsum('ij,ij->i', Y, delta)
            scores[:, i] += scalar

        self.inds = np.array([c.index for c in self.classes], dtype=np.int16)
        mins = np.argmax(scores, axis=-1)
        return self.inds[mins].reshape(shape[:1])
