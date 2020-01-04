import os
import random
import numpy
import multiprocessing
import math
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from ImagemCSV import *
import PIL
import copy
from time import gmtime, strftime
from shutil import copyfile
import pickle	

numpy.set_printoptions(threshold=numpy.nan)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)

NDIM = 2      

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def valorObjetivoGA(individuo, imageAllBands, classificador='GMLC'):
    novoIndividuo=copy.copy(imageAllBands)
    novoIndividuo.RedefinirBandas((numpy.nonzero(individuo)[0]))    
    n_bands=novoIndividuo.NumeroBandas()
    try:
       s,_,a,_,_=novoIndividuo.TesteKNN()
       s =round(((0.8*a)+(0.2/novoIndividuo._nbands)),4)
       #nov =round(((0.8*novoIndividuo.Accuracy())+(0.2/novoIndividuo._nbands)),4)
    except:
        s=0
        n_bands=imageAllBands._nbands

    return s,

def valoresObjetivosPadrao(individuo, imageAllBands, classificador='GMLC'):
    novoIndividuo=copy.copy(imageAllBands)
    novoIndividuo.RedefinirBandas((numpy.nonzero(individuo)[0]))    
    n_bands=novoIndividuo.NumeroBandas()
    try:
       s,_,a,_,_=novoIndividuo.TesteKNN()
    except:
        s=0
        n_bands=imageAllBands._nbands
    return s,

if __name__ == "__main__":
        img="/home/furlan/Downloads/selecao90A.csv"

        ext=".hdr"
        ext_gt="_gt.png"
        
        img_ref=ImagemCSV(img,"")

        pv=0

        s1=img_ref.TesteKNN()

        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, img_ref.NumeroBandas())
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", valorObjetivoGA, imageAllBands=img_ref, classificador="GMLC")    
        
        pasta="/home/furlan/Downloads/Resultados/GA_GMLC_Pavia5"
        
        count =1
        NGEN=1001
        _numero_individuos=144
        _selecao="GA"
        _CXPB=0.50
        if not os.path.exists(pasta+"\log"+"_"+str(count)+".txt"):
            arquivo = open(pasta+"\log"+"_"+str(count)+".txt", 'w+')
            arquivo_front = open(pasta+"\log_front"+"_"+str(count)+".txt", 'w+')
            arquivo.close()
            arquivo_front.close()
        
        copyfile(pasta+"\log"+"_"+str(count)+".txt", pasta+"\logn"+"_"+str(count)+".txt")
        copyfile(pasta+"\log_front"+"_"+str(count)+".txt", pasta+"\logn_front"+"_"+str(count)+".txt")
        log = open(pasta+"\log"+"_"+str(count)+".txt", "w")
        log_front = open(pasta+"\log_front"+"_"+str(count)+".txt", "w")
    
        num_lines = sum(1 for line in open(pasta+"\logn"+"_"+str(count)+".txt"))    
        num_lines_front = sum(1 for line in open(pasta+"\logn_front"+"_"+str(count)+".txt"))    

        gen_ini=1
        
        with open(pasta+"\logn"+"_"+str(count)+".txt", "r") as f:
            for line in f:
              gen_ini=gen_ini+1

              if (gen_ini<=num_lines+1):
                 log.write(line)         	
                 log.flush()

        gen_ini_front=1            
        with open(pasta+"\logn_front"+"_"+str(count)+".txt", "r") as f:
            for line in f:
              gen_ini_front=gen_ini_front+1

              if (gen_ini_front<=num_lines_front+1):
                 log_front.write(line)         	
                 log_front.flush()
                    
        random.seed(None)
        MU = _numero_individuos
        
        if gen_ini>1:
            if os.path.exists(pasta+"\lbkp"+"_"+str(count)+str(gen_ini)+".txt"):
                gen_ini=gen_ini+1
            else:
                gen_ini=gen_ini
            caminho=pasta+"\\lbkp"+"_"+str(count)+str(gen_ini-1)+".txt"
            rest = open(pasta+"\\lbkp"+"_"+str(count)+str(gen_ini-1)+".txt", 'rb')
            pop = pickle.load(rest, encoding='latin1')
            rest.close()
        else:
            pop = toolbox.population(_numero_individuos) 
        
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        print("teste")
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop = toolbox.select(pop, len(pop))

        for gen in range(gen_ini, NGEN):		
            print (gen, strftime("%H:%M:%S", gmtime()))

            if _selecao=="GA": 
                offspring = toolbox.select(pop, len(pop))
            else:
                offspring = tools.selTournamentDCD(pop, len(pop))
                
            offspring = list(map(toolbox.clone, offspring))

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= _CXPB:
                    toolbox.mate(ind1, ind2)
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values
        
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop = toolbox.select(pop + offspring, MU)
            fits = []
        
            best_ind=[]
            best_fit=0
            n_bads=0
            
            if _selecao=="NSGA2": 
                pareto_fronts=tools.sortNondominated(pop,k=1,first_front_only=True)
                best_ind,best_fit,n_bads=selecionaMelhor(pareto_fronts)
        
                for ind in pareto_fronts[0]:
                    fits.append(ind.fitness.values[0])      
            else:                
                for ind in pop:
                    fits.append(ind.fitness.values[0])
                    if ((ind.fitness.values[0])>best_fit):
                        novoIndividuo=copy.copy(img_ref)
                        best_fit=ind.fitness.values[0]

                        novoIndividuo.RedefinirBandas((numpy.nonzero(ind)[0]))    
                        best_ind = str(ind)
                        n_bads=sum(ind)
                        
            best_fit,_,_,_,_=novoIndividuo.TesteKNN()
            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5
        
            log.write(str(gen)+';'+str(min(fits))+';'+str(max(fits))+';'+str(s1)+';'+str(mean)+';'+str(std)+';'+str(best_fit)+';'+str(n_bads)+';'+best_ind+'\n')
            
            if _selecao=="NSGA2":
                for ind in pareto_fronts[0]:
                    log_front.write(str(gen)+';'+str(ind.fitness.values[0])+';'+str(ind.fitness.values[1])+';'+str(ind)+'\n')
                    log_front.flush()

            log.flush()
            bkp = open(pasta+"\lbkp"+"_"+str(count)+str(gen)+".txt", 'wb')
            pickle.dump(pop, bkp)
            bkp.flush()
            bkp.close

            if os.path.exists(pasta+"\lbkp"+"_"+str(count)+str(gen-1)+".txt"):
               os.remove(pasta+"\lbkp"+"_"+str(count)+str(gen-1)+".txt")

        log.close
        log_front.close

def selecionaMelhor(pareto_fronts):
        vet_recall=[]
        vet_n_bands=[]

        for ind in pareto_fronts[0]:
            vet_recall.append(ind.fitness.values[0])
            vet_n_bands.append(ind.fitness.values[1])
            
        max_recall=max(vet_recall)
        min_recall=min(vet_recall)
        max_n_bands=max(vet_n_bands)
        min_n_bands=min(vet_n_bands)
        vet_recall=[]
        vet_n_bands=[]
        vet_recall_n_bands=[]
        for ind in pareto_fronts[0]:
            
            if len(pareto_fronts[0])==1:
                vet_recall.append(1)
                vet_n_bands.append(0)
                vet_recall_n_bands.append([1,0])
            else:
                vet_recall.append((ind.fitness.values[0]-min_recall)/(max_recall-min_recall))
                vet_n_bands.append((ind.fitness.values[1]-min_n_bands)/(max_n_bands-min_n_bands))
                vet_recall_n_bands.append([(ind.fitness.values[0]-min_recall)/(max_recall-min_recall),(ind.fitness.values[1]-min_n_bands)/(max_n_bands-min_n_bands)])

                       
        media_recall=sum(vet_recall) / float(len(vet_recall))
        media_n_bands=sum(vet_n_bands) / float(len(vet_n_bands))
        
        solucao=0
        menor_distancia=100
        solucao_ideal=0
        maior_recall=-1
        solucao_antiga=0
        for v_recall, v_n_bands in vet_recall_n_bands:
            solucao=solucao+1
            if ((v_recall>=media_recall) and (v_n_bands<=media_n_bands)):
                dist_euclidiana = math.sqrt((v_recall-media_recall)**2+(v_n_bands-media_n_bands)**2)
                if dist_euclidiana<menor_distancia:
                    menor_distancia=dist_euclidiana
                    solucao_ideal=solucao
                elif dist_euclidiana==menor_distancia:
                    if maior_recall<v_recall:
                        maior_recall=v_recall
                        solucao_ideal=solucao
                    else:
                        solucao_ideal=solucao_antiga
                    solucao_antiga=solucao_ideal

        solucao=0
        maior_recall=-1
        menor_distancia=100
        if (solucao_ideal==0):
            for v_recall, v_n_bands in vet_recall_n_bands:
                solucao=solucao+1
                dist_euclidiana = math.sqrt((v_recall-(media_recall+(1-media_recall)/2))**2+(v_n_bands-media_n_bands/2)**2)
                if dist_euclidiana<menor_distancia:
                    menor_distancia=dist_euclidiana
                    solucao_ideal=solucao
                elif dist_euclidiana==menor_distancia:
                    if maior_recall<v_recall:
                        maior_recall=v_recall
                        solucao_ideal=solucao
                    else:
                        solucao_ideal=solucao_antiga
                    solucao_antiga=solucao_ideal

        busca_ideal=0
        for ind in pareto_fronts[0]:
            busca_ideal=busca_ideal+1
            if busca_ideal==solucao_ideal:
                return str(ind),ind.fitness.values[0],ind.fitness.values[1]