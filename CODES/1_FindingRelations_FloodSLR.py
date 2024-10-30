"""
Created on Fri Mar 17 08:54:24 2023

@author: sgilclavel
"""

import os as os
import glob
from datetime import date

from tqdm import tqdm

import pickle
import subprocess

# Data Handling
import pandas as pd
tqdm.pandas(desc="my bar!")

import csv

# Transform list representation into list
import ast 

# Network Analysis
import networkx as nx

# Basic XXX Libraries
import numpy as np
from collections import Counter
import random

# Processing text
import nltk as nltk
import regex as re
from bs4 import BeautifulSoup

from itertools import product

# Import Text Processing Libraries
import spacy as spacy
# Next is to tokenize
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex,compile_infix_regex, compile_suffix_regex
# Next is for the pipeline
# nlp = spacy.load('en_core_web_trf') # Small English language model
# nlp = spacy.load('en_core_sci_lg')
# import neuralcoref
# neuralcoref.add_to_pipe(nlp)

# Plots 
# from wordcloud import WordCloud
from matplotlib import pyplot as plt
from matplotlib import patches
# import seaborn as sns
import textwrap as twr

# =============================================================================
# Functions
# =============================================================================
# os.chdir("C:\\Dropbox\\TU_Delft\\Projects\\ML_FindingsGrammar\\CODE\\Processing_PDFs\\")
# os.chdir("C:\\Dropbox\\TU_Delft\\.Final_Deliverables\\4Github\\FUNCTIONS\\")
os.chdir("C:\\Users\\Sofia Gil Clavel\\Documents\\GitHub\\NLP4LitRev\\MainFunctions\\")

import Functions as FN
import DataViz as DV
import FindTextPatterns as PTN

# =============================================================================
# Openning the data and keeping only SLR and flood related articles
# =============================================================================

## It used to be DT1.pickle
with open('C:\\Users\\Sofia Gil Clavel\\Dropbox\\TU_Delft\\Projects\\DataBase\\PROCESSED\\df2.pickle', 'rb') as handle:
    DT1 = pickle.load(handle)

DT1["text2"]=DT1.apply(lambda x: x["dc:title"]+"\n"+x["description"],axis=1)

DT1=DT1[DT1.text2.apply(lambda x: len(re.findall("flood[a-z]*",x))>0 or 
                  len(re.findall("sea(|-|\s)level rise",x))>0 or 
                  len(re.findall("coast[a-z]*",x))>0 or 
                  len(re.findall("heavy precipitation(s|)",x))>0 or
                  len(re.findall("small island(s|)",x))>0)].copy()

DT1=DT1.reset_index(drop=True)

DT1["text2"]=DT1.apply(lambda x: x["description"]+"\n"+x.analysis+"\n"+x.results+"\n"+
                       x.findings+"\n"+x.conclusions+"\n"+x.discussion,axis=1)

### Adding the labeled Actors 
ACTORS=pd.read_csv("C:\\Users\\Sofia Gil Clavel\\Dropbox\\TU_Delft\\Projects\\Floods_CCA\\PROCESSED\\FloodArticlesActors.csv")

DT1=DT1.merge(ACTORS[['dc:identifier','accept']], 
                    on='dc:identifier', how = 'left')

#### Further cleaning the text ####
DIR="C:\\Users\\Sofia Gil Clavel\\Documents\\GitHub\\NLP4LitRev\MainFunctions\\"
DIR_main="C:\\Users\\Sofia Gil Clavel\\Documents\\GitHub\\Database_CCA\\MainFunctions\\"
DT1["text_clean"]=DT1["text2"].progress_apply(lambda x: FN.CleanText(x,DIR,DIR_main))

# =============================================================================
# Openning the models
# =============================================================================
nlp0 = spacy.load("C:\\Users\\Sofia Gil Clavel\\Dropbox\\TU_Delft\\.Final_Deliverables\\DATA\\3_PROCESSED_DATA\\model\\model-best\\")
nlp1 = spacy.load("C:\\Users\\Sofia Gil Clavel\\Dropbox\\TU_Delft\\.Final_Deliverables\\MODELS\\model-best\\") # This model is better at finding the nouns. It is en_core_sci_lg updated
nlp2 = spacy.load('en_core_web_trf') # This model is better to find markers

### Detect all sentences
DT1["sentence"] = DT1["text_clean"].progress_apply(lambda x: list(map(nltk.sent_tokenize,[x]))[0])

# =============================================================================
# Extract (Subject, Verb, Object) from the findings
# =============================================================================
VERBS_dict=PTN.SignDict("C:\\Users\\Sofia Gil Clavel\\Documents\\GitHub\\Database_CCA\\DATA\\Verbs.csv")

### Split sentences into (Subject, Verb, Object)
DT1["SVAOS"]=DT1.sentence.progress_apply(lambda x: [PTN.FindAllSents(ii, nlp1, nlp2, VERBS_dict) 
                                                    for ii in x if PTN.Finding(ii,nlp0)])
### Flatten list of lists
DT1["SVAOS"]=DT1["SVAOS"].progress_apply(lambda x: [item for sublist in x for item in sublist])

### Keeping only articles that have some (Subject, Verb, Object) sentences
DT1=DT1[DT1["SVAOS"].apply(lambda x: len(x)!=0)]
DT1=DT1.reset_index(drop=True)

DT1[DT1.apply(lambda x: any(x[0].find("gender")>-1 for x in x.SVAOS),axis=1)].index
DT1[DT1.apply(lambda x: any(x[2].find("buy - out")>-1 or x[2].find("buyout")>-1 or 
                            x[2].find("buy out")>-1 for x in x.SVAOS),axis=1)].index

# =============================================================================
# Replacing Subjects and Objects with Umbrella Categories
# =============================================================================

# ## Floods adaption strategies
# ADAPT=pd.read_csv("C:/Dropbox/TU_Delft/Projects/Floods_CCA/DATA/FloodCCA_Driver_pyDict_TW.csv",sep = ",")
# ADAPT=ADAPT.rename(columns={"Adaptation_Name":"Adaptation_Name","Adaptation_REGEX":"Adaptation_REGEX"})

## Floods adaption strategies
ADAPT=pd.read_csv("C:\\Users\\Sofia Gil Clavel\\Documents\\GitHub\\FloodSLR_CCA\\DATA\\FloodCCA_Driver_pyDict_TWTF.csv",sep = ",")
ADAPT=ADAPT.rename(columns={"Adaptation_Name":"Adaptation_Name","Adaptation_REGEX":"Adaptation_REGEX"})

## Farmers' factors for FACTation
FACT=pd.read_csv("C:\\Users\\Sofia Gil Clavel\\Documents\\GitHub\\FloodSLR_CCA\\PROCESSED\\Factors_Dictionary_python_SGCTF.csv",sep = ",")
FACT=FACT.rename(columns={"Tatiana_Factor_Strategy":"Factor_Name","Factor_Strategy_RGX":"Factor_REGEX"})

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

DT1["ADAPT"]=DT1.SVAOS.progress_apply(lambda x: FN.LabelMeasures(x,ADAPT,FACT))

# To include Acronym at the beginning of the measure
# DT1["ADAPT_TF"]=DT1.SVAOS.progress_apply(lambda x: FN.LabelMeasures(x,ADAPT_TF,FACT))

# with open('C:\\Dropbox\\TU_Delft\\Projects\\Floods_CCA\\PROCESSED\\DT1_20240814.pickle', 'wb') as handle:
#     pickle.dump(DT1, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('C:\\Dropbox\\TU_Delft\\Projects\\Floods_CCA\\PROCESSED\\DT1_20240814.pickle', 'rb') as handle:
#     DT1 = pickle.load(handle)

# =============================================================================
# Keeping only those sentences that talk about CCA measures and/or factors
# =============================================================================
key="ADAPT"
DT1_FACTORS=DT1.explode(key)
DT1_FACTORS[["NOUNA","SIGN","NOUNB"]]=pd.DataFrame(DT1_FACTORS[key].tolist(), index= DT1_FACTORS.index)
DT1_FACTORS[["source","source_type"]]=pd.DataFrame(DT1_FACTORS.NOUNA.tolist(), index= DT1_FACTORS.index)
DT1_FACTORS[["target","target_type"]]=pd.DataFrame(DT1_FACTORS.NOUNB.tolist(), index= DT1_FACTORS.index)
DT1_FACTORS=DT1_FACTORS[DT1_FACTORS.apply(lambda x: x.source_type!="" and x.target_type!="",axis=1)]
DT1_FACTORS=DT1_FACTORS.reset_index(drop=True)

TOBOLD=Counter(DT1_FACTORS.NOUNA.apply(lambda x: x[0] if x[1]=="ADAPT" else ""))+Counter(DT1_FACTORS.NOUNB.apply(lambda x: x[0] if x[1]=="ADAPT" else ""))
TOBOLD.most_common()

TOBOLD=Counter(DT1_FACTORS.NOUNA.apply(lambda x: x[0] if x[1]=="FACT" else ""))+Counter(DT1_FACTORS.NOUNB.apply(lambda x: x[0] if x[1]=="FACT" else ""))
TOBOLD.most_common()

# with open('C:\\Dropbox\\TU_Delft\\Projects\\Floods_CCA\\PROCESSED\\DT1_FACTORS_20240814.pickle', 'wb') as handle:
#     pickle.dump(DT1_FACTORS, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('C:\\Dropbox\\TU_Delft\\Projects\\Floods_CCA\\PROCESSED\\DT1_FACTORS_20240814.pickle', 'rb') as handle:
#     DT1_FACTORS = pickle.load(handle)

# =============================================================================
# How to extract meaningful relations?
# Check if sentences are the same based on permutations 
# =============================================================================

''' Using the categories''' #### Move Here N2

def PlotGraphNet(DB,TOBOLD,FONT=None,COLORStat=None,COLORS=None,delta_edge=1,
                 resolution=1,N_M=None,edge_sigmoidA=False,edge_sigmoid=False,
                 q=0.75,k=100,L=10):
    # Creating Database with frequencies
    FACTORS=DB.copy()
    
    FACTORS["source1"]=FACTORS.source.apply(lambda x: x.strip())
    FACTORS["target1"]=FACTORS.target.apply(lambda x: x.strip())
    
    FACTORS["source1"]=FACTORS.source1.map(lambda x: DV.AddBreakLine(x,n=1,breakAfter=4))
    FACTORS["target1"]=FACTORS.target1.map(lambda x: DV.AddBreakLine(x,n=1,breakAfter=4))
    
    FACTORS["VALUE"]=1.0
    FACTORS=FACTORS[FACTORS.apply(lambda x: x.source1!=x.target1,axis=1)]
    FACTORS=FACTORS.reset_index(drop=True)
    N_T=len(np.unique(FACTORS.ID_ART))
    N_ID_ART=Counter(FACTORS["ID_ART"])
    FACTORS['N_ID_ART']=[N_ID_ART[x] for x in FACTORS['ID_ART']]
    # Proportion is relative to the ID_ART
    FACTORS=FACTORS.groupby(['ID_ART','source1', 'target1','SIGN','N_ID_ART'], #,'N_T'
                            as_index=False)["VALUE"].sum()
    FACTORS=FACTORS.assign(VALUE= lambda x: x.VALUE/x.N_ID_ART,ONE=1.0)
    FACTORS=FACTORS.drop(columns=['ID_ART','N_ID_ART']) #,'N_T'
    FACTORS=FACTORS.groupby(['source1', 'target1','SIGN'],as_index=False).sum()
    FACTORS=FACTORS.pivot_table(index=['source1', 'target1',"ONE"], #
                                columns='SIGN',values='VALUE',fill_value=0)
    FACTORS=FACTORS.rename_axis(None, axis=1).reset_index()
    FACTORS=FACTORS.groupby(['source1', 'target1'],as_index=False).sum()
    SIGNS=[x for x in FACTORS.columns if x not in ["source1","target1","ONE"]] #
    FACTORS["SIGN"]=FACTORS.apply(lambda x: SIGNS[np.argmax(x[SIGNS])],axis=1)
    FACTORS["VALUE"]=FACTORS.apply(lambda x: x[x.SIGN]*x.ONE,axis=1)
    if edge_sigmoidA:
        FACTORS["VALUE2"]=DV.QuantilesWeight_upQ(list(FACTORS["VALUE"]),q=q)
        FACTORS=FACTORS[FACTORS.VALUE2.apply(lambda x: x>0)]
        FACTORS=FACTORS.reset_index(drop=True)
    FACTORS=FACTORS.rename(columns={"VALUE":"edge_weight",
                                    "ONE":"degree",
                                    "source1":"source",
                                    "target1":"target"})
    
    FACTORS=FACTORS[FACTORS.apply(lambda x: x.source!=x.target, axis=1)]
    FACTORS=FACTORS[FACTORS.apply(lambda x: x.source!="" and x.target!="",axis=1)]
    FACTORS=FACTORS.reset_index(drop=True)
    G,pos=DV.PosModularity(FACTORS,dw=100,delta=50,weight="edge_weight",
                           delta_edge=delta_edge,resolution=resolution)
    DV.plotNet_ChooseColors(G,pos,
            fontsize=FONT, #{k:0.3*v for k,v in dict(G.degree).items()},
            VERTEX=True,MODULARITY=True,
            min_target_margin=5,loc_legend='lower center',title_edge="Association Type",
            TOBOLD=TOBOLD,N_M=N_M,
            colors_node=COLORStat,colors_node_leg={v:k for k,v in COLORS.items()},
            # edge wright transformation
            edge_sigmoid=edge_sigmoid,q=q,L=L)

#### Creating Networks by Actor #### 

#### Classifying Using their Text ####
## Training Model using sklearn
# Useful guide: https://eli5.readthedocs.io/en/latest/tutorials/sklearn-text.html
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn import metrics
from sklearn.model_selection import cross_val_score

DT1_check=DT1[['doi',"dc:title","description",'dc:identifier','accept','analysis','results',
               'findings','conclusions','discussion']].copy()
DT1_check=DT1_check.drop_duplicates()
DT1_check=DT1_check.reset_index(drop=True)

DT1_check["text2"]=DT1_check.apply(lambda x: x["dc:title"]+" "+x.description+" "+ 
                                   x.analysis+" "+x.results+" "+x.findings+" "+
                                   x.conclusions+" "+x.discussion,axis=1)
DT1_check=DT1_check.drop_duplicates()
DT1_check=DT1_check.reset_index(drop=True)
DT1_check['IIHH']=DT1_check.apply(lambda x: 1 if not pd.isna(x.accept) and x.accept.find("IIHH")>-1 else 0,axis=1)
DT1_check['GOV']=DT1_check.apply(lambda x: 1 if not pd.isna(x.accept) and x.accept.find("GOV")>-1 else 0,axis=1)
DT1_check['CC']=DT1_check.apply(lambda x: 1 if not pd.isna(x.accept) and x.accept.find("CC")>-1 else 0,axis=1)
DT1_check['OTHER']=DT1_check.apply(lambda x: 1 if not pd.isna(x.accept) and x.accept.find("OTHER")>-1 and (x.IIHH+x.GOV+x.CC==0) else 0,axis=1)

TRAIN=DT1_check[DT1_check.apply(lambda x: not pd.isna(x.accept) and x.OTHER==0, axis=1)]
TRAIN=TRAIN.reset_index(drop=True)

TEST=DT1_check[DT1_check.apply(lambda x: pd.isna(x.accept) or x.OTHER==1, axis=1)]
TEST=TEST.reset_index(drop=True)
TEST['IIHH']=0
TEST['GOV']=0
TEST['CC']=0

for cc in ['IIHH','GOV','CC']:
    # A basic text processing pipeline - bag of words features and Logistic Regression as a classifier:
    model_tit=Pipeline([('tfidf', TfidfVectorizer()), ("logistic", LogisticRegressionCV(max_iter=1000))])
    model_tit.fit(list(TRAIN['text2']),TRAIN[cc])
    ## Applying it to new database
    TEST[cc]=list(model_tit.predict(TEST["text2"]))

DT1_check=pd.concat([TRAIN,TEST],ignore_index=True)
## Filling out those that could not be classified
DT1_check["OTHER"]=DT1_check.apply(lambda x: 1 if x.IIHH+x.GOV+x.CC==0 else 0, axis=1)
DT1_check["IIHH"]=DT1_check.apply(lambda x: 1 if x.OTHER==1 else x.IIHH, axis=1)
DT1_check["GOV"]=DT1_check.apply(lambda x: 1 if x.OTHER==1  else x.GOV, axis=1)
DT1_check["CC"]=DT1_check.apply(lambda x: 1 if x.OTHER==1  else x.CC, axis=1)

DT1_FACTORS2=DT1_FACTORS.merge(DT1_check[['doi','accept','IIHH','GOV','CC']],on="doi")

### Constraints as Colors Dictionary
# Colors
import distinctipy as distinctipy
import random

random.seed(469)

Constraints=FACT[['Constraint_Type','Factor_Name']]
Constraints['Factor_Name']=\
Constraints['Factor_Name'].apply(lambda x: x.strip())

COLORS_k=np.unique(Constraints['Constraint_Type'])
# COLORS_v=distinctipy.distinctipy.get_colors(len(COLORS_k), pastel_factor=0.7) # , pastel_factor=0.7
# Tatiana's colors:
# https://colorkit.co/palette/feffe0-ffd932-ba1d27-972367-58801d-b6e0f5-76a4f8-36332f/
COLORS_v=[(0.95, 0.71, 0.75),(1, 0.85, 0.20),(0.73, 0.11, 0.15),(0.59, 0.14, 0.4),(0.71, 0.88, 0.96),
          (0.35, 0.5, 0.11),(0.04, 0.32, 0.15),(0.37, 0.27, 0.54),(0.46, 0.64, 0.97)]
distinctipy.color_swatch(COLORS_v)
COLORS=dict(zip(COLORS_k,COLORS_v))

Constraints["COLORS"]=Constraints.apply(lambda x: COLORS[x.Constraint_Type],axis=1)
Constraints.index=Constraints.Factor_Name.apply(lambda x: 
                                        DV.AddBreakLine(x,n=1,breakAfter=4))
Constraints=Constraints.drop_duplicates()
COLORStat=dict(zip(Constraints.index,Constraints["COLORS"]))
COLORStat.pop('TCCA-specific')

DT1_FACTORS2=DT1_FACTORS2[DT1_FACTORS2.apply(lambda x: 
                    x.source!='TCCA-specific ' and x.target!='TCCA-specific ',axis=1)]
DT1_FACTORS2=DT1_FACTORS2.reset_index(drop=True)

### Words to Bold:
TOBOLD=Counter(DT1_FACTORS.apply(lambda x: x.source if x.source_type=="ADAPT" else "",axis=1))+\
        Counter(DT1_FACTORS.apply(lambda x: x.target if x.target_type=="ADAPT" else "",axis=1))
TOBOLD.most_common()
TOBOLD2=[DV.AddBreakLine(x.strip(),n=1,breakAfter=4)\
         for x in TOBOLD.keys()]
# ii=[c for c,x in enumerate(TOBOLD2) if x=='(managed)\n retreat'][0]
# TOBOLD2[ii]='managed\n retreat'

#### Labels to remove

REMOVE=['farming experience','tcca-specific','farm characteristics','farm size']+\
        ['(G) Climate Change Adaptation','(G) Climate Change Maladaptation',
          '(G) Climate Change Transformational Adaptation','(G) political involvement']

### Individuals and Households
DB=DT1_FACTORS2.copy()
DB=DB[DB.apply(lambda x: x.source not in REMOVE and 
               x.target not in REMOVE,axis=1)]
DB=DB[DB.apply(lambda x: x.IIHH==1,axis=1)]
DB=DB.reset_index(drop=True)
DB=DB[['dc:identifier','source', 'target','SIGN']]
DB=DB.rename(columns={'dc:identifier':'ID_ART'})
PlotGraphNet(DB,TOBOLD2,FONT=12,COLORStat=COLORStat,COLORS=COLORS,
             delta_edge=5,edge_sigmoidA=True,q=0.75,L=5,N_M=None)
# plt.savefig("C:\\Dropbox\\TU_Delft\\Projects\\Floods_CCA\\IMAGES\\NetCat_IIHH2.png",
#             dpi=300, bbox_inches='tight')

### Government
DB=DT1_FACTORS2.copy()
DB=DB[DB.apply(lambda x: x.source not in REMOVE and 
               x.target not in REMOVE,axis=1)]
DB=DB[DB.apply(lambda x: x.GOV==1,axis=1)]
DB=DB.reset_index(drop=True)
DB=DB[['dc:identifier','source', 'target','SIGN']]
DB=DB.rename(columns={'dc:identifier':'ID_ART'})
# TOBOLD2=[DV.AddBreakLine(x,n=1,breakAfter=3) for x in TOBOLD.keys()]
PlotGraphNet(DB,TOBOLD2,FONT=12,COLORStat=COLORStat,COLORS=COLORS,
             delta_edge=5,edge_sigmoidA=True,q=0.75,L=2.5,N_M=[0,1,2])
# plt.savefig("C:\\Dropbox\\TU_Delft\\Projects\\Floods_CCA\\IMAGES\\NetCat_GOV2.png",
#             dpi=300, bbox_inches='tight') #

### Communities
DB=DT1_FACTORS2.copy()
DB=DB[DB.apply(lambda x: x.source not in REMOVE and 
               x.target not in REMOVE,axis=1)]
DB=DB[DB.apply(lambda x: x.CC==1,axis=1)]
DB=DB.reset_index(drop=True)
DB=DB[['dc:identifier','source', 'target','SIGN']]
DB=DB.rename(columns={'dc:identifier':'ID_ART'})
# TOBOLD2=[DV.AddBreakLine(x,n=1,breakAfter=3) for x in TOBOLD.keys()]
PlotGraphNet(DB,TOBOLD2,FONT=14,COLORStat=COLORStat,COLORS=COLORS,
             delta_edge=5,edge_sigmoidA=True,q=0.75,L=2.5)
# plt.savefig("C:\\Dropbox\\TU_Delft\\Projects\\Floods_CCA\\IMAGES\\NetCat_CC2.png",
#             dpi=300) #, bbox_inches='tight'

### Other
DB=DT1_FACTORS2.copy()
DB=DB[DB.apply(lambda x: x.source not in REMOVE and 
               x.target not in REMOVE,axis=1)]
DB=DB[DB.apply(lambda x: x.IIHH+x.GOV+x.CC==0,axis=1)]
DB=DB.reset_index(drop=True)
DB=DB[['dc:identifier','source', 'target','SIGN']]
DB=DB.rename(columns={'dc:identifier':'ID_ART'})
# TOBOLD2=[DV.AddBreakLine(x,n=1,breakAfter=3) for x in TOBOLD.keys()]
PlotGraphNet(DB,TOBOLD2,FONT=14,COLORStat=COLORStat,COLORS=COLORS)
# plt.savefig("C:\\Dropbox\\TU_Delft\\Projects\\Floods_CCA\\IMAGES\\NetCat_OTHER.png",
#             dpi=300, bbox_inches='tight') #



#### Figure All: No measure by actor
wACTORS=TOBOLD.copy()

DB=DT1_FACTORS2.copy()
## Keeping only ADAPT-FACT and FACT-ADAPT connection
# DB=DB[DB.apply(lambda x: x.source_type!=x.target_type, axis=1)]
# DB=DB.reset_index(drop=True)
DB=DB[DB.apply(lambda x: x.IIHH+x.GOV+x.CC>0,axis=1)]
DB=DB[DB.apply(lambda x: x.source not in REMOVE and x.target not in REMOVE,axis=1)]
DB=DB.reset_index(drop=True)
DB["source"]=DB.apply(lambda x: "IHH:"+x.source if x.IIHH==1 and x.source not in wACTORS else
             "COM:"+x.source if x.CC==1 and x.source not in wACTORS  else
             "GOV:"+x.source if x.GOV==1 and x.source not in wACTORS  else x.source,
             axis=1)
DB["target"]=DB.apply(lambda x: "IHH:"+x.target if x.IIHH==1 and x.target not in wACTORS else
             "COM:"+x.target if x.CC==1 and x.target not in wACTORS  else
             "GOV:"+x.target if x.GOV==1 and x.target not in wACTORS  else x.target,
             axis=1)
DB=DB[['dc:identifier','source', 'target','SIGN']]
DB=DB.rename(columns={'dc:identifier':'ID_ART'})
# Adding the Actor to the colors
COLORStatwACTOR={}
for aa in ["IHH:","COM:","GOV:"]:
    for k,v in COLORStat.items():
        COLORStatwACTOR[aa+k]=v

resolution=0.85

PlotGraphNet(DB,TOBOLD2,FONT=8,COLORStat=COLORStatwACTOR,COLORS=COLORS,resolution=resolution,
             N_M=None,delta_edge=5,edge_sigmoidA=True,edge_sigmoid=False,q=0.5)
plt.savefig("C:\\Users\\Sofia Gil Clavel\\Pictures\\NetCat_ACTORS.png",
            dpi=600, bbox_inches='tight')

PlotGraphNet(DB,TOBOLD2,FONT=12,COLORStat=COLORStatwACTOR,COLORS=COLORS,resolution=resolution,
             N_M=[0],delta_edge=5,edge_sigmoidA=True,edge_sigmoid=False,q=0.5)
plt.savefig("C:\\Users\\Sofia Gil Clavel\\Pictures\\NetCat_ACTORS_0.png",
            dpi=300, bbox_inches='tight')

PlotGraphNet(DB,TOBOLD2,FONT=12,COLORStat=COLORStatwACTOR,COLORS=COLORS,resolution=resolution,
             N_M=[1],delta_edge=5,edge_sigmoidA=True,edge_sigmoid=False,q=0.5)
plt.savefig("C:\\Users\\Sofia Gil Clavel\\Pictures\\NetCat_ACTORS_1.png",
            dpi=300, bbox_inches='tight')

PlotGraphNet(DB,TOBOLD2,FONT=12,COLORStat=COLORStatwACTOR,COLORS=COLORS,resolution=resolution,
             N_M=[2],delta_edge=5,edge_sigmoidA=True,edge_sigmoid=False,q=0.5)
plt.savefig("C:\\Users\\Sofia Gil Clavel\\Pictures\\NetCat_ACTORS_2.png",
            dpi=300, bbox_inches='tight')

#### Figure Simple Example
W="(R) migration"
DB2=DB[DB.apply(lambda x: x.source==W or x.target==W,axis=1)]
DB2=DB2.reset_index(drop=True)
DB2=DB2[['ID_ART','source', 'target','SIGN']]
# TOBOLD2=[DV.AdDB2reakLine(x,n=1,breakAfter=3) for x in TOBOLD.keys()]
PlotGraphNet(DB2,TOBOLD2,FONT=14,COLORStat=COLORStatwACTOR,COLORS=COLORS,resolution=0.5,
             N_M=None,delta_edge=5,edge_sigmoidA=False,edge_sigmoid=False,q=0,L=5)
plt.savefig("C:\\Users\\Sofia Gil Clavel\\Pictures\\NetCat_Migration.png",
            dpi=300, bbox_inches='tight') # , bbox_inches='tight'

W=["(P) dykes"]
DB2=DB[DB.apply(lambda x: (x.source in W) or (x.target in W),axis=1)]
DB2=DB2.reset_index(drop=True)
DB2=DB2[['ID_ART','source', 'target','SIGN']]
# TOBOLD2=[DV.AdDB2reakLine(x,n=1,breakAfter=3) for x in TOBOLD.keys()]
PlotGraphNet(DB2,TOBOLD2,FONT=14,COLORStat=COLORStatwACTOR,COLORS=COLORS,resolution=0.5,
             N_M=None,delta_edge=5,edge_sigmoidA=False,edge_sigmoid=False,q=0,L=5)
plt.savefig(f"C:\\Users\\Sofia Gil Clavel\\Pictures\\NetCat_dykes.png",
            dpi=300, bbox_inches='tight') # , bbox_inches='tight'

### Only ADAPT-FACT relations
wACTORS=TOBOLD.copy()

DB=DT1_FACTORS2.copy()
## Keeping only ADAPT-FACT and FACT-ADAPT connection
DB=DB[DB.apply(lambda x: x.source_type!=x.target_type, axis=1)]
DB=DB.reset_index(drop=True)
DB=DB[DB.apply(lambda x: x.IIHH+x.GOV+x.CC>0,axis=1)]
DB=DB[DB.apply(lambda x: x.source not in REMOVE and x.target not in REMOVE,axis=1)]
DB=DB.reset_index(drop=True)
DB["source"]=DB.apply(lambda x: "IHH:"+x.source if x.IIHH==1 and x.source not in wACTORS else
             "COM:"+x.source if x.CC==1 and x.source not in wACTORS  else
             "GOV:"+x.source if x.GOV==1 and x.source not in wACTORS  else x.source,
             axis=1)
DB["target"]=DB.apply(lambda x: "IHH:"+x.target if x.IIHH==1 and x.target not in wACTORS else
             "COM:"+x.target if x.CC==1 and x.target not in wACTORS  else
             "GOV:"+x.target if x.GOV==1 and x.target not in wACTORS  else x.target,
             axis=1)
DB=DB[['dc:identifier','source', 'target','SIGN']]
DB=DB.rename(columns={'dc:identifier':'ID_ART'})
# Adding the Actor to the colors
COLORStatwACTOR={}
for aa in ["IHH:","COM:","GOV:"]:
    for k,v in COLORStat.items():
        COLORStatwACTOR[aa+k]=v

resolution=1

PlotGraphNet(DB,TOBOLD2,FONT=8,COLORStat=COLORStatwACTOR,COLORS=COLORS,resolution=resolution,
             N_M=list(range(6)),delta_edge=5,edge_sigmoidA=True,edge_sigmoid=False,q=0.5)
plt.savefig("C:\\Users\\Sofia Gil Clavel\\Pictures\\NetCat_ACTORS_FACT_ADAPT.png",
            dpi=600, bbox_inches='tight')

# with open('C:\\Dropbox\\TU_Delft\\Projects\\Floods_CCA\\PROCESSED\\DT1_FACTORS2wActors_20240814.pickle', 'wb') as handle:
#     pickle.dump(DT1_FACTORS2, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('C:\\Dropbox\\TU_Delft\\Projects\\Floods_CCA\\PROCESSED\\DT1_FACTORS2wActors_20240814.pickle', 'rb') as handle:
#     DT1_FACTORS2 = pickle.load(handle)

DT1_FACTORS2.to_csv('C:\\Dropbox\\TU_Delft\\Projects\\Floods_CCA\\PROCESSED\\DT1_FACTORS2wActors_20240814.csv')

# with open('C:\\Dropbox\\TU_Delft\\Projects\\Floods_CCA\\PROCESSED\\DT1wActors_20240814.pickle', 'wb') as handle:
#     pickle.dump(DT1_check, handle, protocol=pickle.HIGHEST_PROTOCOL)

DT1_check.to_csv("C:/Dropbox/TU_Delft/Projects/Floods_CCA/PROCESSED/DT1wActors.csv")

# with open('C:\\Dropbox\\TU_Delft\\Projects\\Floods_CCA\\PROCESSED\\DT1wActors_20240814.pickle', 'rb') as handle:
#     DT1 = pickle.load(handle)

DT1[['affiliation_Country','StudiedPlace', 'affiliation_ISO2',
'affiliation_Continent', 'StudiedPlace_ISO2', 'StudiedPlace_Continent']].to_csv('C:/Dropbox/TU_Delft/Projects/Floods_CCA/PROCESSED/DT1wActors.csv')


### Copying articles for zotero
import shutil

### Data < August 2022
DIR1="C:\\Dropbox\\TU_Delft\\Projects\\ML_FindingsGrammar\\DATA\\PDFs_Clusters\\" # !!!
### Data August 2022 - January 2024
DIR2="C:\\Dropbox\\TU_Delft\\Projects\\DataBase\\PROCESSED\\SCOPUS_DATA\\pdfs\\" # !!!

COPY="C:\\Dropbox\\Articles_24072019\\.Delft\\Floods_CCA\\SpecialIssue2024\\"

for ii in range(DT1.shape[0]):
    if not pd.isna(DT1['clusters2'].iloc[ii]):
        DIR11=DIR1+"Cluster_"+str(DT1['clusters2'].iloc[ii])+"\\"
        shutil.copyfile(DIR11+DT1['FILE_NAME'].iloc[ii]+".pdf", COPY+DT1['FILE_NAME'].iloc[ii]+".pdf")
    else:
        shutil.copyfile(DIR2+DT1['FILE_NAME'].iloc[ii]+".pdf", COPY+DT1['FILE_NAME'].iloc[ii]+".pdf")








