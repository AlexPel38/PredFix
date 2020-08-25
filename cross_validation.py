# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:27:07 2020

@author: Pelle
"""


# Package de base
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl   
from sklearn import metrics


# Méthode de Cox & courbe de survie
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold 




cancer_type = 'TCGA-BRCA'
random_state = 0

survival = 'dfs'
survivalThreshold = 36.0


# ---- Données ----
dirname = 'G:/Stage 2020/donnees/'

#TCGA-BRCA
data_chemin = 'expression_data_tcga_brca_TCGA-BRCA_log_fpkm_1144_samples_37898_genes.csv'
expgroup = 'G:/Stage 2020/donnees/EpiMed_experimental_grouping_2020.07.13_TCGA-BRCA.xlsx'
#Param réseau de neuronnes
hidden_layer_size = (100,100,100,100)
alpha = 80


# ===================== importation des données =====================
print("transcriptome : " , dirname+data_chemin)
data = pd.read_csv(dirname+data_chemin,sep=";",header=0,decimal =".")
# Suppression d'une colonne identifiant
data=data.drop(["id_gene"],axis='columns')
# Convertir en dataframe
data_df = pd.DataFrame(data, columns = data.columns,index=data.index)
print("Le tableau des données transcriptomes est : \n",data_df)

# ---- Transposition de la matrice de données transcriptomique 
data_df_t = data_df.T
print("le tableau de données transposé est : ", data_df_t)

# Définission des noms des colonnes
data_df_t.columns = data_df_t.iloc[0,:]
# Suppréssion de la colonne en double (qui est devenu l'index)
data_transc_df_t=data_df_t.drop(["gene_symbol"])

## ---------- Données cliniques
data_clinique = pd.read_excel(expgroup,sep=",",header=0,decimal =".")
# convertir en dataframe
data_clinique_df = pd.DataFrame(data_clinique, columns = data_clinique.columns)

print("Le tableau des données cliniques est :", data_clinique_df)

#  ===================== FIN importation des données =====================

# ==== Lien entre les données transcriptomiques et cliniques pour récuparation des colonnnes ===
target = data_transc_df_t.index
y = []
months = []
censor = []
main_gse = []
for i in range(len(data_transc_df_t)) : 
    for j in range(len(data_clinique_df)) :
        if (target[i] in data_clinique_df['id_sample'][j]):
            if True : 
                y.append(data_clinique_df['tissue_status'][j])
                months.append(data_clinique_df[survival+'_months'][j])
                censor.append(data_clinique_df[survival+"_censor"][j])
                main_gse.append(data_clinique_df["main_gse_number"][j])
            else :
                break

len(y)
len(months)
len(censor)
len(main_gse)

data_transc_df_t['Target']=y
data_transc_df_t[survival+'_months']=months
data_transc_df_t[survival+'_censor'] = censor
data_transc_df_t["main_gse"] = main_gse


#On conserve les échantillons tumoraux et de type LUAD
data_transc_df_t = data_transc_df_t[data_transc_df_t.Target == "tumoral"]
data_transc_df_t = data_transc_df_t[data_transc_df_t.main_gse == cancer_type]



# ---------------- Création étiquette pronostique sur le df construit précédemment
etiquette = []
for i in range(len(data_transc_df_t)):
    if (data_transc_df_t[survival+"_months"].iloc[i]>= survivalThreshold) & (data_transc_df_t[survival+"_censor"].iloc[i] == 1):
        etiquette.append("good")
    elif (data_transc_df_t[survival+"_months"].iloc[i] >=survivalThreshold) & (data_transc_df_t[survival+"_censor"].iloc[i] == 0):
        etiquette.append("good")
    elif (data_transc_df_t[survival+"_months"].iloc[i] < survivalThreshold) & (data_transc_df_t[survival+"_censor"].iloc[i]== 1): 
        etiquette.append("bad")
    elif (data_transc_df_t[survival+"_months"].iloc[i] < survivalThreshold) & (data_transc_df_t[survival+"_censor"].iloc[i] == 0): 
        etiquette.append("unknown")
        
        
print(str(len(etiquette)))
print("good : " + str(etiquette.count("good")))
print("bad : " + str(etiquette.count("bad")))
print("ukn : " + str(etiquette.count("unknown")))
data_transc_df_t["etiquette"] = etiquette

# On conserve uniquement les bons/mauvais pronostiques
data_transc_df_t = data_transc_df_t[data_transc_df_t.etiquette != "unknown"]

# ================================= DIVISION DU JEU DE DONNEES ============================================
y = data_transc_df_t.etiquette
df_fn = data_transc_df_t.drop(['Target', survival+"_months" ,survival+"_censor", "etiquette","main_gse"],axis='columns')
random_state = 0
X_train, X_test, y_train, y_test = train_test_split(df_fn, y, stratify=y, random_state=random_state,test_size = 100)

# ================================= DIAGRAMME CIRCULAIRE  ============================================
#Total
labels = ['bon pronostic' + ' : ' + str(np.sum(data_transc_df_t.etiquette ==('good'))) , 'Mauvais pronostic' + ' : ' + str(np.sum(data_transc_df_t.etiquette ==('bad')))]
sizes = [np.sum(data_transc_df_t.etiquette ==('good')),np.sum(data_transc_df_t.etiquette ==('bad'))]
colors = ['lightskyblue', 'lightcoral']

plt.pie(sizes, labels=labels, colors=colors, 
        autopct='%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.title("Répartition des pronostics de"+ " " +cancer_type)
plt.show()

# y_train
labels = ['bon pronostic' + ' : ' + str(np.sum(y_train == "good")) , 'Mauvais pronostic' + ' : ' + str(np.sum(y_train == "bad"))]
sizes = [np.sum(y_train == "good"),np.sum(y_train == "bad")]
colors = ['lightskyblue', 'lightcoral']

plt.pie(sizes, labels=labels, colors=colors, 
        autopct='%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.title("Répartition des pronostics de"+ " " +cancer_type+ " pour l'entraînement")
plt.show()

# y_test
labels = ['bon pronostic' + ' : ' + str(np.sum(y_test == "good")) , 'Mauvais pronostic' + ' : ' + str(np.sum(y_test == "bad"))]
sizes = [np.sum(y_test == "good"),np.sum(y_test == "bad")]
colors = ['lightskyblue', 'lightcoral']

plt.pie(sizes, labels=labels, colors=colors, 
        autopct='%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.title("Répartition des pronostics de"+ " " +cancer_type+ " pour le test")
plt.show()
# =============================================================================

# ================== CROSS VALIDATION ==========
def cross_val(X_train,y_train,hls,alpha):
    cv = KFold(n_splits=3, random_state=0)
    scaler = preprocessing.StandardScaler()
    cph = CoxPHFitter()
    
    for train_index, test_index in cv.split(X_train):
        # Construction des donnees
        X_train_nested, X_test_nested, y_train_nested, y_test_nested = X_train.iloc[train_index,:], X_train.iloc[test_index,:], y_train.iloc[train_index], y_train.iloc[test_index]
      
        # ========================================= Réduction de dimension ==========================================
        ##  =========================== Méthode 1 : Par la variance ==================================
        print("Méthode 1 : Réduction par la variance")
        # Calcul variance et Trie décroissant sur les gênes
        variance=X_train_nested.var(axis=0)
        #Récupération des 5000 premiers
        sort_var2 = variance.sort_values(ascending = False)[0:5000]
       
        #Index des 5000 premiers gênes sur le df contenant les gênes par échantillons   
        X_train_nested = X_train_nested[sort_var2.index]
        X_test_nested = X_test_nested[sort_var2.index]
    
            
        ## ============================= Méthode 2 : Modèle de COX ======================================
        print("Méthode 2 : Réduction par le modèle de Cox")
        ##  Modèle univarié : doit touner autant de fois qu'on à de gêne
        final_df2 = X_train_nested
        final_df2['T'] = data_transc_df_t[survival+"_months"][final_df2.index]
        final_df2['E'] = data_transc_df_t[survival+"_censor"][final_df2.index]
        
        # Gestion des valeurs manquantes (NAN remplacé par 0) ?à voir avec Katia
        final_df2 = final_df2.dropna(axis =1)
        
        n = final_df2.shape[1]
        p_valeur = []
        hazard_ratio = []

        for i in range(n-2):
            #Entrainement du modèle
            cph.fit(final_df2.iloc[:,[i,n-2,n-1]], 'T', event_col='E')
            #Récupération paramètres
            ## P_valeurs
            p_valeur.append(cph.summary.p[0])
            ## Hazard_ratio
            hazard_ratio.append(cph.summary['exp(coef)'][0])
        
        #construction d'un DF par gêne avec les pvaleurs et le hazard ratio
        ind = final_df2.columns
        ind = ind.drop('T')
        ind = ind.drop('E')
        col =['pvaleur' ,'hazard_ratio']
        df_cox = pd.DataFrame([p_valeur,hazard_ratio])
        df_cox =df_cox.T
        df_cox.index= ind
        df_cox.columns =col
        
        #Trie sur la pvaleur
        df_cox = df_cox.sort_values(by = 'pvaleur',ascending = True)[0:1000]
    
        #Df final avec les index des gênes les plus significatifs
        X_train_nested_reduced = X_train_nested[df_cox.index]
        X_test_nested_reduced = X_test_nested[df_cox.index]
        print("après cox : \n", X_train_nested_reduced)
        print("Shape : ", X_train_nested_reduced.shape)
        
        ## ============================ NORMALISATION ================================================
        print("Normalisation : ")
        np.seterr(divide='ignore', invalid='ignore')
        
        X_train_nested_reduced_scaler = scaler.fit_transform(X_train_nested_reduced)
        X_train_nested_reduced_scaler = pd.DataFrame(X_train_nested_reduced_scaler, index=X_train_nested_reduced.index, columns=X_train_nested_reduced.columns)
            
        X_test_nested_reduced_scaler = scaler.transform(X_test_nested_reduced)
        X_test_nested_reduced_scaler = pd.DataFrame(X_test_nested_reduced_scaler, index=X_test_nested_reduced.index, columns=X_test_nested_reduced.columns)
        
        X_train_nested_reduced_scaler = X_train_nested_reduced_scaler.dropna(axis = 1)
        X_test_nested_reduced_scaler = X_test_nested_reduced_scaler.dropna(axis = 1)
        
        print("Recherche sur grille")
        
        
        mlp = MLPClassifier(hidden_layer_sizes = hls ,alpha=alpha , random_state=random_state, early_stopping=False,
                          solver='sgd', learning_rate='adaptive', max_iter=1000, warm_start=False)
        
        X_train_nested_reduced_scaler = X_train_nested_reduced_scaler.dropna(axis =1)
        X_test_nested_reduced_scaler = X_test_nested_reduced_scaler.dropna(axis =1)
      
        accuracy_train = []
        accuracy_test = []
        mlp.fit(X_train_nested_reduced_scaler, y_train_nested)
        y_pred_train = mlp.predict(X_train_nested_reduced_scaler)
        y_pred_test = mlp.predict(X_test_nested_reduced_scaler)
        
        acc_train = metrics.accuracy_score(y_train_nested, y_pred_train)
        accuracy_train.append(acc_train)
        acc_test = metrics.accuracy_score(y_test_nested, y_pred_test)
        accuracy_test.append(acc_test)
        print("Accuracy train:", '{:.2f}'.format(acc_train))
        print("Accuracy test:", '{:.2f}'.format(acc_test))
    

cross_val(X_train,y_train,hidden_layer_size,alpha)
