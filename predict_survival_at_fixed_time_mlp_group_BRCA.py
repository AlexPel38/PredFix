# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 10:57:51 2020

@author: Pelle
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import matplotlib as mpl
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines import CoxPHFitter

from sklearn import metrics

mpl.rcParams['font.family'] = 'Arial'
arialNarrowFont = {'fontname':'Arial', 'stretch' : 'condensed'}

regular = 22
medium = 0.8 * regular
small = 0.7 * regular

def generateSignificanceSymbol(pvalue):
    if (pvalue<=0.01):
        return '***'
    if (pvalue<=0.03):
        return '**'
    if (pvalue<=0.05):
        return '*'
    return ''


def shiftToMaxSurvivalTime(survivalTime, survivalEvent, maxSurvivalTime):
    shiftedTime = survivalTime
    shiftedEvent = survivalEvent
    if (survivalTime>maxSurvivalTime):
        shiftedTime = maxSurvivalTime
        shiftedEvent = 0.0
    return shiftedTime, shiftedEvent

def getGroupQuantiles(result, value, quantiles):
    thresholds = [np.quantile(result['proba'], q) for q in quantiles]
    thresholds = sorted(thresholds)
    group = len(thresholds)
    vmin = 0.0
    for i, threshold in enumerate(thresholds):
        vmax = threshold
        if (value>=vmin and value<vmax):
            group=i
        vmin = vmax
    return group        

def createPrognosisGroup(survival, censor, survivalThreshold):
    group = np.nan
    if (survival is not None) and (np.invert(np.isnan(survival))):
        if (survival>=survivalThreshold):
            group = 0.0
        if (survival<survivalThreshold and censor==1.0):
            group = 1.0
    return group

covariate = 'proba_group'


quantiles = [0.5]
colors = ['royalblue', 'black']

n_features = 1000
random_state = 0

survival = 'dfs'
survivalThreshold = 36.0
targetName = 'prognosis_group'
maxSurvivalTime = 120

survivalName = {'os': 'Overall survival (OS)', 'dfs' : 'Event-free survival'} 
probaGroupName = {0.0 : 'good prognosis', 1.0 : 'bad prognosis'}


project = 'TCGA-BRCA'
data_filename = 'G:/Stage 2020/donnees/expression_data_tcga_brca_TCGA-BRCA_log_fpkm_1144_samples_37898_genes.csv'
expgroup_filename = 'G:/Stage 2020/donnees/EpiMed_experimental_grouping_2020.07.13_TCGA-BRCA.xlsx'

print('Importing targets from',expgroup_filename, '...')
expgroup = pd.read_excel(expgroup_filename,sep=",",header=0,decimal =".")
expgroup.index = expgroup['id_sample']

for key in expgroup.index:
    if (expgroup.loc[key, survival + '_months']!=None):
        expgroup.loc[key, targetName] = createPrognosisGroup(expgroup.loc[key, survival + '_months'], expgroup.loc[key, survival + '_censor'], survivalThreshold)
expgroup = expgroup.loc[np.invert(np.isnan(expgroup[targetName]))]


print('Importing data from', data_filename, '...')
X = pd.read_csv(data_filename, sep=';', decimal='.')
X.index = X['id_gene']
X = X.drop(columns=['id_gene', 'gene_symbol'])
X.columns.name = 'id_sample'

idSamples = list(set(X.columns).intersection(set(expgroup.index)))

expgroup = expgroup.loc[idSamples, :]
y = list(expgroup[targetName])
targetStats = expgroup.groupby([targetName]).size().reset_index(name='counts')

print('Expgroup', expgroup.shape)
print(expgroup.head(3))
print(targetStats)

minAccuracy = 1.0 - targetStats.loc[1, 'counts'] / targetStats.loc[0, 'counts']


X = X[idSamples]

# Transpose pandas dataframe to obtain a standard form: samples in rows and features in columns
X = X.T
X = X.dropna(axis=1)
X.index
X.columns
# Display data
print('Data', X.shape)
print(X.head(3))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state= random_state, stratify=y)

print('X_train', X_train.shape)
print('X_test', X_test.shape)

df_index = pd.DataFrame(index_test,columns = {'index'})

# Variable reduction

print('--- Variable reduction ---')

print('By mean')
means = X_train.mean(axis=0).sort_values(ascending=False)
means = means[means>1.0]
selected_features = list(means.index)
X_train = X_train[selected_features]
X_test = X_test[selected_features]

print('By variance')
variance = X_train.var(axis=0)
variance = variance.sort_values(ascending=False)
selected_features = list(variance.index)[0:5000]

X_train = X_train[selected_features]
X_test = X_test[selected_features]

print('By Cox model')
cph = CoxPHFitter()
final_df2 = X_train
final_df2['T'] = expgroup.dfs_months[X_train.index]
final_df2['E'] = expgroup.dfs_censor[X_train.index]
# Gestion des valeurs manquantes 
final_df2 = final_df2.dropna(axis =1)

n = final_df2.shape[1]
p_valeur = []
hazard_ratio = []

for i in range(n-2):
    #Entrainement du modèle
    cph.fit(final_df2.iloc[:,[i,n-2,n-1]], 'T', event_col='E')
    #Récupération paramètres
    ## P_valeurs
    p_valeur.append(cph.summary['p'].values[0])
    ## Hazard_ratio
    hazard_ratio.append(cph.summary['exp(coef)'].values[0])
 
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
selected_features = df_cox.sort_values(by = 'pvaleur',ascending = True)[0:n_features]
    
X_train = X_train[selected_features.index]
X_test = X_test[selected_features.index]

# Scaling

print('--- Scaling ---')

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)

X_test_scaled = scaler.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

print('X_train_scaled', X_train_scaled.shape)
print('X_test_scaled', X_test_scaled.shape)

print('--- Training a classifier ---')


hidden_layer_size  = (100, 100, 100, 100)
alpha = 80.0
classifier = MLPClassifier(random_state=random_state, hidden_layer_sizes= hidden_layer_size , alpha= alpha, early_stopping=False,
                          solver='sgd', learning_rate='adaptive', max_iter=1000, warm_start=False)


classifier.fit(X_train_scaled, y_train)
y_train_predicted = classifier.predict(X_train_scaled)
y_train_proba = classifier.predict_proba(X_train_scaled)[:, 1] 

y_test_predicted = classifier.predict(X_test_scaled)
y_test_proba = classifier.predict_proba(X_test_scaled)[:, 1] # proba bad prognosis

accuracy_train = metrics.accuracy_score(y_train, y_train_predicted)
accuracy_test = metrics.accuracy_score(y_test, y_test_predicted)

print("Accuracy train:", '{:.2f}'.format(accuracy_train), ', test:', '{:.2f}'.format(accuracy_test), 'min =', '{:.2f}'.format(minAccuracy), 'MLP =', hidden_layer_size, 'alpha =', alpha)

for datasetType in ['train', 'test']:

    result = pd.DataFrame()
    
    if (datasetType=='train'):
        result = pd.DataFrame(index=X_train_scaled.index)
        result['group'] = y_train_predicted
        result['proba'] = y_train_proba
        Title = "TCGA-BRCA train dataset (n=" + str(len(X_train)) +")"
    else:
        result = pd.DataFrame(index=X_test_scaled.index)
        result['group'] = y_test_predicted
        result['proba'] = y_test_proba
        Title = "TCGA-BRCA test dataset (n=" + str(len(X_test)) +")"

         
    result['time'] = expgroup.loc[result.index, survival + '_months']
    result['event'] = expgroup.loc[result.index, survival + '_censor']
    
    
    for i in result.index:
        shiftedTime, shiftedEvent = shiftToMaxSurvivalTime(result.loc[i, 'time'], result.loc[i, 'event'], maxSurvivalTime)
        result.loc[i, 'time'] = shiftedTime
        result.loc[i, 'event'] = shiftedEvent
        proba = result.loc[i, 'proba']
        result.loc[i, 'proba_group'] = getGroupQuantiles(result, proba, quantiles)
    
    print(result)
    print(result['proba'].min(), result['proba'].max(), result['proba'].mean())
    
    probaGroups = sorted(list(result['proba_group'].unique()))
    print('probaGroups', probaGroups)
    
    logrank = multivariate_logrank_test(result['time'], result['proba_group'], result['event'])  
    print(logrank.summary)
    
    data = pd.DataFrame(result)
    data = data[['time', 'event', covariate]]
    cph = CoxPHFitter()
    cph.fit(data, duration_col='time', event_col='event', show_progress=False)
    print(cph.summary)
    cox_pvalue = cph.summary.p[covariate]
    cox_hr = cph.summary['exp(coef)'][covariate]
    cindex = cph.concordance_index_ 
    print('p-value', cph.summary.p)   
    print('\n')
    
    fig = plt.figure(num=None, figsize=(6, 5), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
            
    kmf = KaplanMeierFitter()
    for i, probaGroup in enumerate(probaGroups):
        selection = result[result['proba_group']==probaGroup]
        nSamples = selection.shape[0]
        labelText = probaGroupName[probaGroup] + ' (n=' + str(nSamples) +')'
        kmf.fit(selection['time'], selection['event'], label=labelText)
        kmf.plot(ax=ax, ci_show=False, show_censors=True, color=colors[i], linewidth=3)
        print(probaGroup, 'Samples', nSamples, list(selection.index))
    
    logrankPvalueText = 'logrank p-value = ' + '{:.3f}'.format(logrank.p_value) + ' ' + generateSignificanceSymbol(logrank.p_value)
    if (logrank.p_value<0.001):
        logrankPvalueText = 'logrank p-value < 0.001' + ' ' + generateSignificanceSymbol(logrank.p_value)
        
    coxPvalueText = 'cox p-value = ' +  '{:.3f}'.format(cox_pvalue) + ' ' + generateSignificanceSymbol(cox_pvalue)
    if (cox_pvalue<0.001):
        coxPvalueText = 'cox p-value < 0.001' + ' ' + generateSignificanceSymbol(cox_pvalue)
    
    hrText = 'cox hazard ratio = ' + '{:.1f}'.format(cox_hr)
    cindexText = 'C-index = ' + '{:.2f}'.format(cindex)
    
    ax.set_title(Title + '\n' + logrankPvalueText + '\n' + coxPvalueText + '\n' +  hrText + '\n' + cindexText, fontsize=medium, **arialNarrowFont)  
    
    
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([0.0 - 0.05*maxSurvivalTime, maxSurvivalTime + 0.05*maxSurvivalTime])
    
    step = 10.0 * np.ceil(maxSurvivalTime/100.0)
    xticks = np.arange(0, maxSurvivalTime + step, step)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, **arialNarrowFont)
    ax.set_xlabel('Time in months', fontsize=regular, **arialNarrowFont)
    
    yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, **arialNarrowFont)
    ax.set_ylabel(survivalName[survival], fontsize=regular, **arialNarrowFont)
    ax.tick_params(axis='both', labelsize=medium)
    
    plt.show()
    plt.close(fig)
    
    