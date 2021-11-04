#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd
import statsmodels.api as sm 
import scipy.stats as stats


# In[2]:


crimeclean=pd.read_csv("censusCrimeClean.csv")
crimeclean


# In[3]:


medIncome=crimeclean["medIncome"]
ViolentCrimesPerPop=crimeclean["ViolentCrimesPerPop"]


# In[4]:


corrPearson,pValPearson = stats.pearsonr(medIncome,ViolentCrimesPerPop)
print(corrPearson,pValPearson)
corrSpearman,pValSpearman = stats.spearmanr(medIncome,ViolentCrimesPerPop)


# In[5]:


scatter=crimeclean.plot.scatter(x="medIncome",y="ViolentCrimesPerPop")
scatter


# In[6]:


population=crimeclean['racepctblack']
PctForeignBorn=crimeclean['PctForeignBorn']
corrPearson,pValPearson = stats.pearsonr(population,ViolentCrimesPerPop)
print(corrPearson,pValPearson)
corrSpearman,pValSpearman = stats.spearmanr(population,ViolentCrimesPerPop)
print(corrSpearman,pValSpearman)
corrPearson,pValPearson = stats.pearsonr(PctForeignBorn,ViolentCrimesPerPop)
print(corrPearson,pValPearson)
corrSpearman,pValSpearman = stats.spearmanr(PctForeignBorn,ViolentCrimesPerPop)
print(corrSpearman,pValSpearman)


# In[7]:



fig,ax=plt.subplots(1,2,figsize=(12,6))
ax[0].scatter(x=crimeclean["racepctblack"],y=crimeclean["ViolentCrimesPerPop"])
ax[0].set_xlabel("racepctblack")
ax[0].set_ylabel("Violent Crimes Per Population")
ax[1].scatter(x=crimeclean["PctForeignBorn"],y=crimeclean["ViolentCrimesPerPop"])
ax[1].set_xlabel("PctForeignBorn")
ax[1].set_ylabel("Violent Crimes Per Population")
plt.show()


# In[8]:


heart=pd.read_csv("heart.csv")
heart


# In[9]:


heartreplace=heart.replace({'sex':0,'target':0},'F')


# In[10]:


heartcategory=heartreplace.replace({'sex':1,'target':1},'M')
heartcategory


# In[11]:


heartclean=heartcategory.replace({'sex':'F','target':'F'},1)


# In[12]:


heartclean=heartclean.replace({'sex':'M','target':'M'},0)
heartclean


# In[13]:


heartclean=heartclean.rename(columns={"sex": "gender", "target": "hasHeartDisease"})
heartclean


# Does the resting blood pressure (trestbps) differ between those with the disease and those without?

# In[14]:


heartdisease=heartclean[heartclean['hasHeartDisease']==1]
noheartdisease=heartclean[heartclean['hasHeartDisease']==0]
print("Mean Resting BP of the group having heart disease:",heartdisease['trestbps'].mean())
print("Mean Resting BP of the group not having heart disease:",noheartdisease['trestbps'].mean())
print("STD Resting BP of the group having heart disease:",heartdisease['trestbps'].std())
print("STD Resting BP of the group not having heart disease:",noheartdisease['trestbps'].std())


# In[15]:


fig, axs = plt.subplots(1,2,sharey=True,figsize=(10,6))
axs[0].boxplot(heartdisease['trestbps'])
axs[0].set_title('Heart Disease Rest BP')
axs[1].boxplot(noheartdisease['trestbps'])
axs[1].set_title('No Heart Disease Rest BP')


# In[16]:


fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs[0].hist(heartdisease['trestbps'], bins=20)
axs[1].hist(noheartdisease['trestbps'], bins=20)


# In[17]:


stats.ttest_ind(heartdisease['trestbps'],noheartdisease['trestbps'])


# In[18]:


from numpy import mean
from numpy import var
from math import sqrt
def cohend(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = mean(d1), mean(d2)
	# calculate the effect size
	return (u1 - u2) / s
d=cohend(heartdisease['trestbps'],noheartdisease['trestbps'])
print("Cohen's distance : ",d)


# Try some other (quantitative) variables and compare between heart disease groups. You could also compare by gender but there's probably no good reason to.

# In[19]:


print("Mean Cholestrol of the group having heart disease:",heartdisease['chol'].mean())
print("Mean Cholestrol of the group not having heart disease:",noheartdisease['chol'].mean())
print("STD Cholestrol of the group having heart disease:",heartdisease['chol'].std())
print("STD Cholestrol of the group not having heart disease:",noheartdisease['chol'].std())


# In[20]:


fig, axs = plt.subplots(1,2,sharey=True,figsize=(10,6))
axs[0].boxplot(heartdisease['chol'])
axs[0].set_title('Heart Disease Cholestrol')
axs[1].boxplot(noheartdisease['chol'])
axs[1].set_title('No Heart Disease Cholestrol')


# In[21]:


fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs[0].hist(heartdisease['chol'], bins=20)
axs[1].hist(noheartdisease['chol'], bins=20)


# In[22]:


stats.ttest_ind(heartdisease['chol'],noheartdisease['chol'])


# In[23]:


from numpy import mean
from numpy import var
from math import sqrt
def cohend(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = mean(d1), mean(d2)
	# calculate the effect size
	return (u1 - u2) / s
d=cohend(heartdisease['chol'],noheartdisease['chol'])
print("Cohen's distance : ",d)


# Calculate the proportion of men and women who have heart disease from the sample. To save you time, this code (is one of many ways that) will do it.

# In[24]:


#Count the number with the disease for each gender type
heartclean['gender'] = heartclean.gender.replace({0: "Male", 1: "Female"})
hasDiseaseCount=heartclean[heartclean.hasHeartDisease==True].groupby("gender").count().hasHeartDisease
hasDiseaseCount
#Count the number of gender type
totalCount=heartclean.groupby("gender").count()['hasHeartDisease']

#combine into a dataframe (both are indexed with gender, so will be matched) and specify the columns
p=pd.concat([hasDiseaseCount, totalCount], axis=1)
p.columns = ["heartDiseaseCount", "totalCount"]

#create a new column and calculate the proportion
p['propHeartDisease']=p["heartDiseaseCount"]/p["totalCount"]

#print the results
print(p.head())
p.totalCount[0]


# In[25]:


#Best estimate is p1 - p2. Get p1 and p2 from the chart p above
p_fe = p.propHeartDisease.Female
p_male = p.propHeartDisease.Male
p_total=len(heartdisease)/len(heartclean)


# In[26]:


#calculated in the beginning of the previous example
n1 = p.totalCount.Female
n2 = p.totalCount.Male
se = np.sqrt(p_total*(1-p_total)*(1/n1 + 1/n2))


# In[29]:


#calculate the best estimate
be = p_fe - p_male  #Calculate the hypothesized estimate
#Our null hypothesis is p1 - p2 = 0he = 0  #Calculate the test statistic
test_statistic = (be)/se
test_statistic


# In[32]:


import scipy.stats.distributions as dist
pvalue = 2*dist.norm.cdf(-np.abs(test_statistic))
pvalue


# In[ ]:




