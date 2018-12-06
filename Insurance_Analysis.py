import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
ins=pd.read_csv("insurance.csv")
tot=ins.count()
s=ins['sex'].value_counts()
result=s/tot[0]*100
#% of male and female
fig, axes = plt.subplots(1,1, figsize=(10,3))
plt.pie(result.loc[result.index], labels=result.index, autopct='%.2f')
axes.set(ylabel='', title=result.index, aspect='equal')
axes.set_title("Pie chart for Male & Female percentage")
 
#nmber of children 
s=ins['children'].value_counts()
plt.bar(s.index,s.values,label="BMW",width=0.5)
plt.title('Children Count')

#Smoking or Non smoking
smk=ins['smoker'].value_counts()
smk_p=smk/tot[0]*100 
fig, axes = plt.subplots(1,1, figsize=(10,3)) 
plt.pie(smk_p.loc[smk_p.index], labels=smk_p.index, autopct='%.2f')
axes.set(ylabel='', title=result.index, aspect='equal')
axes.set_title("Pie chart for smoker & Non-smoker percentage")

#BMI values using histogram
b=ins['bmi'].value_counts()
#plt.bar(b.index,b.values,label="BMW",width=0.5)
plt.title('BMI values')
b.hist()

#Age bar chart
ag=ins['age'].value_counts() 
plt.title('age values')
plt.bar(ag.index,ag.values,label="BMW",width=0.5)
plt.title('Age')

# Region percentage
reg=ins['region'].value_counts()
reg_p=reg/tot[0]*100 
plt.pie(reg_p.loc[reg_p.index], labels=reg_p.index, autopct='%.2f') 
plt.title("Pie chart for region percentage")

#Average charges based on region,sex & smoker
chrg=ins.groupby(['region','sex','smoker'])['charges'].mean()
plt.pie(chrg, labels=chrg.index, autopct='%.2f') 
plt.title("Average charges based on region,sex & smoker")