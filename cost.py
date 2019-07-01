import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model, metrics 


data = pd.read_csv("dataset/housing.csv")
data.head()

#scatter plot
X= data['total_bedrooms'].head(50)
Y= data['total_rooms'].head(50)
plt.figure(figsize=(16, 8))
plt.scatter( X,Y, c='black')
plt.xlabel("total_rooms")
plt.ylabel("total_bedrooms")
plt.show()

#calculate cost
costfunc=np.zeros(1000)
a=0.0000001
theta0=0
theta1=0
m=len(x)
print(m)

def Cost(theta0,theta1,x,y):
    s=0
    for i in range(m):
        mul=(theta0+theta1*x[i])-y[i]
        s=s+mul
    return s

def cost1(theta0,theta1,x,y):
    for i in range(m):
        mul=((theta0+theta1*x[i])-y[i])*x[i]
    return mul

for i in range(1000):
    
    costfunc[i]=((Cost(theta0,theta1,x,y))**2)/2*m
    temp0=theta0-(a*float(Cost(theta0,theta1,x,y))/m)
    temp1=theta1-(a*(cost1(theta0,theta1,x,y))/m)
    theta0=temp0
    theta1=temp1
    
 #plot iteration versus cost
 itera=np.arange(0,1000,1)
plt.plot(itera,costfunc, '-r',color='b')
plt.xlabel("iteration")
plt.ylabel("Cost function")
