import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
# a mano
x=np.array([1,2,3,4])
y=np.array([3,6,9,12])
n=len(x)
plt.scatter(x,y)
plt.show()
x_prom=x.mean()
y_prom=y.mean()
sxx=sum((x-x_prom)**2)
b1_est=sum(((x-x_prom)/sxx)*y)
b0_est=y_prom-(b1_est*x_prom)
#librerias especiales
A=np.vstack([x,np.ones(len(x))]).T
m,c=np.linalg.lstsq(A,y)[0]
plt.plot(x,y,'o',label='Original data',markersize=10)
plt.plot(x,m*x+c,'r',label='Fitted line')
plt.legend()
plt.show()
#sklearn
from sklearn import datasets,linear_model
diabetes=datasets.load_diabetes()
diabetes_X=diabetes.data[:,np.newaxis,2]
diabetes_X_train=diabetes_X[:-20]
diabetes_X_test=diabetes_X[-20:]
diabetes_y_train=diabetes.target[:-20]
diabetes_y_test=diabetes.target[-20:]
regr=linear_model.LinearRegression()
regr.fit(diabetes_X_train,diabetes_y_train)
print('Coeficiente: ',regr.coef_)
y_pred=regr.predict(diabetes_X_test)
mse=np.mean((y_pred-diabetes_y_test)**2)
print('Mean squared error: %.2f' % mse)
plt.plot(diabetes_X_test,y_pred,label='Prediccion')
plt.scatter(diabetes_X_test,diabetes_y_test,color='r',label='Datos originales')
plt.legend()
plt.show()