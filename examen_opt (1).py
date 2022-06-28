#!/usr/bin/env python
# coding: utf-8

# In[163]:


import numpy as np
import matplotlib.pyplot  as plt
import pandas as pd
from math import *


# In[166]:


D=1
def f(w):
    return 418.9829-w*sin(np.sqrt(w))


# In[167]:


import tensorflow as tf
x = []
y = []
for i in range(-500,500):
    x.append(i)
for i in x:
    y.append(f(x[i]))
print(y)
plt.plot(x,y)
plt.show()


# In[95]:


def DerrivF (w):
    Dc= -sin(np.sqrt(w[0]))-1/2*(np.sqrt(w[0]))*cos(np.sqrt(w[0]))
    return Dc


# In[96]:


def gd(x, grad, alpha, max_iter=10):
    xs = np.zeros((1 + max_iter, x.shape[0]))
    xs[0, :] = x
    for i in range(max_iter):
        x = x - alpha * grad(x)
        xs[i+1] = x
    return xs


# In[97]:


def gd_momentum(x, grad, alpha, beta=0.9, max_iter=10):
    xs = np.zeros(1 + max_iter)
    xs[0] = x
    v = 0
    for i in range(max_iter):
        v = beta*v + (1-beta)*grad(x)
        vc = v/(1+beta**(i+1))
        x = x - alpha * vc
        xs[i+1] = x
    return xs


# In[130]:


w = np.array([20])

Dc = -sin(np.sqrt(w[0]))-1/2*(np.sqrt(w[0]))*cos(np.sqrt(w[0]))

xs = gd(np.array([20]), DerrivF, alpha= 0.002, max_iter= 100000)

print(xgrad)
plt.plot(xs)


# In[99]:


w = np.linspace(-1.2, 1.2, 100)
y = np.linspace(-1.2, 1.2, 100)
W, Y = np.meshgrid(x, y)
levels = [0.1,1,2,4,9, 16, 25, 36, 49, 64, 81, 100]
Z = w**2 + Y**2
c = plt.contour(W, Y, Z, levels)
pass


# In[124]:


w = np.array([20])

Dc = -sin(np.sqrt(w[0]))-1/2*(np.sqrt(w[0]))*cos(np.sqrt(w[0]))

xs = gd_momentum(w, DerrivF, alpha= 0.002, beta= 0.9, max_iter= 100000)

print(xs)
plt.plot(xs)


# In[102]:


w = np.linspace(-1.2, 1.2, 100)
y = np.linspace(-1.2, 1.2, 100)
W, Y = np.meshgrid(w, y)
levels = [0.1,1,2,4,9, 16, 25, 36, 49, 64, 81, 100]
Z = w**2 + Y**2
c = plt.contour(W, Y, Z, levels)
plt.plot(xs[0], xs[1], 'o-', c='red')
plt.title('Gradieent descent with momentum')
pass


# In[126]:


def gd2_momentum(x, grad, alpha, beta=0.9, max_iter=10):
    xs = np.zeros((1 + max_iter, x.shape[0]))
    xs[0, :] = x
    v = 0
    for i in range(max_iter):
        v = beta*v + (1-beta)*grad(x)
        vc = v/(1+beta**(i+1))
        x = x - alpha * vc
        xs[i+1, :] = x
    return xs


# In[127]:


w = np.array([20])

Dc = -sin(np.sqrt(w[0]))-1/2*(np.sqrt(w[0]))*cos(np.sqrt(w[0]))

xs = gd2_momentum(w, DerrivF, alpha= 0.002, beta= 0.9, max_iter= 100000)

print(xs)
plt.plot(xs)


# In[92]:


w = np.linspace(-1.2, 1.2, 100)
y = np.linspace(-1.2, 1.2, 100)
W, Y = np.meshgrid(w, y)
levels = [0.1,1,2,4,9, 16, 25, 36, 49, 64, 81, 100]
Z = w**2 + Y**2
c = plt.contour(W, Y, Z, levels)
plt.plot(xs[:, 0], xs[:, 1], 'o-', c='red')
plt.title('Gradieent descent with momentum')
pass


# In[110]:


def gd2_rmsprop(x, grad, alpha, beta=0.9, eps=1e-8, max_iter=10):
    xs = np.zeros((1 + max_iter, x.shape[0]))
    xs[0, :] = x
    v = 0
    for i in range(max_iter):
        v = beta*v + (1-beta)*grad(x)**2
        x = x - alpha * grad(x) / (eps + np.sqrt(v))
        xs[i+1, :] = x
    return xs


# In[128]:


w = np.array([20])

Dc = -sin(np.sqrt(w[0]))-1/2*(np.sqrt(w[0]))*cos(np.sqrt(w[0]))

xs = gd2_rmsprop(w, DerrivF, alpha= 0.005, beta= 0.9, eps=1e-8, max_iter= 10)
#xs = gd2_rmsprop(w, DerrivF, alpha= 0.002, beta= 0.9, max_iter= 100000)

print(xs)
plt.plot(xs)


# In[119]:


w = np.linspace(-1.2, 1.2, 100)
y = np.linspace(-1.2, 1.2, 100)
W, Y = np.meshgrid(w, y)
levels = [0.1,1,2,4,9, 16, 25, 36, 49, 64, 81, 100]
Z = w**2 + Y**2
c = plt.contour(W, Y, Z, levels)
plt.plot(xs[0], xs[1], 'o-', c='red')
plt.title('Gradieent descent with momentum')
pass


# In[133]:


def gd2_adam(x, grad, alpha, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=10):
    xs = np.zeros((1 + max_iter, x.shape[0]))
    xs[0, :] = x
    m = 0
    v = 0
    for i in range(max_iter):
        m = beta1*m + (1-beta1)*grad(x)
        v = beta2*v + (1-beta2)*grad(x)**2
        mc = m/(1+beta1**(i+1))
        vc = v/(1+beta2**(i+1))
        x = x - alpha * m / (eps + np.sqrt(vc))
        xs[i+1, :] = x
    return xs


# In[134]:


w = np.array([20])
xgrad = gd2_adam(w, DerrivF, alpha= 0.005, beta1=0.9, beta2=0.999, eps=1e-8, max_iter= 10)
print(xgrad)
plt.plot(xgrad)


# In[137]:


w = np.linspace(-1.2, 1.2, 100)
y = np.linspace(-1.2, 1.2, 100)
W, Y = np.meshgrid(w, y)
levels = [0.1,1,2,4,9, 16, 25, 36, 49, 64, 81, 100]
Z = w**2 + Y**2
c = plt.contour(W, Y, Z, levels)
plt.plot(xgrad [0], xgrad[1], 'o-', c='red')
plt.title('Gradieent descent with momentum')
pass


# In[ ]:




