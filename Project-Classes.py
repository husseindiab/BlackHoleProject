#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:37:00 2020

@author: husseindiab
"""


import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
import scipy.optimize as op
import scipy.interpolate as sci
import PIL as pil
from scipy.constants import pi


R=8  
D=50

pil.Image.MAX_IMAGE_PIXELS=None
image='space2.jpg'
original_image=pil.Image.open(image)
original_X=original_image.size[0]
original_Y=original_image.size[1]
original_angular_X=original_X/360
original_angular_Y=original_Y/180


FOV_Y=180                                          
FOV_X=FOV_Y*original_X/original_Y
#FOV_X=360
#FOV_Y=FOV_X*original_Y/original_X

new_X=original_X
new_Y=original_Y
'''new_Y=new_X'''
new_angular_x=new_X/360
new_angular_y=new_Y/180
new_image = pil.Image.new('RGB', (new_X, new_Y))

# %%
def diff_func(phi,r):
    v0=r[1]
    v1=(2/r[0])*(v0**2)-1.5*R+r[0]
    return v0,v1

def event_horizon(phi,r):
    z=r[0]-R
    return z
event_horizon.terminal = True


def Path(D, alpha):
    
    y0=[D,-D/np.tan(math.radians(alpha))]
    solution=spi.solve_ivp(diff_func, (0,100*pi), y0 ,events=event_horizon)
    
    phi=solution.t
    r=solution.y[0,:]
    
    return phi,r


# %%
def deviated_angle(alpha):
    
    phi,r=Path(D,alpha)
    deviated_angle=(phi[-1]+np.arcsin(D/r[-1])*np.sin(phi[-1]))
    deviated_angle_degrees=math.degrees(deviated_angle)
    return deviated_angle_degrees


def neg_dev_angle(alpha):
    z=-deviated_angle(alpha)
    return z

    
def find_alpha_min():
    if R/D>2/3:
        alpha0=180
    elif R/D>=2/25:
        alpha0=100
    elif R/D>=1/50:
        alpha0=20
    else:
        alpha0=4
    alpha_min=op.fmin(neg_dev_angle,alpha0,disp=False)
   
    return float(alpha_min)


def trajectories():
    
    alpha_min = find_alpha_min()
    print('alpha min =',alpha_min)

    seen_angles = []
    deviated_angles = []

    for alpha in np.linspace(180, alpha_min, num=1000,endpoint=True):
        r, phi = Path(D, alpha)
        s=180-alpha
        d=deviated_angle(alpha)
        
        seen_angles.append(s)
        
        deviated_angles.append(d)
    

    
    f = sci.interp1d(seen_angles, deviated_angles, 'cubic')     
    
    seen_angles_inter=np.linspace(seen_angles[0],seen_angles[-1],50000)
    deviated_angles_inter=f(seen_angles_inter)
    
    
    plt.plot(seen_angles_inter, deviated_angles_inter)
    plt.xlabel('Seen angle')
    plt.ylabel('Deviated angle')
    plt.title('Deviated angle found from observed seen angle')
    # plt.savefig('interpolation',dpi=1000)
    plt.show()
      
    
    return f




# %%


def spheric2cart(theta, phi):
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)
    return x, y, z
    
    
def cart2spheric(x, y, z):
       
    theta = math.acos(z)
    phi = math.atan2(y, x)
    while phi < 0:
        phi += pi + pi
    while theta < 0:
        theta += pi
    if phi == (pi + pi):
        phi = 0
    return theta, phi
    
    

def rotation_matrix(beta):

    a = math.cos(beta / 2.0)
    b = -math.sin(beta / 2.0)
    return np.array([[a**2 + b**2, 0, 0],
                     [0, a**2 - b**2, 2*a*b],
                     [0, -2*a*b, a**2 - b**2]])


# %%

class pixel():
    
    def __init__(self,x,y):
        self.distorted_position=[x,y]
        self.original_position=[0,0]
        self.colour=(0,0,0)
        
        self.find_position()
        self.find_colour()
    
    def find_position(self):
        
        if self.distorted_position[1] == 0:
            self.original_position=self.distorted_position
            return
    
        phi, theta = self.distorted_position[0]*FOV_X/360/original_angular_X, self.distorted_position[1]*FOV_Y/180/original_angular_Y #convert position in spheric coord
        phi, theta = phi+(360-FOV_X)/2, theta+(180-FOV_Y)/2
        # print(phi, theta)
        u, v, w = spheric2cart(math.radians(theta), math.radians(phi)) 
    
        if theta == 90:
            beta = 0
    
        elif phi == 180 or phi == 0:
            beta = pi/2
    
        else:
            beta = -math.atan(w/v) 
    
        v2 = np.matmul(rotation_matrix(beta), [u, v, w]) 
        _, seen_angle = cart2spheric(v2[0], v2[1], v2[2]) 
        seen_angle = math.degrees(seen_angle)
    
        if seen_angle > 360: 
            seen_angle -= 360
    
        if seen_angle > 180: 
            seen_angle = 360-seen_angle
    
            try:
                deviated_angle=360-interpolation(seen_angle) 
                
            except:
                self.original_position=[-1,-1]
                return
                                
        else:
    
            try:
                deviated_angle = interpolation(seen_angle) 
                
            except:
                self.original_position=[-1,-1]
                return
    
        u, v, w = spheric2cart(pi/2, math.radians(deviated_angle)) 
        v2 = np.dot(rotation_matrix(-beta), [u, v, w]) 
        theta, phi = cart2spheric(v2[0], v2[1], v2[2])   
        theta, phi = math.degrees(theta), math.degrees(phi)
        phi, theta = phi-(360-FOV_X)/2, theta-(180-FOV_Y)/2
        x2, y2 = phi*360/FOV_X*original_angular_X, theta*180/FOV_Y*original_angular_Y 
        self.original_position=[int(x2),int(y2)]
         
    
    def find_colour(self):
        original_pixels=original_image.load()
        
        if self.original_position[0]!=-1 and self.original_position[0]<new_X and self.original_position[1]<new_Y:
            (R,G,B)=original_pixels[self.original_position[0],self.original_position[1]]                
            self.colour=(R,G,B)
                
        else:
            self.colour=(0,0,0)
                
            
#%% 

b=pixel(600,500)
d=b.colour
print(d) 

    
# %%

def generate_image():
    
    new_pixels=new_image.load()
    
    for i in range(0,new_X):
        if i == round(new_X/10):
            print("10%")
        elif i == round(new_X/5):
            print("20%")
        elif i == round(new_X*3/10):
            print("30%")
        elif i == round(new_X*2/5):
            print("40%")
        elif i == round(new_X/2):
            print("50%")
        elif i == round(new_X*3/5):
            print("60%")
        elif i== round(new_X*7/10):
            print("70%")
        elif i == round(new_X*4/5):
            print("80%")
        elif i == round(new_X*9/10):
            print("90%")
        for j in range(0,new_Y):
            a=pixel(i, j)
            new_pixels[i,j]=a.colour
    
    new_image.show()
    

# %%

interpolation=trajectories()
# %%
generate_image()

# %%






























