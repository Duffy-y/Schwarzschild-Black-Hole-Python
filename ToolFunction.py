import os
import numpy as np
import math

def WriteArrayToText(array, filename):
    np.save("Array/" + filename + ".npy", array, allow_pickle=True)

def ReadTextToArray(filename):
    array = np.load(filename, allow_pickle=True)
    return array

def CheckIfFileExist(filename):
    if os.path.isfile(filename):
        return True
    else:
        return False

def CreateDirectory(dirName):
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")

def SphericToCartesian(theta, phi):
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)    
        return x,y,z

def CartesianToSpheric(x, y, z):
    theta = math.acos(z)
    phi = math.atan2(y,x)

    while phi < 0:
        phi += math.radians(360)
    while theta < 0:
        theta =+ math.pi
    if phi == math.radians(360):
        phi = 0

    return theta, phi