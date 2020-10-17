from ImageClass import *
from BlackHole import *
import scipy.interpolate as scp

if __name__ == "__main__":
    BaseImage = Image.open("milkyway.jpg")

    rs = int(input("Rs = "))
    d = int(input("D = "))
    BH = BlackHole(rs, d, 360, BaseImage.size[0], BaseImage.size[1])
    #SeenAngle, DeviatedAngle = BH.GetValue()
    #Interpolation = scp.interp1d(SeenAngle, DeviatedAngle, kind='linear', bounds_error=True)
    #Image = ImageClass("milkyway.jpg", Interpolation, d, rs, 360, 0, repetition=1)
