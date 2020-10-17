import multiprocessing as mp
import numpy as np
import cv2 as cv
import sys
import os

from PIL import Image
from tqdm import tqdm
from ToolFunction import *

class ImageClass:
    def __init__(self, FileName, interpolation, D, Rs, FOV, imageType = 0, FPS = 20, time = 5, repetition = 1):
        # Type : 0 => Image     Type : 1 => Rendering multiple sur la même image    Type : 2 => Vidéo
        self.ImageObject = self.ResizeImage(Image.open(FileName))
        ImageName = FileName.split(".")
        self.FileName = f"{ImageName[0]} D={D} Rs={Rs}"

        self.RenderType         = imageType
        self.FPS                = FPS
        self.Time               = time
        self.Repetition         = repetition
        self.Width, self.Height = self.ImageObject.size

        if self.RenderType == 2:
            self.Repetition = self.FPS * self.Time
        elif self.RenderType == 0:
            self.Repetition = 1

        self.D      = D
        self.Rs     = Rs
        self.FOV_X  = FOV
        self.FOV_Y  = FOV * (self.Height / self.Width)
        self.PPDX   = self.Width / self.FOV_X
        self.PPDY   = self.Height / self.FOV_Y

        self.Interpolation  = interpolation
        self.Array          = None

        self.CPUCount   = mp.cpu_count()
        self.ValueArray = [None] * self.CPUCount
        self.WidthArray = np.arange(0, self.Width)
        self.WidthTable = np.array_split(self.WidthArray, self.CPUCount)

        self.ValueArray       = self.MultiThreadedArray()
        self.MultiThreadedRendering()

    def ResizeImage(self, img):
        if img.size[0] % 2 != 0:
            crop = img.crop((0, 0, img.size[0] - 1, img.size[1]))
            img = crop
        if img.size[1] % 2 != 0:
            crop = img.crop((0, 0, img.size[0], img.size[1] - 1))
            img = crop
        if img.size[0] / 2 < img.size[1] and img.size[0] > img.size[1]:
            crop = img.crop((0, 200, img.size[0], img.size[0]/2 + 200))
            img = crop
        elif img.size[0] < img.size[1] / 2 and img.size[0] < img.size[1]:
            crop = img.crop((0, 200, img.size[0], img.size[0]/2 + 200))
            img = crop
		return img

    def EulerMatrices(self, beta):
        """ Retourne la matrice de rotation selon les formules d'Euler autour de l'axe x (équateur) avec un angle de rotation beta """
        a = math.cos( beta / 2 )
        b = -1 * math.sin( beta / 2 )
        aa, bb, ab = a**2, b**2, a * b
        return np.array([[ aa + bb, 0       , 0      ],
                        [ 0      , aa - bb , 2 * ab ],
                        [ 0      , -2 * ab , aa - bb],
                        ])

    def FindPixel(self, x, y):
        """ A partir des coordonnées (x;y) du seen_pixel, on recherche les coordonnées de deviated_pixel """
        if y==0:
            return (x,y)

        phi,theta = x * self.FOV_X / 360 / self.PPDX, y * self.FOV_Y / 180 / self.PPDY
        phi,theta= phi + (360 - self.FOV_X) / 2, theta + (180 - self.FOV_Y) / 2
        u, v, w= SphericToCartesian(math.radians(theta), math.radians(phi))

        if theta == 90:
            beta = 0
        elif phi == 180 or phi == 0:
            beta = math.radians(90)
        else:
            beta = -math.atan(w / v)

        v2              = np.dot( self.EulerMatrices(beta), [u,v,w] )
        _,seen_angle    = CartesianToSpheric(v2[0], v2[1], v2[2])
        seen_angle      = math.degrees(seen_angle)

        if seen_angle > 360:
            seen_angle -= 360
        if seen_angle > 180:
            seen_angle = 360 - seen_angle
            try:
                deviated_angle = 360 - self.Interpolation(seen_angle)
            except:
                return (-1,-1)
        else:
            try:
                deviated_angle = self.Interpolation(seen_angle)
            except:
                return (-1,-1)

        u, v, w     = SphericToCartesian(math.radians(90), math.radians(deviated_angle))
        v2          = np.dot(self.EulerMatrices(-beta), [u,v,w])
        theta, phi  = CartesianToSpheric(v2[0], v2[1], v2[2])
        theta, phi  = math.degrees(theta), math.degrees(phi)
        phi, theta  = phi - (360 - self.FOV_X) / 2, theta - (180 - self.FOV_Y) / 2
        x2, y2      = phi * 360 / self.FOV_X * self.PPDX, theta * 180 / self.FOV_Y * self.PPDY
        return (round(x2),round(y2))

    def CreateArray(self, width, pipe):
        """ Associe à un couple de pixel, la position x et y de pixel dévié, le tout dans un tableau deux dimension. """
        ArrayLink = np.array([[(-1, -1 )] * self.Height] * len(width)) # array_x[x,y]
        for x in tqdm(range(len(width))):
            for y in range(self.Height):
                pixel = self.FindPixel(width[x], y)
                ArrayLink[x,y] = pixel
        pipe.send(ArrayLink)

    def CreatePixelData(self, croppedImg, pipe, processNb):
        w, h = croppedImg.size
        for x in range(w):
            for y in range(h):
                x2, y2 = self.ValueArray[processNb][x, y][0], self.ValueArray[processNb][x, y][1]
                x2, y2 = int(x2), int(y2)
                if x2 != -1:
                    try:
                        p = self.ImageObject.getpixel((x2, y2))
                        croppedImg.putpixel((x, y), p)
                    except:
                        croppedImg.putpixel((x, y), (0, 0, 0))
                else:
                    croppedImg.putpixel((x,y), (0, 0, 0))
        pipe.send(croppedImg)

    def CreateVideo(self, imageArray):
        videoSize = (self.Width, self.Height)
        fourcc = cv.VideoWriter_fourcc(*"avc1")
        video = cv.VideoWriter("Simulation/" + self.FileName + ".mp4", fourcc, self.FPS, videoSize)

        for i in range(self.FPS * self.Time):
            tempImg = imageArray[i].copy()
            video.write(cv.cvtColor(np.array(tempImg), cv.COLOR_RGB2BGR))
        video.release()
        return video

    def MultiThreadedArray(self):
        process = []
        pipe    = []
        array   = [None] * self.CPUCount

        CreateDirectory("Array")
        if CheckIfFileExist("Array/" + self.FileName + ".npy"):
            array = ReadTextToArray("Array/" + self.FileName + ".npy")
        else:
            for i in range(self.CPUCount):
                parentPipe, childPipe = mp.Pipe()
                p = mp.Process(target=self.CreateArray, args=(self.WidthTable[i], childPipe))
                process.append(p)
                pipe.append(parentPipe)
                p.start()

            for i in range(self.CPUCount):
                array[i] = pipe[i].recv()
                process[i].join()

            for i in range(self.CPUCount):
                process[i].close()

            WriteArrayToText(array, self.FileName)
        return array

    def MultiThreadedRendering(self):
        process             = []
        pipe                = []
        imageTable          = []
        finalCroppedTable   = [None] * self.CPUCount
        renderedImage       = []

        FinalImg            = Image.new("RGB", self.ImageObject.size)
        CreateDirectory("Simulation")

        for i in tqdm(range(self.Repetition)):
            if self.RenderType == 2:
                for k in range(self.CPUCount):
                    if i > 0:
                        leftSide = self.ImageObject.crop((0, 0, 10, self.Height))
                        rightSide = self.ImageObject.crop((10, 0, self.Width, self.Height))
                        self.ImageObject.paste(rightSide, (0, 0, self.Width - 10, self.Height))
                        self.ImageObject.paste(leftSide, (self.Width - 10, 0, self.Width, self.Height))
                    img1 = self.ImageObject.crop((self.WidthTable[k][0], 0, self.WidthTable[k][-1] + 1, self.Height))
                    imageTable.append(img1)

            if self.RenderType != 2:
                for k in range(self.CPUCount):
                    img1 = self.ImageObject.crop((self.WidthTable[k][0], 0, self.WidthTable[k][-1] + 1, self.Height))
                    imageTable.append(img1)

            for k in range(self.CPUCount):
                parentPipe, childPipe = mp.Pipe()
                p = mp.Process(target=self.CreatePixelData, args=(imageTable[k], childPipe, k))
                process.append(p)
                pipe.append(parentPipe)
                p.start()

            for k in range(self.CPUCount):
                finalCroppedTable[k] = pipe[k].recv()
                process[k].join()

            for k in range(self.CPUCount):
                FinalImg.paste(finalCroppedTable[k], (self.WidthTable[k][0], 0, self.WidthTable[k][-1] + 1, self.Height))
                process[k].close()

            if self.RenderType != 0:
                process.clear()
                pipe.clear()
                finalCroppedTable.clear()
                finalCroppedTable = [None] * self.CPUCount

            if self.RenderType == 2:
                renderedImage.append(FinalImg.copy())

            if self.RenderType == 1:
                self.ImageObject = FinalImg

        if self.RenderType == 0:
            FinalImg.save("Simulation/" + self.FileName + ".png")
        elif self.RenderType == 1:
            FinalImg.save("Simulation/" +self.FileName + ".png")
        elif self.RenderType == 2:
            self.CreateVideo(renderedImage)
