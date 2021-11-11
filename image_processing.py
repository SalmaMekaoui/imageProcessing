# salma mekaoui
# N131173645
# MASTER SIR
import tkinter as tk
import tkinter
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import os
import sounddevice as sd
import soundfile as sf
import _thread
from PIL import Image,ImageTk
import PIL
from PyQt5.QtWidgets import (QMainWindow, QTextEdit, QVBoxLayout, QPushButton, QAction, QFileDialog, QApplication)
import cv2 as cv
import cv2
import math
import numpy as np
from numpy import *
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
from matplotlib.figure import Figure



def donothing():
     A=0
     
    
root =Tk()
partition11 = LabelFrame(root,text="l'image a traiter")
partition11.pack(side=LEFT,fill="both",expand="yes",padx=10,pady=10)
partition12 = LabelFrame(root,text="l'image resultante")
partition12.pack(side=RIGHT,fill="both",expand="yes",padx=10,pady=10)



def openfn():
    #global filname
    #filename = tkinter.filedialog.askopenfilename(title="CHOISIR UNE IMAGE",filetypes=[('png files','.png'),('all files','.*')])
    filename = filedialog.askopenfilename(title='open')
    return filename
def open_img():
    x = openfn()
    img = Image.open(x)
    img1 = img.save("image.jpg")
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(partition11, image=img)
    panel.image = img
    panel.pack()
    buttonReset = Button(partition11, text = 'reset', command=lambda: panel.pack_forget())
    buttonReset.pack()
    
    
    
    
    


def Sobel():
        seuil=128
        img = cv.imread("image.jpg");
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        IX = img.copy()
        IY = img.copy()
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                 IX[i,j]=(-img[i-1,j-1]-2*img[i,j-1]-img[i+1,j-1])+(img[i-1,j+1]+2*img[i,j+1]+img[i+1,j+1])
                 IY[i,j]=(img[i+1,j-1]-img[i-1,j-1])+(2*img[i+1,j]-2*img[i-1,j])+(img[i+1,j+1]-img[i-1,j+1])
                 
        IR = img.copy()
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                IR[i, j] = math.sqrt(IX[i, j] ** 2 + IY[i, j] ** 2)
                if IR[i, j] < seuil:
                    IR[i, j] = 0
                else:
                    IR[i, j] = 255 
        #enregidtrer l'image  filtrer apres la transformer et l'afficher a l'autre emplacement            
        imag = Image.fromarray(IR)
        img1 = imag.save("imageSobel.jpg")
        img = Image.open('imageSobel.jpg')
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(partition12, image=img)
        panel.image = img
        panel.pack()
        buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
        buttonReset.pack()
        
    


        
def gradient():
        seuil=128
        img = cv.imread('image.jpg');
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        IX = img.copy()
        IY = img.copy()
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                IX[i, j] = img[i, j + 1] - img[i, j]
                IY[i, j] = img[i + 1, j] - img[i, j]
        IR = img.copy()
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                IR[i, j] = math.sqrt(IX[i, j] ** 2 + IY[i, j] ** 2)
                if IR[i, j] < seuil:
                    IR[i, j] = 0
                else:
                    IR[i, j] = 255
        imag = Image.fromarray(IR)
        img1 = imag.save("imageGRAD.jpg")
        img = Image.open('imageGRAD.jpg')
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(partition12, image=img)
        panel.image = img
        panel.pack()
        buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
        buttonReset.pack()
        
def Laplacien():
        seuil=128
        img = cv.imread('image.jpg');
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        IR = img.copy()
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                IR[i, j] = -4 * img[i, j] + img[i - 1, j] + img[i + 1, j] + img[i, j - 1] + img[i, j + 1]
                if IR[i, j] < seuil:
                    IR[i, j] = 0
                else:
                    IR[i, j] = 255
        imag = Image.fromarray(IR)
        img1 = imag.save("imagelaplacien.jpg")
        img = Image.open('imagelaplacien.jpg')
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(partition12, image=img)
        panel.image = img
        panel.pack() 
        buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
        buttonReset.pack()

def robinson():
        seuil=128
        img = cv.imread('image.jpg');
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        IR = img.copy()
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                IR[i, j] = 1/3 * img[i-1, j-1] + 1/3 * img[i - 1, j] +1/3 * img[i - 1, j+1] - 1/3 * img[i+1 , j - 1] - 1/3 * img[i+1, j ] - 1/3 * img[i+1, j ]
                if IR[i, j] < seuil:
                    IR[i, j] = 0
                else:
                    IR[i, j] = 255
        imag = Image.fromarray(IR)
        img1 = imag.save("imageROBINSON.jpg")
        img = Image.open('imageROBINSON.jpg')
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(partition12, image=img)
        panel.image = img
        panel.pack() 
        buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
        buttonReset.pack()

def kirsh():
        seuil=128
        img = cv.imread('image.jpg');
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        IR = img.copy()
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                IR[i, j] =5/15 * img[i-1, j-1] + 5/15 * img[i - 1, j] +5/15 * img[i - 1, j+1] - 3/15 * img[i+1 , j - 1] - 3/15 * img[i+1, j ] - 3/15 * img[i+1, j ] - 3/15 * img[i , j - 1]- 3/15 * img[i , j + 1]
                if IR[i, j] < seuil:
                    IR[i, j] = 0
                else:
                    IR[i, j] = 255
        imag = Image.fromarray(IR)
        img1 = imag.save("imagekirsh.jpg")
        img = Image.open('imagekirsh.jpg')
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(partition12, image=img)
        panel.image = img
        panel.pack()
        buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
        buttonReset.pack()

def Moyenneur3():
        d=3
        img = cv.imread('image.jpg');
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        IR = img.copy()
        k = int((d - 1) / 2)
        for i in range(k, img.shape[0] - k):
            for j in range(k, img.shape[1] - k):
                s = 0
                for n in range(-k, k):
                    for m in range(-k, k):
                        s += img[i + n, j + m] / (d * d)
                IR[i, j] = s
                s = 0
        imag = Image.fromarray(IR)
        img1 = imag.save("imagemoyenneur3.jpg")
        img = Image.open('imagemoyenneur3.jpg')
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(partition12, image=img)
        panel.image = img
        panel.pack()
        buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
        buttonReset.pack()


def Moyenneur5():
        d=5
        img = cv.imread('image.jpg');
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        IR = img.copy()
        k = int((d - 1) / 2)
        for i in range(k, img.shape[0] - k):
            for j in range(k, img.shape[1] - k):
                s = 0
                for n in range(-k, k):
                    for m in range(-k, k):
                        s += img[i + n, j + m] / (d * d)
                IR[i, j] = s
                s = 0
        imag = Image.fromarray(IR)
        img1 = imag.save("imagemoyenneur5.jpg")
        img = Image.open('imagemoyenneur5.jpg')
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(partition12, image=img)
        panel.image = img
        panel.pack()
        buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
        buttonReset.pack()
        
        
def Median():
        d=3
        img = cv.imread('image.jpg');
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        IR = img.copy()
        h = int((d - 1) / 2)
        for i in range(h, img.shape[0] - h):
            for j in range(h, img.shape[1] - h):
                L = []
                for n in range(-h, h):
                    for m in range(-h, h):
                        L.append(IR[i + n, j + m])
                L.sort()
                IR[i, j] = L[h + 1]
                while len(L) > 0: L.pop()
             
        imag = Image.fromarray(IR)
        img1 = imag.save("imagemedian.jpg")
        img = Image.open('imagemedian.jpg')
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(partition12, image=img)
        panel.image = img
        panel.pack()
        buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
        buttonReset.pack()
        
        
def Gaussien():
        img = cv.imread('image.jpg');
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        IR = img.copy()
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                IR[i, j] =1/16 * img[i-1, j-1] + 1/8 * img[i - 1, j] +1/16 * img[i - 1, j+1] + 1/16  * img[i+1 , j - 1] + 1/8 * img[i+1, j ] + 1/16 * img[i+1, j +1] + 1/8 * img[i , j - 1] + 1/8 * img[i , j + 1]+ 1/4 * img[i , j ]
                
        imag = Image.fromarray(IR)
        imag = imag.convert("L")
        img1 = imag.save("imagegaussien.jpg")
        img = Image.open('imagegaussien.jpg')
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(partition12, image=img)
        panel.image = img
        panel.pack()
        buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
        buttonReset.pack()

          
        


def negatif():
    img = cv.imread('image.jpg');
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    IR = img.copy()
    IR = 255-IR
    imag = Image.fromarray(IR)
    img1 = imag.save("imagenegatif.jpg")
    img = Image.open('imagenegatif.jpg')
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(partition12, image=img)
    panel.image = img
    panel.pack()
    buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
    buttonReset.pack()
    
    
    
def rotation180():
    img = cv.imread('image.jpg')
    IR = cv.rotate(img,cv.ROTATE_180)
    imag = Image.fromarray(IR)
    img1 = imag.save("imagerotation180.jpg")
    img = Image.open('imagerotation180.jpg')
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(partition12, image=img)
    panel.image = img
    panel.pack()
    buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
    buttonReset.pack()

    
def rotation90():
    img = cv.imread('image.jpg')
    IR = cv.rotate(img,cv.ROTATE_90_CLOCKWISE)
    imag = Image.fromarray(IR)
    img1 = imag.save("imagerotation90.jpg")
    img = Image.open('imagerotation90.jpg')
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(partition12, image=img)
    panel.image = img
    panel.pack()
    buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
    buttonReset.pack()
    
def redimentionner25():
    img = cv.imread('image.jpg');
    width = int(img.shape[1]*25/100)
    height = int(img.shape[0]*25/100)
    dim=(width,height)
    IR = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    imag = Image.fromarray(IR)
    img1 = imag.save("imageredim25.jpg")
    img = Image.open('imageredim25.jpg')
    img = img.resize((60, 60), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(partition12, image=img)
    panel.image = img
    panel.pack()
    buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
    buttonReset.pack()
    
    
def redimentionner50():
    img = cv.imread('image.jpg');
    width = int(img.shape[1]*50/100)
    height = int(img.shape[0]*50/100)
    dim=(width,height)
    IR = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    imag = Image.fromarray(IR)
    img1 = imag.save("imageredim50.jpg")
    img = Image.open('imageredim50.jpg')
    img = img.resize((125, 125), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(partition12, image=img)
    panel.image = img
    panel.pack()
    buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
    buttonReset.pack()
    
def redimentionner75():
    img = cv.imread('image.jpg');
    width = int(img.shape[1]*75/100)
    height = int(img.shape[0]*75/100)
    dim=(width,height)
    IR = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    imag = Image.fromarray(IR)
    img1 = imag.save("imageredim75.jpg")
    img = Image.open('imageredim75.jpg')
    img = img.resize((190, 190), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(partition12, image=img)
    panel.image = img
    panel.pack()
    buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
    buttonReset.pack()
    
def Dilatation1(): 
    img = cv.imread('image.jpg');
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    n, m = img.shape
    IR = img.copy()
    img = img/255
    B = [[0, 1, 0],
          [1, 1, 1],
          [0, 1, 0]]
    
    for i in range(1, n-1):
        for j in range(1, m-1):
            if img[i-1, j] == 1 and img[i, j-1] == 1 and img[i, j] == 1 and img[i,j+1] == 1 and img[i+1,j] == 1:
                IR[i,j] = 255
            else:
                IR[i,j]=0

    imag = Image.fromarray(IR)
    img1 = imag.save("imagedilate.jpg")
    img = Image.open('imagedilate.jpg')
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(partition12, image=img)
    panel.image = img
    panel.pack()
    buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
    buttonReset.pack()

def Erosion1(): 
    img = cv.imread('image.jpg');
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    n, m = img.shape
    IR = img.copy()
    img = img/255
    B = [[0, 1, 0],
          [1, 1, 1],
          [0, 1, 0]]
    
    for i in range(1, n-1):
        for j in range(1, m-1):
            if img[i-1, j] == 1 or img[i, j-1] == 1 or img[i, j] == 1 or img[i,j+1] == 1 or img[i+1,j] == 1:
                IR[i,j] = 255
            else:
                IR[i,j]=0

    imag = Image.fromarray(IR)
    img1 = imag.save("imageerode.jpg")
    img = Image.open('imageerode.jpg')
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(partition12, image=img)
    panel.image = img
    panel.pack()
    buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
    buttonReset.pack()
    
def Ero(Image): 
    n, m = Image.shape
    a =np.zeros((n,m))
    Image = Image/255
    B = [[0, 1, 0],
          [1, 1, 1],
          [0, 1, 0]]
    for i in range(1, n-1):
        for j in range(1, m-1):
            if Image[i-1, j] == 1 and Image[i, j-1] == 1 and Image[i, j] == 1 and Image[i,j+1] == 1 and Image[i+1,j] == 1:
                a[i,j] = 255
            else:
                a[i,j]=0
    return a


def Dila(Image): 
    n, m = Image.shape
    a =np.zeros((n,m))
    Image = Image/255
    B = [[0, 1, 0],
          [1, 1, 1],
          [0, 1, 0]]
    for i in range(1, n-1):
        for j in range(1, m-1):
            if Image[i-1, j] == 1 or Image[i, j-1] == 1 or Image[i, j] == 1 or Image[i,j+1] == 1 or Image[i+1,j] == 1:
                a[i,j] = 255
            else:
                a[i,j]=0
    return a

def ouverture():
    img = cv.imread('image.jpg');
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image = Ero(img)
    IR = Dila(image)
    imag = Image.fromarray(IR)
    imag = imag.convert("L")
    img1 = imag.save("imageouverture.png")
    img = Image.open('imageouverture.png')
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(partition12, image=img)
    panel.image = img
    panel.pack()
    buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
    buttonReset.pack()
    
def fermeture():
    img = cv.imread('image.jpg');
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image = Dila(img)
    IR = Ero(image)
    imag = Image.fromarray(IR)
    imag = imag.convert("L")
    img1 = imag.save("imagefermeture.png")
    img = Image.open('imagefermeture.png')
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(partition12, image=img)
    panel.image = img
    panel.pack() 
    buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
    buttonReset.pack()

def enregistrer():
    messagebox.showinfo("TRAITEMENT D'IMAGE","chaque image enregistrer sera enregistrer avec le nom image+<operation_effectuer> dans le meme emlacement du projet")

def binarisation():
    img = cv.imread('image.jpg');
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    IR = img.copy()   
    l = int(img.shape[0])
    c = int(img.shape[1])
    
    for i in range(1, l):
        for j in range(1, c):
            if img[i][j] <= 128:
                IR[i][j] = 255
            else:
                IR[i][j] = 0
    imag = Image.fromarray(IR)
    img1 = imag.save("imagebinariser.jpg")
    img = Image.open('imagebinariser.jpg')
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(partition12, image=img)
    panel.image = img
    panel.pack()
    buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
    buttonReset.pack()
    
    
    
def otsu():
        img = cv.imread('image.jpg');
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        pixel_number = img.shape[0] * img.shape[1]
        mean_weigth = 1.0 / pixel_number
        his, bins = np.histogram(img, np.arange(0, 257))
        final_thresh = -1
        final_value = -1
        intensity_arr = np.arange(256)

        for t in bins[1:-1]:
            pcb = np.sum(his[:t])
            pcf = np.sum(his[t:])
            Wb = pcb * mean_weigth
            Wf = pcf * mean_weigth
            mub = np.sum(intensity_arr[:t] * his[:t]) / float(pcb)
            muf = np.sum(intensity_arr[t:] * his[t:]) / float(pcf)
            np.seterr(divide='ignore', invalid='ignore')
            value = Wb * Wf * (mub - muf) ** 2
            if value > final_value:
                final_thresh = t
                final_value = value

        IR = img.copy()
        IR[img > final_thresh] = 255
        IR[img < final_thresh] = 0
        imag = Image.fromarray(IR)
        img1 = imag.save("imageOTSU.jpg")
        img = Image.open('imageOTSU.jpg')
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(partition12, image=img)
        panel.image = img
        panel.pack()
        buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
        buttonReset.pack()
    
def histogramme():
    img = cv.imread('image.jpg');
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    IR = np.zeros(256)
    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[1]):
            v = img[i][j] + 1
            IR[v] = IR[v] + 1
    can=Figure(figsize = (4,4),dpi=70)
    plot1=can.add_subplot(111)
    plot1.plot('a','b',IR)
    canvas=FigureCanvasTkAgg(can,master =partition12)
    canvas.draw()
    canvas.get_tk_widget().pack()
    toolbar=NavigationToolbar2Tk(canvas, partition12)
    toolbar.update()
    canvas.get_tk_widget().pack()

          

def etirement():
        img = cv2.imread('image.jpg')
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        Max = np.max(img)
        Min = np.min(img)
        IR = img.copy()
        for i in range(1,img.shape[0]):
            for j in range(1,img.shape[0]):
                IR[i, j] = (255 / (Max - Min) * img[i, j] - Min)
        imag = Image.fromarray(IR)
        img1 = imag.save("imageETIRE.jpg")
        img = Image.open('imageETIRE.jpg')
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(partition12, image=img)
        panel.image = img
        panel.pack()
        buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
        buttonReset.pack()
    


def k_means():
    img = cv.imread('image.jpg');
    L = img.reshape((-1, 3))
    L = np.float32(L)
    C = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 3
    ret, labels, (centers) = cv2.kmeans(L, k, None, C, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    IR = centers[labels.flatten()]
    IR = IR.reshape(img.shape)
    imag = Image.fromarray(IR)
    imag = imag.convert("L")
    img1 = imag.save("imageK_means.jpg")
    img = Image.open('imageK_means.jpg')
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(partition12, image=img)
    panel.image = img
    panel.pack()
    buttonReset = Button(partition12, text = 'reset', command=lambda: panel.pack_forget())
    buttonReset.pack()
    
        
def sift():
    img = cv.imread('image.jpg')
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite('sift_keypoints.jpg',img)
    
    
    

    
    
    
        
#represente le grand menue
menubar = Menu(root)



#represente les sous menue
fichiermenu = Menu(menubar, tearoff=0)
fichiermenu.add_command(label="Ouvrir Image", command=open_img)
fichiermenu.add_command(label="Enregistrer", command=enregistrer)
fichiermenu.add_command(label="Quitter", command=root.destroy)
menubar.add_cascade(label="Fichier", menu=fichiermenu)


fichiermenu = Menu(menubar, tearoff=0)
fichiermenu.add_command(label="negatif", command=negatif)
fichiermenu.add_command(label="Rotation 90°", command=rotation90)
fichiermenu.add_command(label="Rotation 180°", command=rotation180)
fichiermenu.add_command(label="redimention 25%", command=redimentionner25)
fichiermenu.add_command(label="redimention 50%", command=redimentionner50)
fichiermenu.add_command(label="redimention 75%", command=redimentionner75)
menubar.add_cascade(label="foctionnalite", menu=fichiermenu)

fichiermenu = Menu(menubar, tearoff=0)
fichiermenu.add_command(label="binarisation", command=binarisation)
fichiermenu.add_command(label="OTSU", command=otsu)
fichiermenu.add_command(label="histogramme", command=histogramme)
fichiermenu.add_command(label="etirement", command=etirement)
menubar.add_cascade(label="operation", menu=fichiermenu)


fichiermenu = Menu(menubar, tearoff=0)
fichiermenu.add_command(label="gaussien", command=Gaussien)
fichiermenu.add_command(label="moyenneur 3*3", command=Moyenneur3)
fichiermenu.add_command(label="moyenneur 5*5", command=Moyenneur5)
fichiermenu.add_command(label="median", command=Median)
menubar.add_cascade(label="filtrage", menu=fichiermenu)


fichiermenu = Menu(menubar, tearoff=0)
fichiermenu.add_command(label="sobel", command=Sobel)
fichiermenu.add_command(label="gradien", command=gradient)
fichiermenu.add_command(label="kirsh", command=kirsh)
fichiermenu.add_command(label="robinson", command=robinson)
fichiermenu.add_command(label="laplacien", command=Laplacien)
menubar.add_cascade(label="contour", menu=fichiermenu)


fichiermenu = Menu(menubar, tearoff=0)
fichiermenu.add_command(label="Erosion", command=Erosion1)
fichiermenu.add_command(label="Dilatation", command=Dilatation1)
fichiermenu.add_command(label="ouverture", command=ouverture)
fichiermenu.add_command(label="fermeture", command=fermeture)
menubar.add_cascade(label="morphologie", menu=fichiermenu)

fichiermenu = Menu(menubar, tearoff=0)
fichiermenu.add_command(label="K-means", command=k_means)
menubar.add_cascade(label="segmentation", menu=fichiermenu)


fichiermenu = Menu(menubar, tearoff=0)
fichiermenu.add_command(label="SIFT", command=sift)
menubar.add_cascade(label="point d'interet", menu=fichiermenu)





root.config(menu=menubar)
root.title("projet traitement d'image")
root.geometry("1200x500")
root.mainloop()


