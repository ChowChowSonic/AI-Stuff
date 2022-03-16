import numpy as np
import os
import tensorflow as tf
import shutil
from PIL import Image
from autoEncoder import VarEncoder as Encoder

#Take all images in the dataset and move them to a single folder
'''for root, dirs, files in os.walk("dogs", topdown=False):
    for dir in dirs:
        base = os.path.join(root, dir)
        print(base)
        for _,_,files2 in os.walk(base):
            for file in files2:
                shutil.move(os.path.join(base, file), os.path.join("dogs", file))'''

#arr = np.empty((1080, 1960,3), dtype=np.byte)
#arr = np.append(arr, np.empty([2000-arr.shape[0],*arr.shape[1:]], dtype=np.byte),axis=0)
#arr = np.append(arr, np.empty([arr.shape[0],2000-arr.shape[1],arr.shape[-1]], dtype = np.byte),axis=1)
#print(np.shape(arr))
'''shapes = {}
for _,_, imgs in os.walk("dogs"):
        for dog in imgs:
            img = np.asarray(Image.open("dogs\\"+dog))
            #if(img.shape == [500,800]):
            if(shapes.get(img.shape) == None):
                shapes[img.shape] = 1
            else:
                shapes[img.shape] = shapes.get(img.shape)+1
print("done!")
max = 0
maxval = ()
for value in shapes.keys():
    if shapes[value] >= max:
        max = shapes[value]
        maxval = value
print(max, maxval)'''

def load_data(n = 0, directory = "dogs", size = (375,500,3), grayscale = False):
    if n == 0:
        for _,_, imgs in os.walk(directory):
            n = len(imgs)
            break; 
    data = []
    size2 = np.asarray(size); 
    if size2[0]%2 ==1:
        size2[0]=size2[0]+1
    if size2[1]%2 ==1:
        size2[1]=size2[1]+1
    for _,_, imgs in os.walk(directory):
        isdone = False
        for dog in imgs:
            img = np.asarray(Image.open(directory+"\\"+dog), dtype=np.byte)#Insert image name(s) here
            if img.shape == size:
                img = np.append(img, np.empty([size2[0]-img.shape[0], *img.shape[1:]],dtype=np.byte), axis=0)
                img = np.append(img, np.empty([img.shape[0], size2[1]-img.shape[1], img.shape[-1]],dtype=np.byte), axis=1)
                data.append(img)
            if(len(data) > n):
                isdone = True
                break
        if isdone == True:
            break
    data = np.array(data, dtype=np.byte)
    if grayscale == True:
        avg = np.average(data, axis=3)[..., np.newaxis]
        #print(np.shape(avg), np.shape(data))
        data = avg
    print("Data Loaded:", len(data), "images prepared.")
    return data

epochs = 100
dir="pokemon\\all"
data = load_data(directory=dir, size=(120,120,4), grayscale=True)
encoder = Encoder([120,120,1], 
        conv_filters=(4,4,2),
        conv_kernels=(3,3,3),
        conv_strides=(1,2,1),
        latent_space_dim=1024)
encoder.compile(0.0001)
encoder.train(data, 32, epochs)
name = dir.replace("\\", ".")+str(len(data))+"_"+str(len(encoder.conv_filters))+"cv_"+str(encoder.conv_kernels[0])+"k_"+str(encoder.latent_space_dim)+"d"
print("saving under name:",name)
encoder.save(name)

encoder = Encoder.load("pokemon.all522_3cv_3k_1024d")
imgs, latent = encoder.reconstruct(data[0:6])
imgs = np.asarray(imgs*256, dtype=np.uint8)
imgs = np.pad(imgs, pad_width=((0,0),(0,0),(0,0),(0,2)), mode="minimum")

#randomlatent = np.random.randint(low=0, high=40, size=[6,1024])
#imgs = encoder.generatefromLatentReps(randomlatent)
#imgs = np.asarray(imgs*256, dtype=np.uint8)
#imgs = np.pad(imgs, pad_width=((0,0),(0,0),(0,0),(0,2)), mode="minimum")

for i in range(len(imgs)):
    im = Image.fromarray(imgs[i])
    im.save("Generated/"+name+" ("+str(i)+").jpg")

data = np.pad(data, pad_width=((0,0),(0,0),(0,0),(0,2)), mode="minimum")
data = data.astype(np.uint8)
for i in range(len(imgs)):
    im = Image.fromarray(data[i])
    im.save("Generated/"+"original ("+str(i)+").jpg")