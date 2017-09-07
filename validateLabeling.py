from __future__ import division
import os
import sys
import numpy as np
import glob
import keras
import cv2
import matplotlib.pyplot as plt
import shutil

def labelPictures():
    if not os.path.isdir("./validationSanta"):
        os.makedirs("./validationSanta")
    if not os.path.isdir("./validationNoSanta"):
        os.makedirs("./validationNoSanta")

    folders = os.listdir("./validationImages")
    for folderName in folders:
        folderName = "validationImages/" + folderName
        if os.path.isdir(folderName):
            checkFolder = folderName + "/*.png"
            if glob.glob(checkFolder):
                images = glob.glob(checkFolder)

                if not os.path.isdir(folderName + '/Classified'):
                    os.makedirs(folderName + '/Classified')

                for imgName in images:
                    try:
                        img = cv2.imread("%s"%imgName)
                        imgLine = cv2.imread("%s"%imgName)
                        img = cv2.resize(img,(int(img.shape[1]/2), int(img.shape[0]/2)))
                        f = plt.ion()
                        plt.imshow(img)
                        plt.show()
                        where = raw_input("Where is Santa ? [No,Right,Left,Full]")

                        if where.lower() == "stop":
                            plt.close()
                            sys.exit()

                        elif where.lower() in ["left","l"]:
                            plt.close()
                            plt.imshow(img[:,:int(img.shape[1]/2)])
                            plt.show()
                            raw_input("Showing left part. Press any key to continue")
                            plt.close()
                            title = "./validationSanta/Santa_{:09d}.png".format(len(glob.glob('./validationSanta/*')))
                            cv2.imwrite(title,img[:,:int(img.shape[1]/2)])
                            print "Saved at %s "%title

                        elif where.lower() in ["right","r"]:
                            plt.close()
                            plt.imshow(img[:,int(img.shape[1]/2):])
                            plt.show()
                            raw_input("Showing right part. Press any key to continue")
                            plt.close()
                            title = "./validationSanta/santa_{:09d}.png".format(len(glob.glob('./validationSanta/*')))
                            cv2.imwrite(title,img[:,int(img.shape[1]/2):])
                            print "Saved at %s "%title

                        elif where.lower() in ["full","f"]:
                            raw_input("Showing full image. Press any key to continue")
                            plt.close()
                            title = "./validationSanta/Santa_{:09d}.png".format(len(glob.glob('./validationSanta/*')))
                            cv2.imwrite(title,img)
                            print "Saved at %s "%title

                        elif where.lower() in ["no","n"]:
                            plt.close()
                            title = "./validationNoSanta/noSanta_{:09d}.png".format(len(glob.glob('./validationNoSanta/*')))
                            cv2.imwrite(title,img)
                            print "Saved at %s "%title

                        else:
                            plt.close()
                            print 'Did not classify'
                            continue

                        title = folderName + "/Classified/classified_{:09d}.png".format(len(glob.glob('./%s/Classified/*'%folderName)))
                        cv2.imwrite(title,img)
                        os.unlink(imgName)

                    except Exception as e:
                        print e

if __name__ == '__main__':
    labelPictures()
