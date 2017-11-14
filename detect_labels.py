"""
Label pictures and get bounding boxes coordinates (upper left and lower right).
Using mouse click we can get those coordinates.
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import cv2
import sys
import glob
from random import shuffle
import os
import tensorflow as tf
import imghdr

flags = tf.app.flags
flags.DEFINE_string('annotation_file', False, 'Path to Annotation File')
flags.DEFINE_string('images', 'images', 'Path to images to label')
FLAGS = flags.FLAGS

annotations = FLAGS.annotation_file
if annotations == False:
    print("must pass in annotations csv file to write")
    sys.exit(1)

try:
    fp = open(annotations)
except IOError:
    # If not exists, create the file
    fp = open(annotations, 'w+')
    fp.write('filename,width,height,class,xmin,ymin,xmax,ymax')
    fp.write("\n")

# List of already bounded pictures
with open(annotations) as f:
    already_labeled = [k.strip().split(',')[0] for k in f.readlines()]

# List of characters
character_count = glob.glob(os.path.join(os.getcwd(),'characters/*'))
map_characters = {}
for i in range(len(character_count)):
    map_characters[i] = character_count[i].rsplit("/",1)[1]
characters = list(map_characters.values())
shuffle(characters)

for char in characters:
    print('Working on %s' % char.replace('_', ' ').title())
    print('###################\n###################\n###################\n###################\n###################')
    # all labeled (just name, no bounding box) pictures of the character
    pics = glob.glob('%s/%s/*.*' % (FLAGS.images, char))
    shuffle(pics)
    i = 0
    for p in pics:
        try:
            print p
            pFormat = p.rsplit(".",1)[0] + "." + imghdr.what(p)
            os.rename(p,pFormat)
            print pFormat

        except TypeError:
            print 'Found incompatible file....Ignoring'
            continue

        if pFormat not in already_labeled and imghdr.what(pFormat) in ['jpeg','png']:
            try:
                im = cv2.imread(pFormat)
                height, width, channels = im.shape
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                ax = plt.gca()
                fig = plt.gcf()

                implot = ax.imshow(im)
                position = []
                def onclick(event):
                    """
                    If click, add the mouse position to the list.
                    Closing the plotted picture after 2 clicks (= 2 corners.)
                    Write the position for each picture into the text file.
                    """
                    if event.xdata != None and event.ydata != None:
                        position.append((event.xdata, event.ydata))
                        n_clicks = len(position)
                        if n_clicks == 2:
                            if position[0] == position[1]:
                                r = raw_input('Delete this picture[Y/n] ? ')
                                if r.lower() in ['yes','y']:
                                    os.remove(pFormat)
                                    plt.close()
                                    return

                            # pFormat = p.rsplit(".",1)[0] + "." + imghdr.what(p)
                            if position[0][0]<position[1][0] and position[0][1]<position[1][1]:
                                line = '{0},{1},{2},{3},{4},{5}'.format(pFormat,
                                    width,
                                    height,
                                    char,
                                    ','.join([str(int(k)) for k in position[0]]),
                                    ','.join([str(int(k)) for k in position[1]]))

                                # Open the annotations file to continue to write
                                target = open(annotations, 'a')
                                # Write picture and coordinates
                                target.write(line)
                                target.write("\n")
                                plt.close()

                            else:
                                print "Please make sure first click is top left and second click is bottom right"
                                plt.close()
                                return
                fig.canvas.set_window_title('%s pictures labeled' % i)
                cid = fig.canvas.mpl_connect('button_press_event', onclick)
                plt.show()
                i += 1
            # Common errors, just pass and close the plotting window
            except UnicodeDecodeError:
                plt.close()
                continue
            # When process is interrupted, juste print the number of labeled pictures
            except KeyboardInterrupt:
                plt.close()
                print('\nNumber of pictures with bounding box :')
                with open(annotations) as f:
                    already_labeled = [k.strip().split(',')[5] for k in f.readlines()]
                nb_pic_tot = {p:len([k for k in glob.glob('./%s/%s/*.*' % (FLAGS.images, p))]) for p in characters}

                print('\n'.join(['%s : %d/%d' % (char, nb, nb_pic_tot[char]) for char, nb in sorted(Counter(already_labeled).items(),
                                                     key =lambda x:x[1], reverse=True)]))
                t = np.sum(list(nb_pic_tot.values()))
                sys.exit("Total {}/{} ({}%)" .format(len(already_labeled),
                                                     t,
                                                     round(100*len(already_labeled)/t)))

    plt.close()
