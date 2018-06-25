import os
import sys
import cv2
from itertools import islice
from xml.dom.minidom import Document
import logging

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

xmlpath_new='Annotations/'
foldername='WiderPerson'

imgdir = 'driven_train/'
with open('annotations.txt','r') as f:
    lines = f.readlines()

information = [i.split() for i in lines]

def insertObject(doc, datas):
    obj = doc.createElement('object')
    name = doc.createElement('name')
    name.appendChild(doc.createTextNode('person'))
    obj.appendChild(name)
    pose = doc.createElement('pose')
    pose.appendChild(doc.createTextNode('Unspecified'))
    obj.appendChild(pose)
    truncated = doc.createElement('truncated')
    truncated.appendChild(doc.createTextNode(str(0)))
    obj.appendChild(truncated)
    difficult = doc.createElement('difficult')
    difficult.appendChild(doc.createTextNode(str(0)))
    obj.appendChild(difficult)
    bndbox = doc.createElement('bndbox')

    xmin = doc.createElement('xmin')
    xmin.appendChild(doc.createTextNode(datas[1]))
    bndbox.appendChild(xmin)

    ymin = doc.createElement('ymin')
    ymin.appendChild(doc.createTextNode(datas[2]))
    bndbox.appendChild(ymin)

    xmax = doc.createElement('xmax')
    xmax.appendChild(doc.createTextNode(str((int(datas[1])+int(datas[3])))))
    bndbox.appendChild(xmax)

    ymax = doc.createElement('ymax')
    ymax.appendChild(doc.createTextNode(str((int(datas[2])+int(datas[4])))))
    bndbox.appendChild(ymax)

    obj.appendChild(bndbox)
    return obj

def create():
    for item in information:
        imgname = imgdir+item[0]
        if os.path.exists(imgname):
            # raise IOError,"Image {} not exists!".format(imgname)
            img = cv2.imread(imgname)
            logger.info('Read {} !'.format(imgname[-12:]))
            h,w,c = img.shape
            num_bboxs = (len(item)-1)/5
            bboxs = []
            for num in range(num_bboxs):
                bbox = []
                bbox.append(item[1+5*num])
                bbox.append(item[2+5*num])
                bbox.append(item[3+5*num])
                bbox.append(item[4+5*num])
                bbox.append(item[5+5*num])
                bboxs.append(bbox)
            xmlName = item[0].replace('.jpg', '.xml')
            f = open(xmlpath_new + xmlName, "w")
            doc = Document()
            annotation = doc.createElement('annotation')
            doc.appendChild(annotation)

            folder = doc.createElement('folder')
            folder.appendChild(doc.createTextNode(foldername))
            annotation.appendChild(folder)

            filename = doc.createElement('filename')
            filename.appendChild(doc.createTextNode(item[0]))
            annotation.appendChild(filename)

            source = doc.createElement('source')
            database = doc.createElement('database')
            database.appendChild(doc.createTextNode('The WiderPerson Database'))
            source.appendChild(database)
            source_annotation = doc.createElement('annotation')
            source_annotation.appendChild(doc.createTextNode(foldername))
            source.appendChild(source_annotation)
            image = doc.createElement('image')
            image.appendChild(doc.createTextNode('flickr'))
            source.appendChild(image)
            flickrid = doc.createElement('flickrid')
            flickrid.appendChild(doc.createTextNode('NULL'))
            source.appendChild(flickrid)
            annotation.appendChild(source)

            owner = doc.createElement('owner')
            flickrid = doc.createElement('flickrid')
            flickrid.appendChild(doc.createTextNode('NULL'))
            owner.appendChild(flickrid)
            name = doc.createElement('name')
            name.appendChild(doc.createTextNode('peddetection'))
            owner.appendChild(name)
            annotation.appendChild(owner)

            size = doc.createElement('size')
            width = doc.createElement('width')
            width.appendChild(doc.createTextNode(str(w)))
            size.appendChild(width)
            height = doc.createElement('height')
            height.appendChild(doc.createTextNode(str(h)))
            size.appendChild(height)
            depth = doc.createElement('depth')
            depth.appendChild(doc.createTextNode(str(c)))
            size.appendChild(depth)
            annotation.appendChild(size)

            segmented = doc.createElement('segmented')
            segmented.appendChild(doc.createTextNode(str(0)))
            annotation.appendChild(segmented)
            #annotation.appendChild(insertObject(doc, datas))
            for bbox in bboxs:
                annotation.appendChild(insertObject(doc, bbox))
            try:
                f.write(doc.toprettyxml(indent = '    '))
                f.close()
            except:
                pass


if __name__ == '__main__':
    create()
