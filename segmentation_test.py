import numpy as np
import os
import sys
import time
from skimage.io import imread, imsave
from segmentation import train_unary_model, segmentation

def load_data(path):
    if path[-1] != '/':
        path += '/'
    lines = open(path + 'gt.txt', 'r').readlines()
    imgs = []
    gt = []
    for line in lines:
        _, imname, gtname = (line[:-1] if line[-1] == '\n' else line).split(" ")
        imgs.append(imread(path+"images/"+imname,plugin="pil"))
        gtimg = imread(path+"gt/"+gtname,plugin="pil")
        if len(gtimg.shape)==3:
            gtimg = gtimg[:,:,0].reshape((gtimg.shape[:2]))
        gt.append(np.where(gtimg>0,np.ones(gtimg.shape,dtype=np.int),np.zeros(gtimg.shape,dtype=np.int)))
    return (imgs, gt)

def computeIoU(predicted, gt):
    if len(predicted)!=len(gt):
        raise "List lenghts do not match"
    sum = 0.
    for m_predicted, m_gt in zip(predicted, gt):
        intersection = np.count_nonzero(np.logical_and(m_predicted,m_gt))
        union = np.count_nonzero(np.logical_or(m_predicted, m_gt))
        sum+=float(intersection)/union
    return sum/len(gt)

if len(sys.argv) < 3:
    print("Usage: %s train_folder test_folder [-v vis_folder]" % sys.argv[0])
    sys.exit(1)
start_time = time.time()
train_dir = sys.argv[1]
test_dir = sys.argv[2]
visualisation_needed = (len(sys.argv) > 3) and (sys.argv[3] == '-v')
if visualisation_needed:
    if len(sys.argv)<5:
        print("Error: visualisation folder not set")
        sys.exit(1)
    else:
        vis_dir = sys.argv[4]

train_imgs, train_gt = load_data(train_dir)
unary_model = train_unary_model(train_imgs, train_gt)
del train_imgs, train_gt
test_imgs, test_gt = load_data(test_dir)
test_predicted = segmentation(unary_model, test_imgs)
result = computeIoU(test_predicted, test_gt)
print("Result: %.4f" % result)
if visualisation_needed:
    for mask_ind, mask in enumerate(test_predicted):
        imsave(os.path.join(vis_dir,"%03d.png"%mask_ind),mask.astype(np.uint8)*255)
end_time = time.time()
print("Running time: %.2f s (%.2f minutes)" %
      (round(end_time - start_time, 2), round((end_time - start_time) / 60, 2)))