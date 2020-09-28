import copy
import os
import sys

import itertools
import pickle

import cv2
import numpy as np


def remove_duplicate_images(dup):
    out_list = []
    for image in dup:
        foundDupe = False
        for image2 in out_list:
            if np.array_equal(image, image2):
               foundDupe = True
               print("found duplicate")
        if not foundDupe:
            out_list.append(image)
    return out_list

def ssd(vec1, vec2):
    v_sum = 0
    for i in range(0,len(vec1)):
        diff = vec1[i] - vec2[i]
        v_sum += (diff * diff)
    return v_sum

def find_best_matches(des1, kp1, des2, kp2):
    best_matches = [] #list of pairs of keypoints that match
    values = []       #values of ssd of pairs
    for i in range(len(des1)):
        hund = len(des1) // 100
        if i % hund == 0:
            sys.stdout.write("\rFinding best match {0}%".format(round((i / len(des1) * 100))))
            sys.stdout.flush()
        #if i % 10 == 0:
            #print("best match for ", i)
        lowest_val = 999999999
        lowest_index = -1
        for j in range(len(des2)):
            temp_val = ssd(des1[i], des2[j])
            if temp_val < lowest_val:
                lowest_val = temp_val
                lowest_index = j
        best_matches.append((kp1[i].pt, kp2[lowest_index].pt))
        values.append(lowest_val)
    print("")
    return best_matches, values 

def closeEnough(calculated_point, expected_point, threshold):
    return (abs(calculated_point - expected_point) <= threshold).all()

def ransac(key_list1, key_list2, threshold=3):
    
    indexes = list(range(len(key_list1)))
    index_groups = list(itertools.combinations(indexes, 4))
    print("number of keys in 1", len(key_list1))
    print("number of keys in 2", len(key_list2))
    print("combinations: ", len(index_groups))

    mostInliers = 0
    bestCombo = None
    bestH = None

    for combo in index_groups:
        
        small_list1 = np.float32([key_list1[i] for i in combo])
        small_list2 = np.float32([key_list2[i] for i in combo])

        dst = []
        H = cv2.getPerspectiveTransform(small_list1, small_list2)
        transform_list = np.float32([key_list1])
        dst = cv2.perspectiveTransform(transform_list, H)

        inliers = 0
        outliers = 0

        for idx, pt in enumerate(dst[0]):
            actual_point = key_list2[idx]
            if closeEnough(pt, actual_point, threshold):
                inliers += 1
            else:
                outliers += 1
            
        if inliers > mostInliers:
            mostInliers = inliers
            bestH = H
            bestCombo = combo
    print("most inliers", mostInliers)
    print("best combo", bestCombo)
    return bestH

#takes two grayscale photos (gray & gray2) as input
def find_homography(gray, gray2, extension='yeh'):

    #use SIFT to find keypoints and feature vectors
    sift = cv2.xfeatures2d.SIFT_create()
    kp1,des1 = sift.detectAndCompute(gray, None)
    kp2,des2 = sift.detectAndCompute(gray2, None)

    print("found", len(kp1), "keypoints for first image")
    print("found", len(kp2), "keypoints for second image")

    #stores results to file for easier debugging

    #'''
    bm, ssd_vals = find_best_matches(des1, kp1, des2, kp2)
    f = open('store_bm' + extension + '.pckl', 'wb')
    pickle.dump(bm, f)

    f2 = open('store_ssd_vals' + extension + '.pckl', 'wb')
    pickle.dump(ssd_vals, f2)
    '''

    f = open('store_bm' + extension + '.pckl', 'rb')
    bm = pickle.load(f)

    f2 = open('store_ssd_vals' + extension + '.pckl', 'rb')
    ssd_vals = pickle.load(f2)
    #'''


    coords_1 = []
    coords_2 = []

    threshold = 1150
    sorted_list = sorted(ssd_vals)

    #remove values below threshold
    sorted_list = [i for i in sorted_list if i <= threshold]

    #push best keys 

    for i in range(len(sorted_list)):
        small_index = ssd_vals.index(sorted_list[i])
        img1_coord = bm[small_index][0]
        img2_coord = bm[small_index][1]

        coords_1.append(img1_coord)
        coords_2.append(img2_coord)
    return coords_1, coords_2

def merge_right(images, extension="", coords_1=None, coords_2=None):
    colorImages = copy.deepcopy(images)

    image1 = colorImages[0]
    image2 = colorImages[1]

    #convert images to grayscale
    gray= cv2.cvtColor(images[0],cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)
    
    if coords_1 == None or coords_2 == None:
        coords_1, coords_2 = find_homography(gray, gray2, extension)

    if coords_1[0][0] > coords_2[0][0]:
        print("swapping images")
        temp = coords_2
        coords_2 = coords_1
        coords_1 = temp

        imageTemp = image1
        image1 = image2
        image2 = imageTemp


    npkey1 = []
    for ke in coords_1:
        temp_key = []
        for dim in ke:
            temp_key.append(dim)
        npkey1.append(temp_key)


    npkey2 = []
    for ke in coords_2:
        temp_key = []
        for dim in ke:
            temp_key.append(dim)
        npkey2.append(temp_key)


    npkey1 = np.float32(npkey1)
    npkey2 = np.float32(npkey2)

    h,w = gray.shape

    #use RANSAC algorithm to find the best transition matrix
    trans = ransac(npkey1, npkey2, 0.5)
    
    #warped = right picture
    #border = left picture

    warped = cv2.warpPerspective(image1, trans, (w * 2,h))
    
    border = image2

    #place unaltered image over the warped image
    warped[0:border.shape[0], 0:border.shape[1]] = border
    return warped
    

def merge_left(images, extension="", coords_1=None, coords_2=None):
    colorImages = copy.deepcopy(images)

    image1 = colorImages[0]
    image2 = colorImages[1]

    #convert images to grayscale
    gray= cv2.cvtColor(images[0],cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)
   
    if coords_1 == None or coords_2 == None:
        coords_1, coords_2 = find_homography(gray, gray2, extension)

    if coords_1[0][0] < coords_2[0][0]:
        print("swapping images")
        temp = coords_2
        coords_2 = coords_1
        coords_1 = temp

        imageTemp = image1
        image1 = image2
        image2 = imageTemp


    npkey1 = []
    for ke in coords_1:
        temp_key = []
        for dim in ke:
            temp_key.append(dim)
        npkey1.append(temp_key)


    npkey2 = []
    for ke in coords_2:
        temp_key = []
        for dim in ke:
            temp_key.append(dim)
        npkey2.append(temp_key)


    npkey1 = np.float32(npkey1)
    npkey2 = np.float32(npkey2)

    h,w = gray.shape

    #adjust coordinates by image width so that they fit on the image space
    x_adjust = np.float32([[1, 0, w], [0, 1, 0], [0, 0, 1]])
    npkey2 = cv2.perspectiveTransform(np.float32([npkey2]), x_adjust)
    npkey2 = npkey2[0]


    #use RANSAC algorithm to find the best transition matrix
    trans = ransac(npkey1, npkey2, 0.5)
    
    #warped = right picture
    #border = left picture

    warped = cv2.warpPerspective(image1, trans, (w * 2,h))
    
    border = image2

    #place unaltered image over the warped image
    warped[0:border.shape[0], border.shape[1]:] = border
    return warped

def main():

    images = []
    dirPath = ""

    if len(sys.argv) == 2:
        dirPath = sys.argv[1]
    else:
        dirPath = "../data"

    filenames = []

    for filename in os.listdir(dirPath):
        filenames.append(filename)

    filenames.sort()

    for filename in filenames:
        print("Reading ", os.path.join(dirPath, filename))
        img = cv2.imread(os.path.join(dirPath, filename))
        images.append(img)
    
    #remove duplicates
    images = remove_duplicate_images(images)

    grayImages = [cv2.cvtColor(i,cv2.COLOR_BGR2GRAY) for i in images]
    output_image = None

    if len(images) == 2:    
        warped = merge_right(images, "a")
        output_image = warped
    if len(images) == 3:
        print("Comparing pair 1/3")
        group1_coords_1, group1_coords_2 = find_homography(grayImages[0], grayImages[1], "a")
        print("Comparing pair 2/3")
        group2_coords_1, group2_coords_2 = find_homography(grayImages[1], grayImages[2], "b")
        print("Comparing pair 3/3")
        group3_coords_1, group3_coords_2 = find_homography(grayImages[0], grayImages[2], "c")


        smallLen = min(len(group1_coords_1), len(group2_coords_1), len(group3_coords_1))
        middleImage = None
        leftImage = None
        rightImage = None
        deleteLine = None

        rightImage = None
        leftImage = None

        mr_coords = None
        ml_coords = None
        r_coords = None
        l_coords = None

        if len(group1_coords_1) == smallLen:
            print("Image 2 is the center")
            middleImage = 2
            deleteLine = 0
        if len(group2_coords_1) == smallLen:
            print("Image 0 is the center")
            middleImage = 0
            deleteLine = 1
        if len(group3_coords_1) == smallLen:
            print("Image 1 is the center")
            middleImage = 1
            deleteLine = 2

        if middleImage == 1:
            if group1_coords_1[0][0] > group1_coords_2[0][0]:
                mr_coords = group2_coords_1
                ml_coords = group1_coords_2
                r_coords = group2_coords_2
                l_coords = group1_coords_1
                leftImage = 0
                rightImage = 2
            else:
                mr_coords = group1_coords_2
                ml_coords = group2_coords_1
                r_coords = group1_coords_1
                l_coords = group2_coords_2
                leftImage = 2
                rightImage = 0
        if middleImage == 0:
            if group1_coords_1[0][0] > group1_coords_2[0][0]:
                mr_coords = group1_coords_1
                ml_coords = group3_coords_1
                r_coords = group1_coords_2
                l_coords = group3_coords_2
                leftImage = 2
                rightImage = 1
            else:
                mr_coords = group3_coords_1
                ml_coords = group1_coords_1
                r_coords = group3_coords_2
                l_coords = group1_coords_2
                leftImage = 1
                rightImage = 2
        if middleImage == 2:
            if group2_coords_1[0][0] > group2_coords_2[0][0]:
                mr_coords = group3_coords_2
                ml_coords = group2_coords_2
                r_coords = group3_coords_1
                l_coords = group2_coords_1
                leftImage = 1
                rightImage = 0
            else:
                mr_coords = group2_coords_2
                ml_coords = group3_coords_2
                r_coords = group2_coords_1
                l_coords = group3_coords_1
                rightImage = 1
                leftImage = 0

        
        leftSide = merge_left([images[leftImage], images[middleImage]], None, l_coords, ml_coords)
        rightSide = merge_right([images[middleImage], images[rightImage]], None, mr_coords, r_coords)
        h,w,d = leftSide.shape
        single_w = w // 2

        finalImage = np.concatenate((leftSide, rightSide[:,single_w:]), axis=1)
        output_image = finalImage

        
        

    cv2.imwrite(os.path.join(dirPath, "panorama.jpg"), output_image)

if __name__ == "__main__":
    main()
