# # Creating Train / Val / Test folders (One time use)
import os, shutil, sys, random
import numpy as np

#root_dir = 'C:\\Workspace_studies\\Project_main\\Input\\' # data root path
root_dir = '/Users/Qboszka/coding/Spine_Project/Input/'
classes_dir = ['female', 'male'] #total labels

val_ratio = 0.10
test_ratio = 0.10

for cls in classes_dir:
    os.makedirs(root_dir + 'train/' + cls)
    os.makedirs(root_dir + 'val/' + cls)
    os.makedirs(root_dir + 'test/' + cls)

# Creating partitions of the data after shuffeling
for cls in classes_dir:
    src = root_dir + cls # Folder to copy images from

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames), 
    [int(len(allFileNames) * (1 - (val_ratio + test_ratio))), 
    int(len(allFileNames) * (1 - test_ratio))])

    train_FileNames = [src + '/' + name for name in train_FileNames.tolist()]
    val_FileNames = [src + '/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src + '/' + name for name in test_FileNames.tolist()]
    
    print('Class name: ' + cls)
    print('\n')
    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))
    print('\n')

    # Copy-pasting images

    for name in train_FileNames:
        shutil.copy(name, root_dir + 'train/' + cls)

    for name in val_FileNames:
        shutil.copy(name, root_dir + 'val/' + cls)

    for name in test_FileNames:
        shutil.copy(name, root_dir + 'test/' + cls)