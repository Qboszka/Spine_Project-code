import os
from load_data import get_data
import seaborn as sns
import matplotlib.pyplot as plt

labels = ['female', 'male']

#fetch Windows
#train_data = get_data('C:\\Workspace_studies\\Project_main\\Input\\train')
#val_data = get_data('C:\\Workspace_studies\\Project_main\\Input\\val')
#test_data = get_data('C:\\Workspace_studies\\Project_main\\Input\\test')

#fetch unified data Windows
train_data = get_data('C:\\Workspace_studies\\Project_main\\Input_unified\\train')
val_data = get_data('C:\\Workspace_studies\\Project_main\\Input_unified\\val')
test_data = get_data('C:\\Workspace_studies\\Project_main\\Input_unified\\test')

#fetch MacOS
#train_data = get_data('/Users/Qboszka/coding/Spine_Project/Input/train')
#val_data = get_data('/Users/Qboszka/coding/Spine_Project/Input/val')
#test_data = get_data('/Users/Qboszka/coding/Spine_Project/Input/test')

train = []
for i in train_data:
    if(i[1] == 0):
        train.append("female")
    else:
        train.append("male")
        
val = []
for i in val_data:
    if(i[1] == 0):
        val.append("female")
    else:
        val.append("male")

test = []
for i in test_data:
    if(i[1] == 0):
        test.append("female")
    else:
        test.append("male")

#visualize data in loop
plots = [train, val, test]
names = ["train_set", "validation_set", "test_set"]

num = 0
for plot in plots:
    sns.set_style('darkgrid')
    sns.countplot(plot)
    plt.title(names[num])
    plt.show()
    num += 1
    
plt.figure(figsize = (5, 5))
plt.imshow(train_data[1][0])
plt.title(labels[train_data[0][1]])
plt.show()

plt.figure(figsize = (5, 5))
plt.imshow(train_data[-1][0])
plt.title(labels[train_data[-1][1]])
plt.show()

print(len(train_data))
print(len(val_data))
print(len(test_data))