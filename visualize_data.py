from load_data import get_data
import seaborn as sns
import matplotlib.pyplot as plt

labels = ['female', 'male']
img_size = 224

#fetch data

train = get_data('C:\\Workspace_studies\\Project_main\\Input\\train')
val = get_data('C:\\Workspace_studies\\Project_main\\Input\\val')
test = get_data('C:\\Workspace_studies\\Project_main\\Input\\test')

#train = get_data('/Users/Qboszka/coding/Spine_Project/Input/train')
#val = get_data('/Users/Qboszka/coding/Spine_Project/Input/val')
#test = get_data('/Users/Qboszka/coding/Spine_Project/Input/test')

tr = []
for i in train:
    if(i[1] == 0):
        tr.append("female")
    else:
        tr.append("male")
        
v = []
for i in val:
    if(i[1] == 0):
        v.append("female")
    else:
        v.append("male")

t = []
for i in test:
    if(i[1] == 0):
        t.append("female")
    else:
        t.append("male")

#visualize data in loop
plots = [tr, v, t]

for plot in plots:
    sns.set_style('darkgrid')
    sns.countplot(plot)
    plt.show()
    
plt.figure(figsize = (5,5))
plt.imshow(train[1][0])
plt.title(labels[train[0][1]])
plt.show()