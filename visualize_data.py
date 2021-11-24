from load_data import get_data
import seaborn as sns
import matplotlib.pyplot as plt

labels = ['female', 'male']
img_size = 224

#fetch data
train = get_data('C:\\Workspace_studies\\Project_main\\Input\\train')
val = get_data('C:\\Workspace_studies\\Project_main\\Input\\val')
test = get_data('C:\\Workspace_studies\\Project_main\\Input\\test')

l = []
for i in train:
    if(i[1] == 0):
        l.append("female")
    else:
        l.append("male")

sns.set_style('darkgrid')
sns.countplot(l)
#plt.show()

plt.figure(figsize = (5,5))
plt.imshow(train[1][0])
plt.title(labels[train[0][1]])
plt.show()
