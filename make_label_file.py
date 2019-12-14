import codecs
import os


train_dir = ''
val_dir = ''
train_label = 'train_labels.txt'

fi = codecs.open(train_label, 'w', encoding='utf-8')
img_files = os.listdir(train_dir)

for img_file in img_files:
    if not img_file.endswith('g'):
        continue

    img_label = img_file.split('.')[0].split('_')[-1]

    fi.write('{}:{}'.format(img_file, img_label))
    fi.write('\n')

fi.close()
val_label = 'val_labels.txt'

fi = codecs.open(val_label, 'w', encoding='utf-8')
img_files = os.listdir(val_dir)

for img_file in img_files:
    if not img_file.endswith('g'):
        continue

    img_label = img_file.split('.')[0].split('_')[-1]

    fi.write('{}:{}'.format(img_file, img_label))
    fi.write('\n')

fi.close()