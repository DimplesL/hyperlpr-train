import codecs
import os
import random
import shutil


input_dir = ''
train_dir = ''
val_dir = ''

if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(val_dir):
    os.mkdir(val_dir)

img_files = random.shuffle(os.listdir(input_dir))

all_num = len(img_files)
print(all_num)
train_img = img_files[:int(0.8 * all_num)]

val_img = img_files[int(0.8 * all_num):]

for img_file in train_img:
    if not img_file.endswith('.*g'):
        continue
    shutil.copy(os.path.join(input_dir, img_file), train_dir)
for img_file in val_img:
    if not img_file.endswith('.*g'):
        continue
    shutil.copy(os.path.join(input_dir, img_file), val_dir)
