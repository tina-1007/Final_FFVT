import os

root = './train/'
class_list = ""
train_list = ""
valid_list = ""

class_names = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

for i, c in enumerate(class_names):
	class_list += '{} {}\n'.format(i, c)
	images = os.listdir(root + c)
	train_len = int(len(images) * 0.8)
	for img in images[:train_len]:
		train_list += '{} {}\n'.format(img, i)
	for img in images[train_len:]:
		valid_list += '{} {}\n'.format(img, i)

# class names (classes.txt)
# <class_id> <class_name>
class_file = open('classes.txt', 'w')
class_file.write(class_list)
class_file.close()

# class labels (image_class_labels.txt)
# <image_name> <class_id>
image_file = open('train_image_labels82.txt', 'w')
image_file.write(train_list)
image_file.close()

image_file = open('valid_image_labels82.txt', 'w')
image_file.write(valid_list)
image_file.close()

test_dir = ['test_stg1/', 'test_stg2/']
test_list = ""

for d in test_dir:
	imgs = os.listdir(d)
	for img in imgs:
		test_list += '{}{}\n'.format(d, img)
test_file = open('test_order.txt', 'w')
test_file.write(test_list)

# image file (images.txt)
# <image_id> <image_name>

# Train/test split (train_test_split.txt)
# <image_> <is_training_image> (1/0)