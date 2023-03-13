import os

# check, if file exists, make link
def check_link(in_dir, basename, out_dir):
	in_file = os.path.join(in_dir, basename)
	if os.path.exists(in_file):
		link_file = os.path.join(out_dir, basename)
		rel_link = os.path.relpath(in_file, out_dir)  # from out_dir to in_file
		os.symlink(rel_link, link_file)
 
def add_splits(data_path):
	images_path = os.path.join(data_path, 'Img/img_align_celeba')
	train_dir = os.path.join(data_path, 'splits', 'train')
	valid_dir = os.path.join(data_path, 'splits', 'valid')
	test_dir = os.path.join(data_path, 'splits', 'test')
	if not os.path.exists(train_dir):
		os.makedirs(train_dir)
	if not os.path.exists(valid_dir):
		os.makedirs(valid_dir)
	if not os.path.exists(test_dir):
		os.makedirs(test_dir)
 
	# these constants based on the standard CelebA splits
	NUM_EXAMPLES = 202599
	TRAIN_STOP = 162770
	VALID_STOP = 182637
 
	for i in range(0, TRAIN_STOP):
		basename = "{:06d}.jpg".format(i+1)
		check_link(images_path, basename, train_dir)
	for i in range(TRAIN_STOP, VALID_STOP):
		basename = "{:06d}.jpg".format(i+1)
		check_link(images_path, basename, valid_dir)
	for i in range(VALID_STOP, NUM_EXAMPLES):
		basename = "{:06d}.jpg".format(i+1)
		check_link(images_path, basename, test_dir)


def public_private_splits(data_path):
	file_path = os.path.join(data_path, 'identity_CelebA.txt')
	public_train_path = os.path.join(data_path,'splits_11','identity_public_train.txt')
	public_test_path = os.path.join(data_path,'splits_11','identity_public_test.txt')
	private_train_iden_path = os.path.join(data_path,'splits_11','identity_private_train.txt')
	private_test_iden_path = os.path.join(data_path,'splits_11','identity_private_test.txt')
	images_path = os.path.join(data_path, 'img_align_celeba')
	public_train_dir = os.path.join(data_path, 'splits_11', 'public', 'train')
	public_test_dir  = os.path.join(data_path, 'splits_11', 'public', 'test')
	private_train_dir = os.path.join(data_path, 'splits_11', 'private', 'train')
	private_test_dir = os.path.join(data_path, 'splits_11', 'private', 'test')

	if not os.path.exists(public_train_dir):
		os.makedirs(public_train_dir)
	if not os.path.exists(public_test_dir):
		os.makedirs(public_test_dir)
	if not os.path.exists(private_train_dir):
		os.makedirs(private_train_dir)
	if not os.path.exists(private_test_dir):
		os.makedirs(private_test_dir)

	TRAIN_STOP = int(21104 * 0.9)
	cnt_public = 0
	cnt_private = 0

	f = open(file_path, "r")
	f_public_train = open(public_train_path, "w")
	f_public_test = open(public_test_path, "w")
	f_private_train = open(private_train_iden_path, "w")
	f_private_test = open(private_test_iden_path, "w")

	for line in f.readlines():
		img_name, iden = line.strip().split(' ')
	
		if 0 < int(iden) < 1001:
			if cnt_private < TRAIN_STOP:
				check_link(images_path, img_name, private_train_dir)
				f_private_train.write(line)
			else:
				check_link(images_path, img_name, private_test_dir)
				f_private_test.write(line)
			cnt_private += 1

		if 1000 < int(iden) < 2001:
			if cnt_public < TRAIN_STOP:
				check_link(images_path, img_name, public_train_dir)
				f_public_train.write(line)
			else:
				check_link(images_path, img_name, public_test_dir)
				f_public_test.write(line)
			cnt_public += 1
			# check_link(images_path, img_name, public_dir)
			# f_public.write(line)
			

	# print('number of images in public dir is:'.format())

def combine(f, foldername, base_path):

	img_path = os.path.join(base_path, foldername)
	for i in range(1000):
		basename = "{:05d}.png".format(i+int(foldername))
		check_link(img_path, basename, base_path)
		print("creat symlink for image ", basename)
		f.write(basename)
		f.write('\n')
	
def getListOfFiles(f, dirName):
	# create a list of file and sub directories 
	# names in the given directory 
	listOfFile = os.listdir(dirName)
	allFiles = list()
	# Iterate over all the entries
	for entry in listOfFile:
		# Create full path
		fullPath = os.path.join(dirName, entry)
		# If entry is a directory then get the list of files in this directory 
		if os.path.isdir(fullPath):
			allFiles = allFiles + getListOfFiles(fullPath)
		else:
			allFiles.append(fullPath)
			f.write(entry)
			f.write('\n')
				
	return allFiles


def cifar_split(base_path, save_path):
	public_path = os.path.join(save_path, 'ganset.txt')
	private_train = os.path.join(save_path, 'train.txt')
	private_test = os.path.join(save_path, 'test.txt')
	f_public = open(public_path, "w")
	f_private_train = open(private_train, "w")
	f_private_test = open(private_test, "w")

	# train file 50,000
	listOfFile = os.listdir(base_path)
	for entry in listOfFile:
		# import pdb; pdb.set_trace()
		# print(entry)
		if entry.endswith('.png'):
			img_name, label = os.path.splitext(entry)[0].strip().split('_')
			if int(img_name) <= 50000:
				# train
				if int(label) in [0, 1, 2, 3, 4]:
					# private
					f_private_train.write(entry + ' ' + str(label))
					f_private_train.write('\n')

				else:
					# public
					f_public.write(entry)
					f_public.write('\n')

			else:
				# test
				if int(label) in [0, 1, 2, 3, 4]:
					# private
					f_private_test.write(entry + ' ' + str(label))
					f_private_test.write('\n')

				else:
					# public
					f_public.write(entry)
					f_public.write('\n')


def cifar_eval(base_path, save_path):
	train = os.path.join(save_path, 'train_eval.txt')
	test = os.path.join(save_path, 'test_eval.txt')
	f_train = open(train, "w")
	f_test = open(test, 'w')

	# train file 50,000
	listOfFile = os.listdir(base_path)
	for entry in listOfFile:
		# import pdb; pdb.set_trace()
		# print(entry)
		if entry.endswith('.png'):
			img_name, label = os.path.splitext(entry)[0].strip().split('_')
			if int(img_name) <= 50000:
				# train
				f_train.write(entry + ' ' + str(label))
				f_train.write('\n')

			else:
				# test
				f_test.write(entry + ' ' + str(label))
				f_test.write('\n')
 
if __name__ == '__main__':
	base_path = '/home/chensi/data/CIFAR_imgs'
	save_path = './cifar_splits'
	os.makedirs(save_path, exist_ok=True)
	cifar_split(base_path, save_path)
	cifar_eval(base_path, save_path)

	# f.close()