import os.path as osp
import os, cv2, shutil
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageDraw
import pandas as pd

'''
Position Name Description
1 2 3 4 5 6 7
8 9

1. Frame number: Indicate at which frame the object is present
2. Identity number: Each pedestrian trajectory is identified by a unique ID (−1 for detections)
3-6 Bounding box left top width height 
7. Confidence score: It acts as a flag whether the entry is to be considered (1) or ignored (0)
8. Class: Indicates the type of object annotated
9. Visibility: Visibility ratio, a number between 0 and 1 that says how much of that object is visible. Can be due to occlusion and due to image border cropping.

Label ID
Pedestrian 1
Person on vehicle 2
Car 3 
Bicycle 4 
Motorbike 5 
Non motorized vehicle 6 
Static person 7 
Distractor 8 
Occluder 9 
Occluder on the ground 10 
Occluder full 11 
Reflection 12 
Crowd 13
'''
fps=25
MOT20_class_ID = {
	1: 3,  # Pedestrian
	2: 3,  # Person on vehicle
	3: 0,  # Car
	4: 7,  # Bicycle
	5: 7,  # Motorbike
	6: 7,  # Non motorized vehicle
	7: 3,  # Static person
	8: -1,  # Distractor
	9: -1,  # Occluder
	10: -1,  # Occluder on the ground
	11: -1,  # Occluder full
	12: -1,  # Reflection
	13: 6  # Crowd
}
MOT20_ID_class = {
	3: 'Person',
	0: 'Car',
	7: 'Bike',
	6: 'Ignored'
}

data_root = 'dataset/MOT20'
ori_img_path= data_root+'/images/train'
target_txt = data_root+'/labels_with_ids/train'
target_img_path = ori_img_path

def convert_mot_to_jde(seq_root, label_root):
	shutil.rmtree(label_root, ignore_errors=True)
	os.makedirs(label_root, exist_ok=True)
	seqs = glob(seq_root+'/*/')
	seqs = [s.split('/')[-2] for s in seqs]

	tid_curr = 0
	tid_last = -1
	for seq in seqs:
		seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
		seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
		seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

		gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
		gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

		seq_label_root = osp.join(label_root, seq, 'img1')
		os.makedirs(seq_label_root, exist_ok=True)

		for fid, tid, x, y, w, h, confidence, label, Visibility in tqdm(gt):
			if confidence == 0:
				continue
			class_id = MOT20_class_ID[label]
			if class_id == -1:
				continue
			fid = int(fid)
			tid = int(tid)
			#keep tid unique in whole dataset, assuming tid monotonically increasing
			if not tid == tid_last:
				tid_curr += 1
				tid_last = tid
			x += w / 2
			y += h / 2
			label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
			label_str = f'{class_id:.0f} {tid_curr:.0f} {x / seq_width:.6f} {y / seq_height:.6f} {w / seq_width:.6f} {h / seq_height:.6f}\n'
			with open(label_fpath, 'a') as f:
				f.write(label_str)


def gen_train_val():
	tasks = glob(target_img_path+'/**/*.jpg', recursive=True)
	#split train val
	ratio = 0.9
	train_set = tasks[:int(len(tasks)*ratio)]
	val_set = tasks[int(len(tasks)*ratio):]

	with open(f'{data_root}/mot20.train', 'w') as f:
		for image in train_set:
			f.write(image+'\n')
	print(f'{data_root}/mot20.train generated')

	with open(f'{data_root}/mot20.val', 'w') as f:
		for image in val_set:
			f.write(image+'\n')
	print(f'{data_root}/mot20.val generated')

colors = np.random.randint(1, 255, (5000,3))

def render_img(img_path, save_path=None):
	img = Image.open(img_path)
	txt = img_path.replace('images', 'labels_with_ids').replace('.jpg', '.txt')
	w, h = img.width, img.height
	draw = ImageDraw.Draw(img, 'RGBA')
	anno = pd.read_csv(
		txt, names=['class', 'track_id', 'x_center', 'y_center', 'w_p', 'h_p'], sep=' ')
	anno['x1'] = (anno.x_center - anno.w_p/2) * w
	anno['y1'] = (anno.y_center - anno.h_p/2) * h
	anno['x2'] = (anno.x_center + anno.w_p/2) * w
	anno['y2'] = (anno.y_center + anno.h_p/2) * h
	for i, row in anno.iterrows():
		c = tuple(colors[int(row.track_id)])
		draw.rectangle([(row.x1, row.y1), (row.x2, row.y2)], outline=c, width=2)
		draw.text((row.x1, row.y1), f'{MOT20_ID_class[int(row["class"])]}({row.track_id})', c)
	if not save_path:
		return np.asarray(img)
	else:
		os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
		img.save(save_path)


def render_video(target_img_path):
	tasks = glob(target_img_path + '/*/')
	for task in tasks:
		imgs = glob(target_img_path + '/**/*.jpg', recursive=True)
		imgs = sorted(imgs)
		img1 = cv2.imread(imgs[0])
		height, width, _ = img1.shape
		temp_video = task[:-1] + '_temp.mp4'
		output_video_path = task[:-1] + '.mp4'
		if os.path.exists(output_video_path):
			print(f'Skip {output_video_path}')
			continue
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
		for img in tqdm(imgs, desc=f'rendering {output_video_path}'):
			img_arr = render_img(img)
			img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
			writer.write(img_arr)
		writer.release()

		cmd_str = f'ffmpeg -y -i {temp_video} -c:v libx264 -preset fast -x264-params crf=25 -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" -r {fps} {output_video_path}'
		print(f'>>>Running ffmpeg with cmd: {cmd_str}')
		os.system(cmd_str)
		os.remove(temp_video)

if __name__ == '__main__':
	convert_mot_to_jde(ori_img_path, target_txt)
	gen_train_val()
	render_video()
