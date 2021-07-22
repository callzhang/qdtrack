import pandas as pd
import xmltodict
from glob import glob
import os, cv2
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np

# xml_file = 'dataset/DETRAC/DETRAC-Train-Annotations-XML-v3/MVI_20011_v3.xml'
# data = pd.read_xml(xml_file, xpath='.//frame')

data_root = 'dataset/DETRAC'
target_txt_path = data_root+'/train/labels_with_ids'
target_img_path = data_root+'/train/images'
ori_img_path_train = data_root+'/Insight-MVT_Annotation_Train'
ori_img_path_test = data_root+'/Insight-MVT_Annotation_Test'
kitti_class_ID = {
	'Car': 0,
	'Van': 1,
	'Truck': 2,
	'Pedestrian': 3,
	'Person_sitting': 3,
	'Cyclist': 3,
	'Tram': 4,
	'Misc': 5,
	'DontCare': 6,
	'Person': 3
}
ID_class = {v: k for k, v in kitti_class_ID.items()}

height, width = 540, 960
fps = 25

DETRAC_class_ID = {
	'Sedan': 0,
	'Suv': 0,
	'Taxi': 0,
	'Van': 1,
	'Truck-Box-Large': 2,
	'Hatchback': 0,
	'Bus': 4,
	'Police': 0,
	'MiniVan': 1,
	'Truck-Box-Med': 2,
	'Truck-Util': 2,
	'Truck-Pickup': 2,
	'Truck-Flatbed': 2,
	'car':0,
	'van': 1,
	'bus': 4,
	'others': 5,
}
DETRAC_ID_class = {v: k for k, v in DETRAC_class_ID.items()}

## parse xml
def convert_xml():
	all_xml = glob('dataset/DETRAC/DETRAC-Train-Annotations-XML-v3/*.xml')
	all_xml += glob('dataset/DETRAC/DETRAC-Test-Annotations-XML/*.xml')
	for xml_file in tqdm(all_xml, desc='Converting XML to CSV'):
		csv_file = xml_file.replace('.xml', '.csv')
		if not os.path.exists(csv_file):
			with open(xml_file, 'r') as f:
				data = xmltodict.parse(f.read())

			detections = []
			for frame in data['sequence']['frame']:
				frame_id = int(frame['@num'])
				targets = frame['target_list']['target']
				if type(targets) is not list:
					targets = [targets]
				for target in targets:
					tid = int(target['@id'])
					box = target['box']
					attributes = target['attribute']
					vtype = attributes['@vehicle_type']
					occlusion = True if 'occlusion' in target else False
					box['id'] = tid
					box['type'] = vtype
					box['occlusion'] = occlusion
					box['frame'] = frame_id
					detections.append(box)

			detections = pd.DataFrame(detections).set_index('frame')
			detections.to_csv(csv_file)


# convert to FairMOT
def convert_cst_to_mot():
	all_csv = glob('dataset/DETRAC/DETRAC-Train-Annotations-XML-v3/*.csv')
	all_csv += glob('dataset/DETRAC/DETRAC-Test-Annotations-XML/*.csv')
	all_results = []
	for csv_path in all_csv:
		data = pd.read_csv(csv_path)
		all_results.append(data)
	all_results = pd.concat(all_results)
	all_types = all_results.type.value_counts()
	print(all_types)

	last_max_tid = 0
	for csv_path in tqdm(all_csv, desc='converting CSV to MOT'):
		data = pd.read_csv(csv_path)
		data.type = data.type.apply(lambda x: DETRAC_class_ID[x])
		data['x_center'] = (data['@left'] + data['@width']/2)/width
		data['y_center'] = (data['@top'] + data['@height']/2)/height
		data['w_p'] = data['@width']/width
		data['h_p'] = data['@height']/height
		task = csv_path.split('/')[-1].replace('_v3.csv', '').replace('.csv', '')
		os.makedirs(f'{target_img_path}/{task}', exist_ok=True)
		os.makedirs(f'{target_txt_path}/{task}', exist_ok=True)
		# max tid for this seq
		data = data.query('id != -1')
		max_tid = data.id.max()+1
		data.id += last_max_tid  # add last_max_tid as base
		print(f'processing seq: {task} with max tid: {max_tid} starting with base: {last_max_tid}')
		last_max_tid += max_tid  # for next seq
		for f in range(1, data.frame.max()+1):
			txt_path = f'{target_txt_path}/{task}/{f:06}.txt'
			img_path = f'{target_img_path}/{task}/{f:06}.jpg'
			if not os.path.exists(img_path):
				try:
					img = Image.open(f'{ori_img_path_train}/{task}/img{f:05}.jpg')
				except OSError:
					img = Image.open(f'{ori_img_path_test}/{task}/img{f:05}.jpg')
				assert width == img.width and height == img.height
				img.save(img_path)
			txt_df = data.query('frame == @f')[['type', 'id', 'x_center', 'y_center', 'w_p', 'h_p']]
			# if not os.path.exists(txt_path):
			txt_df.to_csv(txt_path, header=False, index=False, sep=' ')


def gen_train_val():
	tasks = glob(target_img_path+'/*')
	tasks = [t.split('/')[-1] for t in tasks]
	#split train val
	ratio = 0.9
	train_set = tasks[:int(len(tasks)*ratio)]
	val_set = tasks[int(len(tasks)*ratio):]

	with open(f'{data_root}/DETRAC.train', 'w') as f:
		for task in train_set:
			images = glob(f'{target_img_path}/{task}/*.jpg')
			for t in images:
				f.write(t+'\n')
	print(f'{data_root}/DETRAC.train generated')

	with open(f'{data_root}/DETRAC.val', 'w') as f:
		for task in val_set:
			images = glob(f'{target_img_path}/{task}/*.jpg')
			for t in images:
				f.write(t+'\n')
	print(f'{data_root}/DETRAC.val generated')

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
		draw.rectangle([(row.x1, row.y1), (row.x2, row.y2)])
		draw.text((row.x1, row.y1), f'{DETRAC_ID_class[row["class"]]}({row.track_id})')
	if not save_path:
		return np.asarray(img)
	else:
		os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
		img.save(save_path)


def render_video():
	tasks = glob(target_img_path + '/*/')
	for task in tasks:
		temp_video = task[:-1] + '_temp.mp4'
		output_video_path = task[:-1] + '.mp4'
		if os.path.exists(output_video_path):
			print(f'Skip {output_video_path}')
			continue
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
		imgs = glob(f'{task}/*.jpg')
		for img in tqdm(imgs, desc=f'rendering {task}'):
			img_arr = render_img(img)
			img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
			writer.write(img_arr)
		writer.release()

		cmd_str = f'ffmpeg -y -i {temp_video} -c:v libx264 -preset fast -x264-params crf=25 -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" -r {fps} {output_video_path}'
		print(f'>>>Running ffmpeg with cmd: {cmd_str}')
		os.system(cmd_str)
		os.remove(temp_video)

if __name__ == '__main__':
	# convert_xml()
	convert_cst_to_mot()
	gen_train_val()
	render_video()
