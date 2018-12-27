from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
np.random.seed(12345)

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
tf.set_random_seed(12345)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.log_device_placement = True 
sess = tf.Session(config=config)
set_session(sess)

import keras
import keras.backend as K
import argparse
import h5py
import pandas as pd
import math
import pickle

model = None

def create_parser():
	parser = argparse.ArgumentParser(description='Predict Binding Affinity between a compound and kinase')

	parser.add_argument('--input', '-i', required=True, type=str, 
		help='A hdf file with compound and kinase')
	parser.add_argument('--output', '-o', type=str, default='predictions.csv',
		help="Output file's name")
	parser.add_argument('--csv-file', '-c', type=str, default='',
		help='CSV file')
	parser.add_argument('--error', '-e', action='store_true', default=False,
		help='Get error of predicted affinity')
	parser.add_argument('--model', '-m', type=str, default='best.hdf5',
		help='Path to model')
	parser.add_argument('--verbose', action='store_true', default=False,
		help='Increase output verbosity')	

	return parser
	
def make_box(coords, features):
	box_size = 21
	grid_coords = coords + 10
	grid_coords = grid_coords.round().astype(int)
    
	in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)
	grid = np.zeros((box_size, box_size, box_size, features.shape[1]), dtype=np.float32)
	
	for (x, y, z), f in zip(grid_coords[in_box], features[in_box]):
		grid[x, y, z] += f

	return grid
	
def create_model(dropout, rate):
	global model
	#
	#Create Model
	#
	model = keras.Sequential()
	
	#Conv_1
	model.add(keras.layers.Conv3D(64, kernel_size=(5, 5, 5), activation='relu',
		input_shape=(21, 21, 21, 19), padding='same'))
	model.add(keras.layers.Conv3D(64, kernel_size=(5, 5, 5), activation='relu',
		padding='same'))
	model.add(keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
	#Conv_2
	model.add(keras.layers.Conv3D(128, kernel_size=(5, 5, 5), activation='relu',
		padding='same'))
	model.add(keras.layers.Conv3D(128, kernel_size=(5, 5, 5), activation='relu',
		padding='same'))
	model.add(keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
	#Conv_3
	model.add(keras.layers.Conv3D(256, kernel_size=(5, 5, 5), activation='relu',
		padding='same'))
	model.add(keras.layers.Conv3D(256, kernel_size=(5, 5, 5), activation='relu',
		padding='same'))
	model.add(keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
	
	#Dense_1
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(1000, activation='relu'))
	model.add(keras.layers.Dropout(dropout))
	#Dense_2
	model.add(keras.layers.Dense(500, activation='relu'))
	model.add(keras.layers.Dropout(dropout))
	#Dense_3
	model.add(keras.layers.Dense(200, activation='relu'))
	model.add(keras.layers.Dropout(dropout))
	
	#Output
	model.add(keras.layers.Dense(1, activation='relu'))
	model.add(keras.layers.Dropout(dropout))
	
	model.compile(loss=loss_function, optimizer=keras.optimizers.Adam(lr=rate), metrics=['mse', 'mae', loss_function])
	
	return model
	
def main(argv):
	global model
	
	if argv.verbose:
		print('''
Input File:    {}
Ouput File:    {}
CSV File:      {}
Model Path:    {}
Validation:    {}
			'''.format(argv.input, argv.output, argv.csv_file, argv.model, argv.error))
	#
	#Get Data
	#
	if argv.verbose:
		print("Reading dataset and creating 20 A boxes")
		
		
	data = []
	affinity = []
	with h5py.File(argv.input, 'r') as f:
		keys = np.array(list(f.keys()))
		rot = np.array(pickle.load(open('rotations.pkl', 'rb')))

		rand_rot = [0] if not argv.error else np.random.choice(rot.shape[0], (4), replace=False)

		for key in keys:
			complex = f[key]

			for rotation in rand_rot:
				coords = np.dot(complex[:,:3], rot[rotation])
				features = complex[:,3:]
			
				data.append(make_box(coords, features))
				if argv.error:
					try:
						affinity.append(-math.log10(float(complex.attrs['affinity']) / 1e9))
					except:
						affinity.append(0)

	if argv.verbose:
		print('Loading model and predicting')
	data = np.array(data)
	if argv.error:
		affinity = np.array(affinity).flatten()
	
	model = create_model(0.5, 1e-5)
	model = keras.models.load_model(argv.model, custom_objects={'loss_function': loss_function})
	
	y_pred = model.predict(data, batch_size=5)
	y_pred = np.array(y_pred).flatten()
	
	if len(argv.csv_file) > 0:
		df = pd.read_csv(argv.csv_file)
		df = df.assign(pred=y_pred)
		df.rename(columns={'pred': 'pKd_[M]_pred'}, inplace=True)
	
		if not os.path.isfile(argv.output):
			open(argv.output, 'w')
	
		df.to_csv(argv.output, index=False, encoding='utf-8')

	if argv.error:
		#RMSE
		rmse = np.sqrt(np.mean(np.square(affinity - y_pred)))
		
		#MAE
		mae = np.mean(np.absolute(affinity - y_pred))
		
		#Pearson correlation
		R = np.corrcoef(affinity, y_pred)[0, 1]

		print('RMSE: ', rmse)
		print('MAE: ', mae)
		print('R: ', R)
		
def loss_function(y_true, y_pred):
	l2 = 0
	for w in model.layers:
		if type(w) is keras.layers.Conv3D or type(w) is keras.layers.Dense:
			weight = np.array(w.get_weights())
			l2 += np.sum(np.square(weight[0]))
	return K.mean(K.square(y_true - y_pred)) + (l2 * 1e-3)
	#return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0)) 
	
if __name__ == '__main__':
	parser = create_parser()

	argv = parser.parse_args()

	main(argv)
