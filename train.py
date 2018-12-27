from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
np.random.seed(123)

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
tf.set_random_seed(123)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.log_device_placement = True 
sess = tf.Session(config=config)
set_session(sess)

import keras
import keras.backend as K
import math
import h5py
import argparse
import pickle
#import matplotlib.pyplot as plt

model = None

def create_parser():
	parser = argparse.ArgumentParser(description='Train the network')

	parser.add_argument('--input', '-i', required=True, type=str, 
		help='A hdf file with compound, kinase and affinity')
	parser.add_argument('--output', '-o', type=str, default='tmp/model_weights.hdf5',
		help="Output file's name")
	parser.add_argument('--percent', '-p', type=float, default=.25,
		help='Percentage of dataset used for validation')
	parser.add_argument('--learning-rate', '-l', type=float, default=1e-5,
		help='The learning rate of Adam Optimizer')
	parser.add_argument('--batch-size', '-b', type=int, default=5, 
		help='Batch size')
	parser.add_argument('--epochs', '-e', type=int, default=20,
		help='Number of epochs')
	parser.add_argument('--dropout', '-d', type=float, default=0.5,
		help='Dropout rate')
	parser.add_argument('--continue', '-c', type=int, default=-1,
		dest='cont', help='Continue training from sample batch')
	parser.add_argument('--log', '-g', action='store_true', default=False,
		help='Log loss and validation loss to training.log')
	parser.add_argument('--verbose', action='store_true', default=False,
		help='Increase output verbosity')	

	return parser

def make_box(coords, features):
	box_size = 21
	grid_coords = coords + 10.0
	grid_coords = grid_coords.round().astype(int)
    
	in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)
	grid = np.zeros((box_size, box_size, box_size, features.shape[1]), dtype=np.float32)
	
	for (x, y, z), f in zip(grid_coords[in_box], features[in_box]):
		grid[x, y, z] += f

	return grid#np.reshape(grid, (1, 21, 21, 21, 19))
			
def get_model(dropout, learning_rate):
	global model
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
	
	model.compile(loss=loss_function, optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['mse', 'mae', loss_function])
	
	return model

def main(argv):
	global model
	v = 1 if argv.verbose else 0
	model = get_model(argv.dropout, argv.learning_rate)
	
	checkpointer = keras.callbacks.ModelCheckpoint(filepath=argv.output, verbose=v)
	rot = np.array(pickle.load(open('rotations.pkl', 'rb')))
	flag = True
	
	#
	#Check and Create Log File
	#
	if argv.log and not os.path.exist('training.log'):
		with open('training.log', 'w') as log:
			log.write('batch_num,loss,val_loss\n')
	
	with h5py.File(argv.input, 'r') as f:
		keys = np.array(list(f.keys()))
		
		rand_spots = np.random.choice(keys.shape[0], (keys.shape[0]//70, 70), replace=False)
		
		for i, spots in enumerate(rand_spots):
			if i < argv.cont:
				flag = False
				continue

			data, affinity = [], []
			best_model = keras.callbacks.ModelCheckpoint(filepath='tmp/{}_best.hdf5'.format(i), verbose=v, save_best_only=True)
			
			#
			#Get Data
			#
			for key in spots:
				complex = f[keys[key]]
				coords = complex[:,:3]
				features = complex[:,3:]
				
				#
				#Get 24 Cube Rotations
				#
				for r in rot:
					rot_coords = np.dot(coords, r)
					data.append(make_box(rot_coords, features))
					try:
						affinity.append(-math.log10(float(complex.attrs['affinity']) / 1e9))
					except:
						affinity.append(0)

			#
			#Train
			#
			data = np.array(data)
			affinity = np.array(affinity).flatten()
			
			if flag:
				history = model.fit(data, affinity, epochs=argv.epochs, batch_size=argv.batch_size, verbose=v, validation_split=argv.percent, callbacks=[checkpointer, best_model])
				flag = False
			else:
				model = keras.models.load_model('tmp/{}_best.hdf5'.format(i-1), custom_objects={'loss_function': loss_function})
				history = model.fit(data, affinity, epochs=argv.epochs, batch_size=argv.batch_size, verbose=v, validation_split=argv.percent, callbacks=[checkpointer, best_model])
				
			#
			#Log File
			#
			if argv.log:
				with open('training.log', 'a') as log:
					for v, l, mse  in zip(history.history['val_loss'], history.history['loss'], history.history['mean_squared_error']):
						log.write('{},{},{}\n'.format(i, v, l, math.sqrt(mse)))
			

def loss_function(y_true, y_pred):
	l2 = 0
	for w in model.layers:
		if type(w) is keras.layers.Conv3D or type(w) is keras.layers.Dense:
			weight = np.array(w.get_weights())
			l2 += np.sum(np.square(weight[0]))
	#l2 = 1e-3 * K.sum([K.sum(K.square(w.get_weights())) for w in model.layers])
	return K.mean(K.square(y_true - y_pred)) + (l2 * 1e-3)
	#return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0)) 
	
if __name__ == '__main__':
	parser = create_parser()

	argv = parser.parse_args()

	main(argv)
