from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import numpy as np
import h5py
import pybel
import argparse
import csv

def create_parser():
	parser = argparse.ArgumentParser(description='Prepare data for network')

	parser.add_argument('--input', '-i', required=True, type=str, 
		help='A csv file with names of compound and kinase (affinity optional)')
	parser.add_argument('--output', '-o', type=str, default='dataset.hdf',
		help="Output file's name")
	parser.add_argument('--affinity', '-a', action='store_false', default=True,
		help='If true binding affinity is in input file')
	parser.add_argument('--path', '-p', type=str, default='data/',
		help='Path to molecular data')
	parser.add_argument('--compound', '-c', type=str, default='mol',
		help='File format for compounds')
	parser.add_argument('--kinase', '-k', type=str, default='fasta',
		help='File format for kinase')
	parser.add_argument('--next', '-n', type=int, default=-1,
		help='Next drug-kinase pair to retrieve')
	parser.add_argument('--verbose', action='store_true', default=False,
		help='Increase output verbosity')	

	return parser

def get_smarts(mol):
	PATTERN = []
	SMARTS = [
		'[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
		'[a]',
		'[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
		'[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
		'[r]'

	]
	SMARTS_LABELS = ['hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring']

	for s in SMARTS:
		PATTERN.append(pybel.Smarts(s))

	features = np.zeros((len(mol.atoms), len(PATTERN)))

	for (pattern_id, pattern) in enumerate(PATTERN):
		atoms_with_prop = np.array(list(*zip(*pattern.findall(mol))),
			dtype=int) -1
		features[atoms_with_prop, pattern_id] = 1.0

	return features

def get_features(mol, code=0, use_pafnucy=True):
	FEATURES = []
	ATOM_PROPS = ['hyb', 'heavyvalence', 'heterovalence', 'partialcharge']
	ATOM_CODES = {}
	metals = ([3, 4, 11, 12, 13] + list(range(19, 32)) + list(range(37, 51)) + list(range(55, 84)) + list(range(87, 104)))
	ATOM_CLASSES = [
		(5, 'B'),
		(6, 'C'),
		(7, 'N'),
		(8, 'O'),
		(15, 'P'),
		(16, 'S'),
		(34, 'Se'),
		([9, 17, 35, 53], 'halogen'),
		(metals, 'metal')
	]
	
	for code, (atom, name) in enumerate(ATOM_CLASSES):
		if type(atom) is list:
			for a in atom:
				ATOM_CODES[a] = code
		else:
			ATOM_CODES[atom] = code
		FEATURES.append(name)
	
	NUM_ATOM = len(ATOM_CLASSES)

	coords = []
	features = []
	atoms = []

	for i, atom in enumerate(mol):
		if atom.atomicnum > 1:
			atoms.append(i)
			coords.append(atom.coords)
			ad_hoc = np.zeros(NUM_ATOM)
			try:
				ad_hoc[ATOM_CODES[atom.atomicnum]] = 1.0
			except:
				pass

			features.append(np.concatenate((
				(ad_hoc),
				[atom.__getattribute__(prop) for prop in ATOM_PROPS],
				[],
			)))

	coords = np.array(coords, dtype=np.float32)
	features = np.array(features, dtype=np.float32)

	if use_pafnucy:
		features = np.hstack((features, code * np.ones((len(features), 1))))

	features = np.hstack([features, get_smarts(mol)[atoms]])

	return coords, features

def main(argv):
	if argv.verbose:
		print('''
Input File:      {}
Ouput File:      {}
Has Affinity:    {}
Compound Format: {}
Kinase Format:   {}
Next Pair: 		 {}
			'''.format(argv.input, argv.output, argv.affinity, argv.compound, argv.kinase, argv.next))

	compound_kinase_affinity = []
	with open(argv.input, 'r') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
		for row in csvreader:
			if argv.affinity:
				compound_kinase_affinity.append([row[0], row[1], row[2]])
			else:
				compound_kinase_affinity.append([row[0], row[1]])

	with h5py.File(argv.output, 'w') as f:
		i = 0
		for index in enumerate(compound_kinase_affinity):
			if i < argv.next:
				i += 1
				continue
			compound = index[1][0]
			kinase = index[1][1].split(',')
			affinity = None
			if argv.affinity:
				affinity = index[1][2]

			#for j in range(len(kinase)):
			c = next(pybel.readfile(argv.compound, '{}{}.{}'.format(argv.path, compound, argv.compound)))
			k = next(pybel.readfile(argv.kinase, '{}{}.{}'.format(argv.path, kinase[0].strip(), argv.kinase)))

			c_coords, c_features = get_features(c, 1)
			k_coords, k_features = get_features(k, -1)

			center = c_coords.mean(axis=0)
			c_coords -= center
			k_coords -= center

			data = np.concatenate(
				(np.concatenate((c_coords, k_coords)), 
				np.concatenate((c_features, k_features))),
				axis=1,
			)
			print(index[0], '{}_{}'.format(compound, kinase[0]))
			dataset = f.create_dataset('pair_{}'.format(i+1), data=data, shape=data.shape, dtype='float32', compression='lzf')
				
			if affinity:
				dataset.attrs['affinity'] = affinity
			i += 1

if __name__ == '__main__':
	parser = create_parser()

	argv = parser.parse_args()

	main(argv)
