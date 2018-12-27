from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import requests
import os
import time

def get_unique(csv_file=None, save_uniques=False, output_file='unique.csv'):
	if csv_file is None:
		print('File name not provided')
		return 

	unique = {'compound': [], 'kinase': [] }
	
	data = pd.read_csv(csv_file)
	shape = data.shape

	for i in range(shape[0]):
		c_id = data['compound_id'][i]
		k_id = data['target_id'][i]
		
		if c_id not in unique['compound']:
			unique['compound'].append(c_id)
		if k_id not in unique['kinase']:
			unique['kinase'].append(k_id)
	
	if save_uniques:
		if len(unique['compound']) < len(unique['kinase']):
			unique['compound'] += [-1] * (len(unique['kinase']) - len(unique['compound']) )
		else:
			unique['kinase'] += [-1] * (len(unique['compound']) - len(unique['kinase']) )
		
		df = pd.DataFrame(data=unique)
		df.to_csv(output_file, index=False, encoding='utf-8')
	else:
		return unique
	
def download_properties(info, folder='training'):
	if not os.path.exists(folder):
		os.mkdir(folder)
	
	for i in range(len(info['kinase'])):
		kinase = str(info['kinase'][i]).split(',')
		
		for k in kinase:
			uniprot = requests.get('https://www.uniprot.org/uniprot/{}.fasta'.format(k.strip()))
			if uniprot.status_code == 200:
				with open('{}/{}.fasta'.format(folder, k.strip()), 'w') as f:
					f.write(uniprot.text)
			else:
				print('Error getting {}.fasta'.format(k.strip()))
		
			time.sleep(0.1)
	
	for i in range(len(info['compound'])):
		chembl = requests.get('https://www.ebi.ac.uk/chembl/api/data/molecule/{}?format=mol'.format(info['compound'][i]))
		if chembl.status_code == 200:
			with open('{}/{}.mol'.format(folder, info['compound'][i]), 'w') as f:
				f.write(chembl.text)
		else:
			print('Error getting {}.mol'.format(info['compound'][i]))
		
		time.sleep(0.1)

def main():
	u = get_unique('misc/KD_DTC_data.csv')
	download_properties(u)

if __name__ == '__main__':
	main()
