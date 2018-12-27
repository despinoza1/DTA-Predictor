from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import requests
import time

def download(info, folder='data'):
	for i in range(len(info['kinase'])):
		uniprot = requests.get('https://www.uniprot.org/uniprot/{}.fasta'.format(info['kinase'][i]))
		if uniprot.status_code == 200:
			with open('{}/{}.fasta'.format(folder, info['kinase'][i]), 'w') as f:
				f.write(uniprot.text)
		else:
			print('Error getting {}.fasta'.format(info['kinase'][i]))
		
		time.sleep(0.1)

def main(csv_file='round_1_template.csv'):
	unique = {'compound': [], 'smile' : [], 'kinase': [] }
	pairs = {'compound': [], 'kinase': []}
	data = pd.read_csv(csv_file)
	shape = data.shape
	
	for i in range(shape[0]):
		c_id = data['Compound_Name'][i]
		c_smi = data['Compound_SMILES'][i]
		k_id = data['UniProt_Id'][i]
		
		if '/' in c_id:
			c_id = c_id.replace('/', '_')
		
		if c_id not in unique['compound']:
			unique['compound'].append(c_id)
			with open('data/' + c_id + '.smi', 'w') as f:
				f.write(c_smi)
		if k_id not in unique['kinase']:
			unique['kinase'].append(k_id)
			
		pairs['compound'].append(c_id)
		pairs['kinase'].append(k_id)
	
	df = pd.DataFrame(data=pairs)
	df.to_csv('pred_pairs.csv', index=False, encoding='utf-8')
	
	download(unique)

if __name__ == '__main__':
	main()
