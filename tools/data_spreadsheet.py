from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import xlsxwriter
import math

def get_meta(csv_file='data/DTC_data.csv', save_xlsx=True):
	workbook = None
	worksheet = None
	if save_xlsx:
		workbook = xlsxwriter.Workbook('metadata.xlsx', {'nan_inf_to_errors': True})
		worksheet = workbook.add_worksheet()

	dataset = pd.read_csv(csv_file)

	info = (
		[], 	#compound ids
		{},		#target ids
		{},		#number of targets
		{}		#drug kinase Kd
		)

	shape = dataset.shape
	
	for i in range(shape[0]):
		c_id = dataset['compound_id'][i]
		value, unit = dataset['standard_value'][i], dataset['standard_units'][i]
		t_id = dataset['target_id'][i]

		if unit != 'NM' or value == float('nan'):
			continue
		
		if type(c_id) is float or type(t_id) is float:
			continue

		#Add compound+target Kd
		if c_id in info[3]:
			info[3][c_id].append(value)
		else: 
			info[3][c_id] = []
			info[3][c_id].append(value)	

		#Add compound id
		if c_id not in info[0]:
			info[0].append(c_id)

		#Add target id
		if c_id in info[1]:
			info[1][c_id].append(t_id)
		else:
			info[1][c_id] = []
			info[1][c_id].append(t_id)

		#Increase number of targets
		if c_id in info[2]:
			info[2][c_id] += 1
		else:
			info[2][c_id] = 1

	row, col = 1, 0

	if save_xlsx:
		worksheet.write(0, 0, "Compound ID")
		worksheet.write(0, 1, "# of Targets")
		worksheet.write(0, 2, "Target ID")
		worksheet.write(0, 3, "Kd")

		total = 0

		for i in range(len(info[0])):
			worksheet.write(row, col, info[0][i])
			worksheet.write(row, col+1, info[2][info[0][i]])
			
			col = 2
			for j in range(len(info[1][info[0][i]])):
				try:
					worksheet.write(row, col, info[1][info[0][i]][j])
					worksheet.write(row, col+1, info[3][info[0][i]][j])
				except IndexError:
					print(j, info[1][info[0][i]], info[3][info[0][i]], sep=' ')
				col += 2
			

			total += info[2][info[0][i]]
			row += 1
			col = 0

		print("Total number of compounds: ", len(info[0]))
		print("Total number of targets: ", total)

		workbook.close()
	else:
		with open('binding_KD.csv', 'w') as f:
			for i in range(len(info[0])):
				c_id = info[0][i]
				entry = ''
				#entry = '{}'.format(c_id)
				for j in range(info[2][c_id]):
					entry = entry + '{},"{}",{}\n'.format(c_id, info[1][c_id][j], info[3][c_id][j])
				f.write(entry)

def split_data(Type='Kd', csv_file='Data/DTC_data.csv'):
	dataset = pd.read_csv(csv_file)

	dataset.drop(dataset[dataset.standard_type != 'Kd'].index, inplace=True)
	#shape = dataset.shape
	#for i in range(shape[0]):
	#	if dataset['standard_type'][i].upper() != Type:
	#		dataset = dataset.drop(i, axis=0)
	#		i -= 1

	dataset.to_csv("data/{}_DTC_data.csv".format(Type), index=False, encoding='utf-8')

def main():
	get_meta('misc/KD_DTC_data.csv', False)
	#split_data()

if __name__ == '__main__':
	main()
