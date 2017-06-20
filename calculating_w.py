#code for finding correlation
from collections import defaultdict
import os
d = defaultdict(list)

import numpy as np

from scipy import linalg
#creation of map with tag id and list with image id int his way got wij



#setting dictionary for 81 concepts
file_names = []
#getting all files and then recurring over it
i=0
for file in os.listdir("/home/rashmi/Desktop/sat/original_dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_groundtruth"):
	
	if file.endswith("_Train.txt") and file!="Lite_GT_Train.txt":
		print file
		file_no=0
		for f1 in open('/home/rashmi/Desktop/sat/original_dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_groundtruth/'+file):
			a = f1.strip();
			if(a=='1'):
				d[i].append(file_no)
					
			file_no = file_no+1
		#print d[0]
		i=i+1
		file_names.append(file)
print len(d)

#creation of 81*81 matrix

w, h = len(d),len(d)
Matrix = [[0 for x in range(w)] for y in range(h)]
for i in range(len(d)):
	for j in range(len(d)):
		list1 = d[i]
		list2 = d[j]
		interlen = len(set(list1).intersection(list2))
		unionlen = len(set(list1).union(list2))*1.0
		Matrix[i][j] = interlen/unionlen
#print Matrix[1][9]
#print Matrix[0][1]

#Applying matrx decomposion so that we get 2 matrix U and U^T

U = np.linalg.cholesky(Matrix)
print U
U = np.array(U)
U_transpose = U.transpose()
print U_transpose












