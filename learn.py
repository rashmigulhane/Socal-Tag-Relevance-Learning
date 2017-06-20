#Applying knn on all the images

import numpy as np
from sklearn.neighbors import NearestNeighbors
alll=[];
#Declaring 2 array 1 after another
#for loop iterating each file line and putting inside all

a1=[] 
b1=[]
c1=[]
d1=[]
e1=[]
for f1 in open('//home/admin6019/Documents/rashmi/Dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_features/Normalized_CH_Lite_Train.dat'):
	a = f1.strip();
	a = a.split(" ");
	a1.append(a);
for f2 in open('/home/admin6019/Documents/rashmi/Dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_features/Normalized_CM55_Lite_Train.dat'):
	b = f2.strip();
	b = b.split(" ");
	b1.append(b);
for f3 in open('/home/admin6019/Documents/rashmi/Dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_features/Normalized_CORR_Lite_Train.dat'):
	c = f3.strip();
	c = c.split(" ");
	c1.append(c);
for f4 in open('/home/admin6019/Documents/rashmi/Dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_features/Normalized_EDH_Lite_Train.dat'):
	d = f4.strip();
	d = d.split(" ");
	d1.append(d);
for f5 in open('/home/admin6019/Documents/rashmi/Dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_features/Normalized_WT_Lite_Train.dat','r'):
	e = f5.strip();
	e = e.split(" ");
	e1.append(e);

for i in range(len(a1)): 
	getall=[]
	getall.extend(a1[i]);
	getall.extend(b1[i]);
	getall.extend(c1[i]);
	getall.extend(d1[i]);
	getall.extend(e1[i]);
	alll.append(getall);
print "Total number of input"
print len(alll);
print "Reading Test data"
k = input("Enter the value of k\n")
k=k-1
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(alll) 

from collections import defaultdict
import os
k_NN_list = defaultdict(list)


for i in range(len(alll)):
	neigh_array =  neigh.kneighbors(alll[i], return_distance=False)
	neigh_array1 = neigh_array[0].tolist()
	neigh_array1.append(i)
	for j in range(len(neigh_array1)):
		k_NN_list[i].append(neigh_array1[j])


print k_NN_list


#finding value for U and U^T

d = defaultdict(list)

file_names = []
#getting all files and then recurring over it
i=0
for file in os.listdir("/home/admin6019/Documents/rashmi/Dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_groundtruth"):
	
	if file.endswith("_Train.txt") and file!="Lite_GT_Train.txt":
		print file
		file_no=0
		for f1 in open('/home/admin6019/Documents/rashmi/Dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_groundtruth/'+file):
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
#print U
U = np.array(U)
U_transpose = U.transpose()
#print U_transpose



#we get px in 2 stages ... 1st calculate in which all training data is taht tag occuring.

d_tags = defaultdict(list)


file_names_tags = []
#getting all files and then recurring over it
i=0
for file in os.listdir("/home/admin6019/Documents/rashmi/Dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_groundtruth"):
	
	if file.endswith("_Train.txt") and file!="Lite_GT_Train.txt":
		print file
		file_no=0
		for f1 in open('/home/admin6019/Documents/rashmi/Dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_groundtruth/'+file):
			a = f1.strip();
			if(a=='1'):
				d_tags[i].append(file_no)
					
			file_no = file_no+1
		#print d[0]
		i=i+1
		file_names_tags.append(file)

#v_value

v_value =  [ [1] * 1 ] * (k+1)

#GETTING CLASS_NAME FOR A FILE
f7 = []
for f6 in open('/home/admin6019/Documents/rashmi/Dataset/NUS-WIDE-Lite/image list/Train_imageOutPutFileList.txt'):
	a = f6.strip();
	a = a.encode('string-escape').split("\\")
	#if a[0].isalpha():
	f7.append(str(a[0]))


class_details ={}
class_details_1 ={}
alll_class=[]
#print len(a1)
#print len(f7)


for f1 in open('/home/admin6019/Documents/rashmi/Dataset/NUS-WIDE-Lite/image list/Train_imageOutPutFileList.txt'):
	a = f1.strip();
	a = a.encode('string-escape').split("\\")
	#if a[0].isalpha():
	class_details[a[0]] = 1

i=1
for key, value in class_details.iteritems():
	#print class_details[i]
	class_details_1[key] = i
	i=i+1



for i in range(len(f7)): 
	ty = str(f7[i])
	#print ty
	#ss = str(str(class_details_1[ty]) + " " + a1[i] + " "+ b1[i] + " "+ c1[i] + " "+ d1[i] + " "+ e1[i])
	ss = (str(class_details_1[ty]))
	alll_class.append(ss)

#GETTING A MAPPING OF ALL FILES VS ALL TAGS IN WHICH IT OCCURS.

alll_files_tags  =  defaultdict(list)
i=1
for file in os.listdir("/home/admin6019/Documents/rashmi/Dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_groundtruth"):
	
	if file.endswith("_Train.txt") and file!="Lite_GT_Train.txt":
		#print file
		file_no=0
		j1 =0 
		for f1 in open('/home/admin6019/Documents/rashmi/Dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_groundtruth/'+file):
			a = f1.strip();
			if(a=='1'):
				#kt = " " + str(i) + ":" + "1"
				alll_files_tags[j1].append(i)
				#alll[j1] = alll[j1] + kt
			j1 = j1+1
			#print j1		
			
		#print d[0]
		i=i+1



















f23 = open('/home/admin6019/Documents/rashmi/Dataset/train.dat','w')

for jj in range(len(k_NN_list)):
	neigh_array1 = 	k_NN_list[jj]
	w, h = len(neigh_array1),len(d_tags)
	Matrix_px = [[0 for x in range(h)] for y in range(w)]
	mu = 4 * (len(k_NN_list))
	for i in range(w):
		for j in range(h):
			no_with_that_tag = len(d_tags[j])
			totol_no_images = len(k_NN_list)
			special_val = 0;
			if neigh_array1[i] in d_tags[j]:
				special_val = 1;
		
			Matrix_px[i][j] = (((mu*special_val)+no_with_that_tag)*1.0)/(mu+totol_no_images)
	Matrix_px = np.array(Matrix_px)
	Matrix_px = Matrix_px.transpose()
	tag_value = {}
	writa = ""
	writa = alll_class[jj]+ " "
	for i1 in range(len(d_tags)):
		if i1 in alll_files_tags[jj]:
			Matrix_ee = [0 for x in range(len(d_tags))]
			Matrix_ee[i1]=1
			Matrix_ee = np.array(Matrix_ee)
			Matrix_ee = Matrix_ee.transpose()
			part1 = np.dot(Matrix_ee,U_transpose)
			part1 = np.dot(part1,U)
			part1 = np.dot(part1,Matrix_px)
			tag_value[i1] = np.dot(part1,v_value)
			print v_value
			print part1
			print np.dot(part1,v_value)
			writa = writa + str(i1+1) + ":" + str(tag_value[i1][0]/100)+" "
	f23.write(writa.strip()+"\n")
f23.close()			
	





#running svm rank

import os
os.system("/home/admin6019/Documents/rashmi/svm_rank/svm_rank_learn -c 20 /home/admin6019/Documents/rashmi/Dataset/train.dat /home/admin6019/Documents/rashmi/Dataset/model.dat")







