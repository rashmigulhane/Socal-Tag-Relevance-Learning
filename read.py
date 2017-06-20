from sklearn.neighbors import NearestNeighbors
alll=[];
#Declaring 2 array 1 after another
#for loop iterating each file line and putting inside all

a1=[] 
b1=[]
c1=[]
d1=[]
e1=[]
for f1 in open('/home/admin6019/Documents/rashmi/Dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_features/Normalized_CH_Lite_Train.dat'):
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
print "Total number of training instance"
print len(alll);

k = input("Enter the value of k\n")
k=k-1
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(alll) 

#Reading test data and comparing with the model


alll_test=[];
#Declaring 2 array 1 after another
#for loop iterating each file line and putting inside all
print "Reading Test data"
a1=[] 
b1=[]
c1=[]
d1=[]
e1=[]
for f1 in open('/home/admin6019/Documents/rashmi/Dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_features/Normalized_CH_Lite_Test.dat'):
	a = f1.strip();
	a = a.split(" ");
	a1.append(a);
for f2 in open('/home/admin6019/Documents/rashmi/Dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_features/Normalized_CM55_Lite_Test.dat'):
	b = f2.strip();
	b = b.split(" ");
	b1.append(b);
for f3 in open('/home/admin6019/Documents/rashmi/Dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_features/Normalized_CORR_Lite_Test.dat'):
	c = f3.strip();
	c = c.split(" ");
	c1.append(c);
for f4 in open('/home/admin6019/Documents/rashmi/Dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_features/Normalized_EDH_Lite_Test.dat'):
	d = f4.strip();
	d = d.split(" ");
	d1.append(d);
for f5 in open('/home/admin6019/Documents/rashmi/Dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_features/Normalized_WT_Lite_Test.dat'):
	e = f5.strip();
	e = e.split(" ");
	e1.append(e);
size_imageset = len(a1)
for i in range(len(a1)): 
	getall=[]
	getall.extend(a1[i]);
	getall.extend(b1[i]);
	getall.extend(c1[i]);
	getall.extend(d1[i]);
	getall.extend(e1[i]);
	alll_test.append(getall);
import numpy as np
print "Enter the id of test data"
idd = input("Id of test data from file\n")
print "Record for " + str(idd)  + " is as follow\n "
print str(alll_test[idd-1])+"\n"
neigh_array =  neigh.kneighbors(alll_test[idd-1], return_distance=False)

neigh_array1 = neigh_array[0].tolist()
neigh_array1.append(idd-1)
print neigh_array1
#neigh_array.append(idd-1)



#Applying knn Algorithm




#code for finding correlation
from collections import defaultdict
import os
d = defaultdict(list)



from scipy import linalg
#creation of map with tag id and list with image id int his way got wij



#setting dictionary for 81 concepts
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








d_test = defaultdict(list)


file_names_test = []
#getting all files and then recurring over it
i=0
for file in os.listdir("/home/admin6019/Documents/rashmi/Dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_groundtruth"):
	
	if file.endswith("_Test.txt") and file!="Lite_GT_Train.txt":
		print file
		file_no=0
		for f1 in open('/home/admin6019/Documents/rashmi/Dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_groundtruth/'+file):
			a = f1.strip();
			if(a=='1'):
				d_test[i].append(file_no)
					
			file_no = file_no+1
		#print d[0]
		i=i+1
		file_names_test.append(file)




w, h = len(neigh_array1),len(d_test)
Matrix_px = [[0 for x in range(h)] for y in range(w)]

mu = 4 * (len(e1))
for i in range(w):
	for j in range(h):
		no_with_that_tag = len(d_test[j])
		totol_no_images = len(e1)
		special_val = 0;
		if neigh_array1[i] in d_test[j]:
			special_val = 1;
		
		Matrix_px[i][j] = (((mu*special_val)+no_with_that_tag)*1.0)/(mu+totol_no_images)
#print Matrix_px


#get the content of the model.dat file and create a map with feature id  :  value


print "Tags of K nearest images"
for i in range(w):
	if neigh_array1[i]==k:
		print "Tag of " + (str(neigh_array1[i])+1)
	else
		print "Tag of " + str(neigh_array1[i])
	for j in range(h):
		special_val = 0;
		if neigh_array1[i] in d_test[j]:
			print file_names_test[j]


print "given tag of test data"
j=0
n_line=0
for d41 in open('/home/admin6019/Documents/rashmi/Dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_tags/Lite_Tags81_Test.txt','r'):
	if n_line<=idd:
		if n_line==idd:
			test_data = d41.strip();
		n_line=n_line+1
n_line = n_line.split(" ")	
j=0
for j in range(h):
	if n_line[j]=='1':
		print file_names_test[j]	
	
i=1
for d4 in open('/home/admin6019/Documents/rashmi/Dataset/model.dat','r'):
	if i==12:
		all_features = d4
	i=i+1
all_features = all_features.split(" ")
final_dict = {}
for i in range(len(all_features)):
	if i!=0:
		ktt = all_features[i]
		ktt = ktt.strip()
		ktt = ktt.split(":")
		print ktt
		if len(ktt)==2:
			final_dict[int(ktt[0])-1]=ktt[1]
print final_dict	
final_v_val = []
for i in range(w):
	ansss = 0.0
	for j in range(h):
		if neigh_array1[i] in d_test[j]:
			print neigh_array1[i]
			#print d_test[j]
			#print final_dict[str(j)]
			print j	
			if final_dict.has_key(j):
				print final_dict[j]
				ansss=ansss+float(final_dict[j])

	
	final_v_val.append(ansss)

#print final_v_val
v_value =  [ [0] * 1 ] * (k+1)

for i in range(len(final_v_val)):
	v_value[i] = [final_v_val[i]]
print v_value
	
import operator
Matrix_px = np.array(Matrix_px)
Matrix_px = Matrix_px.transpose()	
tag_value = {}
#Calculate value of v and multiply with v also to get the result.

for i in range(len(d_test)):
	Matrix_ee = [0 for x in range(len(d_test))]
	Matrix_ee[i]=1
	Matrix_ee = np.array(Matrix_ee)
	Matrix_ee = Matrix_ee.transpose()
	part1 = np.dot(Matrix_ee,U_transpose)
	part1 = np.dot(part1,U)
	part1 = np.dot(part1,Matrix_px)
	tag_value[i] = np.dot(part1,v_value)
#	print tag_value[i]
	

tag_value = sorted(tag_value.items(), key=operator.itemgetter(1),reverse=True)
tag_value[0][1]
tag_value[1][1]

print "top 10 predicted tag_value is"
#print tag_value
#print tag_value[0][1]
#tag_value.reverse()
print file_names[tag_value[0][0]]
print file_names[tag_value[1][0]]
print file_names[tag_value[2][0]]
print file_names[tag_value[3][0]]
print file_names[tag_value[4][0]]
print file_names[tag_value[5][0]]
print file_names[tag_value[6][0]]
print file_names[tag_value[7][0]]
print file_names[tag_value[8][0]]
print file_names[tag_value[9][0]]
print file_names[tag_value[10][0]]










	
	


