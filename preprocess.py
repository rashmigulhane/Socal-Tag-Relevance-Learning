#Creating a Dictionary for storing classes
from collections import defaultdict
import os
class_details ={}
class_details_1 ={}

for f1 in open('/home/rashmi/Desktop/sat/original_dataset/NUS-WIDE-Lite/image list/Train_imageOutPutFileList.txt'):
	a = f1.strip();
	a = a.encode('string-escape').split("\\")
	#if a[0].isalpha():
	class_details[a[0]] = 1
	
print class_details
print len(class_details)
i=1
for key, value in class_details.iteritems():
	#print class_details[i]
	class_details_1[key] = i
	i=i+1



print len(class_details_1)

print class_details_1


#get all trianing data 



alll_test=[];
#Declaring 2 array 1 after another
#for loop iterating each file line and putting inside all

a1=[] 
b1=[]
c1=[]
d1=[]
e1=[]
f7 = []
for f1 in open('/home/rashmi/Desktop/sat/original_dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_features/Normalized_CH_Lite_Train.dat'):
	a = f1.strip();
	a1.append(a);
for f2 in open('/home/rashmi/Desktop/sat/original_dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_features/Normalized_CM55_Lite_Train.dat'):
	b = f2.strip();
	b1.append(b);
for f3 in open('/home/rashmi/Desktop/sat/original_dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_features/Normalized_CORR_Lite_Train.dat'):
	c = f3.strip();
	c1.append(c);
for f4 in open('/home/rashmi/Desktop/sat/original_dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_features/Normalized_EDH_Lite_Train.dat'):
	d = f4.strip();
	d1.append(d);
for f5 in open('/home/rashmi/Desktop/sat/original_dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_features/Normalized_WT_Lite_Train.dat'):
	e = f5.strip();
	e1.append(e);
for f6 in open('/home/rashmi/Desktop/sat/original_dataset/NUS-WIDE-Lite/image list/Train_imageOutPutFileList.txt'):
	a = f6.strip();
	a = a.encode('string-escape').split("\\")
	#if a[0].isalpha():
	f7.append(str(a[0]))
alll=[]
#print len(a1)
#print len(f7)
for i in range(len(a1)): 
	ty = str(f7[i])
	#print ty
	#ss = str(str(class_details_1[ty]) + " " + a1[i] + " "+ b1[i] + " "+ c1[i] + " "+ d1[i] + " "+ e1[i])
	ss = (str(class_details_1[ty]))
	alll.append(ss)
#print i
#print alll[0]
#print len(alll[0])		
alll_final = []						
#split alll and provide numbering to it.
'''for i in range(alll):
	breaker = alll[i].split(" ")
	str1=""
	for j in range(breaker):
		if j!=0:
			str1 = str1 + str(j+1) +":"+ breaker[j]
	alll_final.append(str1)
'''
#get a for loop which will get all the files of train type (copy paste that code) and now start from variable 7000 for each file iterate over alll and append if 1.
print len(alll)
i=1
for file in os.listdir("/home/rashmi/Desktop/sat/original_dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_groundtruth"):
	
	if file.endswith("_Train.txt") and file!="Lite_GT_Train.txt":
		#print file
		file_no=0
		j1 =0 
		for f1 in open('/home/rashmi/Desktop/sat/original_dataset/NUS-WIDE-Lite/NUS-WIDE-Lite_groundtruth/'+file):
			a = f1.strip();
			if(a=='1'):
				kt = " " + str(i) + ":" + "1"
				alll[j1] = alll[j1] + kt
			j1 = j1+1
			#print j1		
			
		#print d[0]
		i=i+1
		
print alll

f23 = open('/home/rashmi/Desktop/sat/original_dataset/NUS-WIDE-Lite/for_Rank.dat','w')
for i in range(len(alll)):
	f23.write(alll[i].strip()+"\n")
f23.close()	




#write alll_final to a file line by line.

#now give this file input to rank algo and get the model file

#now get your k nearest neighbour.Remember to make your old changes

#get the features which you want.preferable tags and add there weights.


#now you get a v vector			
