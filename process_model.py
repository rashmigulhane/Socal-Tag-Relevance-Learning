i=1
for d4 in open('/home/rashmi/Desktop/sat/svm_rank/model.dat','r'):
	if i==12:
		all_features = d4
	i=i+1
all_features = all_features.split(" ")
final_dict = {}
for i in range(len(all_features)):
	if i!=0:
		ktt = all_features[i]
		ktt = ktt.split(":")
		final_dict[ktt[0]]=ktt[1]
print final_dict

