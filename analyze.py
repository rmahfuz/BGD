import sys
print(len(sys.argv))
if len(sys.argv) != 1:
	print("Usage: python3 analyze.py <filename>")

set1 = []
set2 = []
with open(sys.argv[1], 'r') as infile:
	lines = infile.readlines()
	for line in lines:
		nums = line.split(',')
		set1.append(float(nums[0][1:]))
		set2.append(float(nums[1][:-2]))
#print("set1 = ", set1, "set2 = ", set2)
print("Average of ", sys.argv[1], "= [", sum(set1)/len(set1),',', sum(set2)/len(set2),']') 


