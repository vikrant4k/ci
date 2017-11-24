import numpy as np
from random import randint
noise = np.random.normal(0,0.6,100)
data=[0.0 ,0.0,1.0,42.08138888888889, 0.929139, -30.30390967180778, 0.45105, 0.403999, 0.389748, 0.402855, 0.448311, 0.50568, 0.546986, 0.600517, 0.671169, 0.767011, 0.902196, 1.10405, 1.43201, 2.04031, 7.56573, 44.9581, 33.5482, 16.4486, 12.6209]
out = open('noise.csv', 'w')
data = np.array(data)
data = data.reshape(1,25)
for i in range(0,100):
	index=randint(3,24)
	temp_data=data
	temp_data[0][index]=temp_data[0][index]+noise[i]
	for j in range(0,25):
	    if(j<24):
	    	out.write('%f,' % temp_data[0][j])
	    else:
	    	out.write(str(temp_data[0][j]))
      
	out.write("\n")
	print(temp_data)
out.close()
