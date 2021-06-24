## Determine source position for data where it's not in the header (MIRI)

#Last edit Megan Mansfield 6/24/21

import numpy as np

def source_pos(data,meta):
	subdata=data[0,meta.ywindow[0]:meta.ywindow[1],meta.xwindow[0]:meta.xwindow[1]]
	y_pos = np.argmax(np.sum(subdata, axis = 1))
	return y_pos