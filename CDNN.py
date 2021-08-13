'''
If you use this code or CDNN algorithm for your research, please cite this paper.
@article{nguyen2015robust,
  title={Robust biometric recognition from palm depth images for gloved hands},
  author={Nguyen, Binh P and Tay, Wei-Liang and Chui, Chee-Kong},
  journal={IEEE Transactions on Human-Machine Systems},
  volume={45},
  number={6},
  pages={799--804},
  year={2015},
  publisher={IEEE}
}
'''

import numpy as np

def CDNN(input, label, test_sample, k):
	'''
	input: is a list with shape N*d where N is the number of samples, d is the sample dimension
	label: is a list with shape N, where N is the number of samples
	test_sample: is a list with shape d where d is the sample dimension
	k: number of nearest neighbors
	'''
	input_dim = len(input[0])
	#calculate distance
	d = []
	for i in range(len(input)):
		distance = np.linalg.norm(test_sample - input[i])
		d.append(distance)
	d = np.asarray(d)

	#get k lowest distance and save to Sx
	index = np.argpartition(d, k) # return k indexes of lowest value in d
	Sx = dict()
	for idx in range(k):
		key = index[idx]
		if label[key] in Sx:
			Sx[label[key]].append(input[key])
		else:
			Sx[label[key]] = []
			Sx[label[key]].append(input[key])

	#calculate centroid
	px = dict()
	for key in Sx:
		sum_item = np.zeros(input_dim)
		for i in range(len(Sx[key])):
			sum_item += Sx[key][i]

		px_item = sum_item/len(Sx[key])

		px[key] = px_item

	#calculate new centroid
	qx = dict()
	for key in Sx:
		sum_item = np.zeros(input_dim)
		for i in range(len(Sx[key])):
			sum_item+=Sx[key][i]
		sum_item += test_sample
		qx_item = sum_item/(len(Sx[key]) + 1)
		qx[key] = qx_item

	#calculate displacement
	theta = dict()
	for key in px:
		if key in qx:
			theta[key] = np.linalg.norm(px[key] - qx[key])
	#get the label
	return min(theta, key=theta.get)