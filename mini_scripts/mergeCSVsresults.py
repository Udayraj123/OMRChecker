import pandas as pd

d = {}
list_csv = ["modular2.csv"]


for a in list_csv:
	x = pd.read_csv(a)
	count = 0
	for y in x.iterrows():
		if y[1]['roll'] in d.keys():
			if y[1]['roll'] != d[y[1]['roll']]['roll']:
				print(y[1])
				print(d[y[1]['roll']])
			count = count + 1
		else:
			d[y[1]['roll']] = y[1]

	print(count)
