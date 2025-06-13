import csv
import matplotlib.pyplot as plt

NUMERICAL_FIELDS = ['real_time', 'cpu_time', 'iterations']

def read_csv(filename):
	with open(filename, mode='r') as csv_file:
		while(True):
			pos = csv_file.tell()
			l = csv_file.readline()
			s = l.split(',')
			if len(s) > 1:
				csv_file.seek(pos)
				break

		csv_reader = csv.DictReader(csv_file)
		line_count = 0
		bench = {}
		for row in csv_reader:
			vals = {key:val for key,val in row.items() if key!='name' and val}
			bench[row['name']] = vals

	return bench


def parse_name(name):
	"""Parse the name of a benchmark"""
	s = name.split('/')
	return [s[0], [int(i) for i in s[1:]]]


def rearrange_by_name(dict):
	"""
	Take a dictionnary as returned by read_cvs, and group results with same benchmark name
	"""
	family = {}
	for name, val in dict.items():
		n, x = parse_name(name)
		if n in family.keys():
			# for each name, we want a dict of vectors, one for parameters, one for each possible value
			family[n]['param'].append(x)
			for k,v in val.items():
				family[n][k].append(v)
		else:
			family[n] = {k:[v] for k,v in val.items()}
			family[n]['param'] = [x]

	#we sort the parameters and reorganize the other values accordingly
	for name, val in family.items():
		n = len(val['param'])
		# This line sorts the parameters. Upon completion, idx contains the indices
		# leading to the sort: param = [param[i] for i in idx]
		param, idx = map(list, zip(*sorted(zip(val['param'],range(n)))))
		for k in val.keys():
			if k in NUMERICAL_FIELDS:
				val[k] = [float(val[k][i]) for i in idx]
			else:
				val[k] = [val[k][i] for i in idx]

	return family


def read_bench(filename):
	"""Read a CSV benchmark file."""
	bench = read_csv(filename)
	return rearrange_by_name(bench)


def match_name(s, l):
	""" Return if s matches l, where s can include wildcards '*'."""
	assert(isinstance(s,str))
	assert(isinstance(l,str))

	spl = s.split('*')
	if not l.find(spl[0])==0:
		return False

	start = len(spl[0])
	for si in spl[1:]:
		idx = l.find(si,start)
		if idx<0:
			return False
		else:
			start = idx + len(si)

	return start==len(l) or spl[-1]==''


def match_names(s, l):
	""" Return all elements of l that match s, where s can include wildcards '*'."""
	if isinstance(l,str):
		l = [l]

	return [li for li in l if match_name(s,li)]


def match_all_names(s, l):
	""" Return all elements of l that match an element of s, where s[i] can include wildcards '*'."""
	if isinstance(s,str):
		s = [s]
	if isinstance(l,str):
		l = [l]

	all = []
	for si in s:
		all.extend(match_names(si,l))

	return list(set(all))


def plot_curves(data, names, category='cpu_time', title=None, logx = False, logy = False, filename = None, filetype='png'):
	l = match_all_names(names, data.keys())

	plt.figure()
	for n in l:
		plt.plot(data[n]['param'], data[n][category], label=n, marker='.')
	if logx:
		plt.xscale('log')
	if logy:
		plt.yscale('log')
	plt.title(title)
	plt.legend()
	if filename:
		plt.savefig(filename+'.'+filetype, format=filetype)
	else:
		plt.show()

def plot_relative_curves(data, names, baseline, category='cpu_time', title=None, logx = False, logy = False, filename = None, filetype='png'):
	l = match_all_names(names, data.keys())

	if not title:
		title = 'Comparison with ' + baseline

	plt.figure()
	for n in l:
		plt.plot(data[n]['param'], [a/b for a,b in zip(data[n][category],data[baseline][category])], label=n, marker='.')
	if logx:
		plt.xscale('log')
	if logy:
		plt.yscale('log')
	plt.title(title)
	plt.legend()
	if filename:
		plt.savefig(filename+'.'+filetype, format=filetype)
	else:
		plt.show()
