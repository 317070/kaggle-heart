import utils

d = utils.load_pkl('train.pkl')
pid = 89

print len(d['p0'].keys())

print d['p0'][pid]
print d['p1'][pid]
print d['t0'][pid]
print d['t1'][pid]
