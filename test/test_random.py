import re

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


x = ['505/study/sax_17.pkl', '505/study/sax_20.pkl']
y =  ['505/study/sax_40.pkl', '505/study/sax_20.pkl']
x.sort(key=natural_keys)
y.sort(key=lambda x: int(re.search(r'/sax_(\d+)\.pkl$', x).group(1)))
print x
print y

print re.search(r'/sax_(\d+)\.pkl$','505/study/sax_589.pkl' ).group(1)