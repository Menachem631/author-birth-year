g = 'P. G. (Pelham Grenville) Wodehouse'
g.strip()
first = g.find('(')
last = g.find(')')
if first>=0:
    new = (g[:first] + g[last+1:]).replace(' ', '')
else:
    new = g
print(new)