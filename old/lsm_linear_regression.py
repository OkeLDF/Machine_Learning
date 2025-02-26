import matplotlib.pyplot as plt

###   'LEAST SQUARES METHOD' (LSM) LINEAR REGRESSION   ###

'GETTER E PRINT DA BASE DADOS:'
def _only(allowed_str, tested_str):
    for c in tested_str:
        if c not in allowed_str:
            return False
    return True

def _numof_digits(x:int, base=10):
    n = 0
    if not x: return 1
    while x!=0:
        d = [x//base, x%base]
        if d[1] or d[0]:
            n += 1
            x = d[0]
            continue
        break
    return n
        
def get_data(filename:str, has_header=True, sep=','): # testar função!!! 
    t = []
    fp = open(filename, 'r')
    i=0

    for line in fp:
        div = line.rstrip().split(sep)

        if has_header:
            t.append(div)
            has_header = False
            i+=1
            continue
        
        t.append([])
        for at in div:
            if _only('0123456789.-+%', at):
                if '%' in at:
                    t[i].append(float(at.strip('%'))/100)
                elif '.' in at:
                    t[i].append(float(at))
                else:
                    t[i].append(int(at))
            else:
                continue
        i+=1
    return t

def printdb(db, verbose=False):
    i = 0
    n = _numof_digits(len(db))
    for line in db:
        print(' ' * (n - _numof_digits(i)), i, sep='', end=' |  ')
        for feat in line:
            print(feat, end='')
            if verbose: print(' (%s)' % (str(type(feat)).removeprefix("<class '").removesuffix("'>")), end='')
            print('  ', end='')
        print()
        i+=1

'PLOTAGEM DOS DADOS:'
def showdata(dataset, x_coln=0, y_coln=1, has_header=True):
    plt.clf()
    if has_header:
        plt.xlabel(dataset[0][x_coln])
        plt.ylabel(dataset[0][y_coln])
        db = dataset[1:]
    else: db = dataset

    for inst in db:
        plt.scatter(inst[x_coln], inst[y_coln], marker='o', color='black')

    plt.grid(True)
    plt.show()

def fit(dataset:list, x_coln=0, y_coln=1, has_header=True):
    x = []
    y = []

    if has_header:
        db = dataset[1:]
    else: db = dataset

    for inst in db:
        x.append(inst[x_coln])
        y.append(inst[y_coln])
    
    x2 = x[:]
    x2 = list(map(lambda x: x**2, x2))
    xy = []
    for i in range(len(db)):
        xy.append(x[i]*y[i])

    n = len(db)

    angular_coef = ((n*sum(xy) - (sum(x)*sum(y)))) / (n*sum(x2) - (sum(x)**2))
    intercept = (sum(y) - (angular_coef*(sum(x)))) / n

    xm = min(x)
    xM = max(x)

    plt.plot((xm, xM), (angular_coef*xm+intercept, angular_coef*xM+intercept), color='red', marker='')
    plt.scatter(x, y, marker='o', color='black')
    plt.grid(True)
    plt.text(xm+2, intercept, 'a = %.3f\nb = %.3f' % (angular_coef,intercept), fontsize=15)
    plt.show()

    return angular_coef, intercept

if __name__ == '__main__':
    dataset = [[1,2.5], [2,4.6], [3,5.3], [4,8.6], [5,9.6], [6,11.1]]
    showdata(dataset, has_header=False)
    fit(dataset, has_header=False)