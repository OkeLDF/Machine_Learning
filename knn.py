import matplotlib.pyplot as plt

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
        
def get_data(filename:str, header_first=True, sep=','): # testar função!!! 
    t = []
    fp = open(filename, 'r')
    i=0

    for line in fp:
        div = line.rstrip().split(sep)

        if header_first:
            t.append(div)
            header_first = False
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
                t[i].append(at)
        i+=1
        # t.append([float(div[0]), float(div[1]), div[2].strip()])
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

'MATEMÁTICA DA COISA:'
def _nan(x):
    return not (isinstance(x, int) or isinstance(x, float))

def _dist(p1, p2):
    s = 0
    for i in range(len(p1)):
        if _nan(p1[i]) or _nan(p2[i]): continue
        s += (p1[i] - p2[i]) ** 2
    return s ** 0.5

def _histogram(t):
    d={}
    for e in t:
        if e in d:
            d[e] += 1
        else:
            d[e] = 1
    return d

def _mostfrequent(t):
    d = _histogram(t)
    mostfreq = ['', 0]
    for k in d:
        if d[k] > mostfreq[1]:
            mostfreq[0] = k
            mostfreq[1] = d[k]
    return mostfreq[0]

def _print_histogram(t):
    d = _histogram(t)
    for k in d:
        print("        %s: %s" % (str(k), str(d[k])))
    print()

def _knn_classify(dados, target_inst, target_coln, target_classname='x', k=3):
    dists=[]
    for inst in dados:
        if inst is target_inst: continue
        if inst[target_coln] == target_classname: continue
        distance = _dist(inst, target_inst)
        dists.append((distance, inst[target_coln]))
    
    dists.sort()
    ngb=[]
    for e in dists[:k]: ngb.append(e[1])
    _print_histogram(ngb)
    return _mostfrequent(ngb)

'PLOTAGEM DOS DADOS:'
def showdata(database, x_coln:int, y_coln:int, class_coln:int, target_classname='x', header_first=True):
    markers = ['s', '^', 'p', 'D', 'h']
    colors = ['red', 'blue', 'lime', 'purple', 'orange']
    i=0
    d = {}
    plt.clf()

    if header_first:
        plt.xlabel(database[0][x_coln])
        plt.ylabel(database[0][y_coln])
        db = database[1:]
    else: db = database

    for inst in db:
        if inst[class_coln] == target_classname:
            plt.scatter(inst[x_coln], inst[y_coln], marker='x', color='black')
            continue

        if inst[class_coln] not in d:
            d[inst[class_coln]] = [markers[i], colors[i]]
            i+=1
        plt.scatter(inst[x_coln], inst[y_coln], marker=d[inst[class_coln]][0], color=d[inst[class_coln]][1])

    plt.grid(True)
    plt.show()

'FUNÇÃO "INTERFACE":'
def classify(database:list, target_coln:int, target_classname='x', k=7, header_first=True):
    '''
    Dada a base de dados 'database', o algoritmo irá classificar
    todas as instâncias cuja classe (identificada pelo número
    da coluna do atributo meta 'target_coln') seja igual a uma
    classe genérica chamada 'target_classname'.

    'k' definirá o número de vizinhos pelo qual o algoritmo 
    classificará a nova instância.
    'header_first' deve ser True se sua base de dados tem
    cabeçalho.
    '''
    k = abs(k)
    if k%2==0: k+=1
    
    i=0
    adder=0
    if header_first:
        db = database[1:]
        adder=1

    classified=[]
    
    print('\nKNN (K = %i) Classification Results:' % (k))

    for inst in db:
        if inst[target_coln] == target_classname:
            print('\n    instance %i:' % (i+adder))
            classified.append([i, _knn_classify(db, inst, target_coln=target_coln, target_classname=target_classname, k=k)])
        i+=1
    
    for c in classified:
        print("    instance %i classified as '%s'" % (c[0]+adder, c[1]))
        db[c[0]][target_coln] = c[1]
    
    return