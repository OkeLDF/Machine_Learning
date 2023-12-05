import matplotlib.pyplot as plt

def showdata(database:list, header_first=True):
    order_pairs = {}

    if header_first:
        db = database[1:]
        plt.xlabel(database[0][0])
        plt.ylabel(database[0][1])
    else: db = database

    for i in db:
        order_pairs[i[0]] = i[1]
      
    plt.scatter(db.keys(), db.values(), marker='o')
    plt.grid(True)
    plt.show()

def _only(allowed_str, tested_str):
    for c in tested_str:
        if c not in allowed_str:
            return False
    return True

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

def mean(t):
    if len(t) == 0: return 0
    return sum(t) / len(t)

def get_r(beta0, beta1, dados):
    r = []
    for x in dados.keys():
        yhat = beta0 + beta1 * x
        r.append(yhat - dados[x])
    
    return r

def stddev(set):
    n = len(set)
    m = mean(set)
    s = 0
    for i in set:
        s += (i-m)**2
    return (s/n)**(1/2)

def simple_linreg(dados:dict):
    beta = [0, 1]
    error = [0, 0]
    std = [0, 0]

    xm = min(dados.keys())
    xM = max(dados.keys())

    std[0] = stddev(get_r(beta[0], beta[1], dados))

    for i in range(1, 1001):
        for j in range(1, 1001):
            beta1_hat = i/j
            r = get_r(beta[0], beta1_hat, dados)
            std[1] = stddev(r)

            if std[0] > std[1]:
                # print('i:', i, 'j:', j, 'beta1_hat:', beta1_hat) 
                std[0] = std[1]
                beta[1] = beta1_hat
    
    error[0] = sum(map(lambda n: n**2, r))

    for i in range(int(xm/4), xM*100):
        for j in range(1, 1001):
            beta0_hat = i/j
            r = get_r(beta0_hat, beta[1], dados)
            error[1] = sum(map(lambda n: n**2, r))

            if error[0] > error[1]:
                # print('i:', i, 'j:', j, 'beta0_hat:', beta0_hat)
                error[0] = error[1]
                beta[0] = beta0_hat
    
    print('\nbeta:', beta)

    plt.plot((xm, xM), (beta[1]*xm+beta[0], beta[1]*xM+beta[0]), color='red', marker='+')
    plt.scatter(dados.keys(), dados.values(), marker='o')
    plt.grid(True)
    plt.text(xm+2, beta[0], 'a = %.3f\nb = %.3f' % (beta[1],beta[0]), fontsize=15)
    plt.show()
    return beta

db = get_data('data/linreg_data.csv')
showdata(db)