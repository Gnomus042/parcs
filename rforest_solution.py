import subprocess
import sys


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


#install('numpy')
#install('sklearn')
#install('pickle')


import cPickle
import numpy as np
from sklearn.ensemble.forest import RandomForestClassifier

from Pyro4 import expose


class VotingClassifier:

    def __init__(self, classifiers):
        self.classifiers = classifiers

    def predict(self, data):
        res = []
        predictions = np.array(list(map(lambda c: c.predict(data), self.classifiers))).T.tolist()
        for p in predictions:
            res.append(max([(p.count(i), i) for i in range(10)])[1])
        return res


class Solver:
    def __init__(self, workers=None, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.workers = workers
        print("Inited")

    def solve(self):
        print("Job Started")
        print("Workers %d" % len(self.workers))
        data = np.loadtxt(self.input_file_name, dtype=float)
        n_per_instance = 4096 / len(self.workers)

        mapped = []
        data_packed = cPickle.dumps(data[:15000])
        for i in xrange(0, len(self.workers)):
            mapped.append(self.workers[i].mymap(data_packed, n_per_instance))

        reduced = self.myreduce(mapped)

        self.write_output(cPickle.dumps(reduced))

    @staticmethod
    @expose
    def mymap(data, N):
        data = cPickle.loads(str(data))
        x = data[:, :-1]
        y = data[:, -1]
        model = RandomForestClassifier(n_estimators=N, max_depth=6)
        model = model.fit(x, y)
        return cPickle.dumps(model)

    @staticmethod
    @expose
    def myreduce(mapped):
        return [str(x.value) for x in mapped]

    def read_input(self):
        f = open(self.input_file_name, 'r')
        line = f.readline()
        f.close()
        return int(line)

    def write_output(self, output):
        f = open(self.output_file_name, 'w')
        f.write(output)
        f.write('\n')
        f.close()


def load():
    v = list(map(cPickle.loads, cPickle.load(open('output.txt', 'rb'))))
    model = VotingClassifier(v)
    data = np.loadtxt('embedded-test.txt', dtype=float)
    x = data[:, :64].astype(float)
    y = data[:, -1].astype(int)
    yy = model.predict(x)
    print 'Accuracy: %0.3f%%' % (np.sum(yy == y) * 100.0 / len(y))


if __name__ == '__main__':
    load()
