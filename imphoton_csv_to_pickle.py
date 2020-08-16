import csv
import gzip
import pickle
import random
from sys import argv

random.seed(123)

def main():
    photons = []
    with open(argv[1], 'r') as f:
        r = csv.reader(f, delimiter=' ')
        for row in r:
            photons.append(tuple(float(val) for val in row))
    with gzip.open(argv[1]+".pickle.gz", 'wb') as f:
        f.write(pickle.dumps(photons))
    with gzip.open(argv[1]+"_100k.pickle.gz", 'wb') as f:
        f.write(pickle.dumps(random.sample(photons, 100000)))


if __name__ == "__main__":
    main()
