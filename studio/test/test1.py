from pprint import pprint

from studio.process.preprocess.inspect_data import inspect_data

analysis = inspect_data("../dataset.csv")

pprint(analysis)
