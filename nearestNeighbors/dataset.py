import numpy as np

class Dataset(object):

    def __init__(self, filename):
        self._read_csv(filename)

    def _nominal_to_num(self, csv):
        csv_file = None
        rows = []

        for i in range(len(csv[0])):
            row = [data[i] for data in csv]

            if isinstance(row[0], np.bytes_):
                features = np.unique(row)
                row = np.asarray([np.where(features == item)[0][0] for item in row])
                row = (row - row.mean()) / row.std() if i != len(csv[0]) - 1 else row

            rows.append(row)
            
        # put rows back together
        csv_file = [list(item) for item in zip(*rows[:-1])]
        self.data = np.array(csv_file)
        self.target = np.array(rows[-1])

    def _nominal_row_to_num(self, row, features):
        return [np.where(features == item)[0] for item in row]

    def _read_csv(self, filename):
        csv_file = np.genfromtxt(filename, delimiter=",", dtype=None)
        self._nominal_to_num(csv_file)

