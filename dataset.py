import csv

from helper import Helper


class Dataset:
    def __init__(self, filename):
        self._trials = []
        self._step = 0
        self._lastObservation = []
        with open(filename) as csv_file:
            reader = csv.DictReader(csv_file)
            for line in reader:
                self._trials.append({k: Helper.parse_string_to_list(line[k]) for k in
                                     ('stepOne_Param', 'stepOneTwo_Param', 'stepTwoTwo_Param', 'rewards_Param')})

    @property
    def trials(self):
        return self._trials
