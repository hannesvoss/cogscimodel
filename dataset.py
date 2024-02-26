import csv

from helper import Helper


class Dataset:
    """
    This class is a dataset for the two-step task.
    """
    def __init__(self, filename):
        self._trials = []
        with open(filename) as csv_file:
            reader = csv.DictReader(csv_file)
            for line in reader:
                self._trials.append({k: Helper.parse_string_to_list(line[k]) for k in
                                     ('stepOne_Param', 'stepOneTwo_Param', 'stepTwoTwo_Param', 'rewards_Param')})

    @property
    def trials(self):
        return self._trials
