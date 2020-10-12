from . import map2sat
from . import day2night


def dataset(args):
    d = args['data']['dataset']
    if d == 'SAT2MAP' or d == 'MAP2SAT':
        return map2sat.MapToSatDataset(args)
    else:
        return day2night.DayToNightDataset(args)



