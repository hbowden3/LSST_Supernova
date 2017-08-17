import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, Column
import math
from astropy.io import ascii


def read_data(asciifile):
    data = np.genfromtxt(asciifile, dtype=[('ra', np.float), ('dec', np.float), ('filter', 'object'), 
                         ('peakday', np.float), ('peakmag', np.float), ('guess peakday', np.float),
                         ('guess peakmag', np.float), ('number of observations', np.float),
                         ('meets requirements', np.bool)], skip_header = 1)
    return data


asciifile = 'minion_1016_wfd1.txt'
data = read_data(asciifile)
data = Table(data)


mag_of_peak = np.arange(17,25,1)
filterNames = ['u', 'g', 'r', 'i', 'z']


percent_well_sampled_table = Table(names=('mag', 'u', 'g', 'r', 'i', 'z'), dtype=('int', 'float', 'float',
                                                                                  'float', 'float', 'float'))
percent_good_fit_table = Table(names=('mag', 'u', 'g', 'r', 'i', 'z'), dtype=('int', 'float', 'float',
                                                                                  'float', 'float', 'float'))
std_day_table = Table(names=('mag', 'u', 'g', 'r', 'i', 'z'), dtype=('int', 'float', 'float',
                                                                                  'float', 'float', 'float'))
mean_day_table = Table(names=('mag', 'u', 'g', 'r', 'i', 'z'), dtype=('int', 'float', 'float',
                                                                                  'float', 'float', 'float'))
std_mag_table = Table(names=('mag', 'u', 'g', 'r', 'i', 'z'), dtype=('int', 'float', 'float',
                                                                                  'float', 'float', 'float'))
mean_mag_table = Table(names=('mag', 'u', 'g', 'r', 'i', 'z'), dtype=('int', 'float', 'float',
                                                                                  'float', 'float', 'float'))
min_mag = 0
max_mag = 0
min_day = 0
max_day = 0
for peakmag in mag_of_peak:
    mMatch = data['peakmag'] == peakmag
    data_match = data[mMatch]
    percent_list1 = [peakmag]
    percent_list2 = [peakmag]
    mean_mag_difference_list = [peakmag]
    mean_day_difference_list = [peakmag]
    std_mag_difference_list = [peakmag]
    std_day_difference_list = [peakmag]
    for f in filterNames:
        fMatch = data_match['filter'] == f
        data_match2 = data_match[fMatch]
        well_sampled = 0.
        good_fit = 0.
        mag_difference_list = []
        day_difference_list = []
        if len(data_match2['ra']) != 0:
            for line in data_match2:
                if line['number_of_observations'] >= 4 and line['meets_requirements'] == True:
                    well_sampled += 1
                    peak_day_difference = line['peakday'] - line['guess_peakday']
                    day_difference_list.append(peak_day_difference)
                    peak_mag_difference = line['peakmag'] - line['guess_peakmag']
                    mag_difference_list.append(peak_mag_difference)
                    if (abs(peak_day_difference) <= 5 and
                        abs(peak_mag_difference) <= 0.5):
                            good_fit += 1
            percent_well_sampled = well_sampled/(len(data_match2['ra']))*100
            if well_sampled == 0:
                percent_good_fit = np.nan
            else:
                percent_good_fit = good_fit/(well_sampled)*100
        else:
            percent_well_sampled = np.nan
        percent_list1.append(percent_well_sampled)
        percent_list2.append(percent_good_fit)
        try:
            iq_range_day = np.percentile(day_difference_list, 75) - np.percentile(day_difference_list, 25)
            min_day = np.percentile(day_difference_list, 50) - 1.5*iq_range_day
            max_day = np.percentile(day_difference_list, 50) + 1.5*iq_range_day
            iq_range_mag = np.percentile(mag_difference_list, 75) - np.percentile(mag_difference_list, 25)
            min_mag = np.percentile(mag_difference_list, 50) - iq_range_mag
            max_mag = np.percentile(mag_difference_list, 50) + iq_range_mag
            mag_difference_list = [mag for mag in mag_difference_list if mag < max_mag]
            mag_difference_list = [mag for mag in mag_difference_list if mag > min_mag]
            day_difference_list = [day for day in day_difference_list if day < max_day]
            day_difference_list = [day for day in day_difference_list if day > min_day]
        except IndexError:
            print('emtpy list')
        mean_mag_difference_list.append(np.mean(mag_difference_list))
        mean_day_difference_list.append(np.mean(day_difference_list))
        std_mag_difference_list.append(np.std(mag_difference_list))
        std_day_difference_list.append(np.std(day_difference_list))
    percent_well_sampled_table.add_row(percent_list1)
    percent_good_fit_table.add_row(percent_list2)
    std_day_table.add_row(std_day_difference_list)
    mean_day_table.add_row(mean_day_difference_list)
    std_mag_table.add_row(std_mag_difference_list)
    mean_mag_table.add_row(mean_mag_difference_list)


percent_well_sampled_table.pprint(max_lines = 10, max_width = 500)
file2 = open("percent_well_sampled.txt", "w+")
ascii.write(percent_well_sampled_table, file2)
file2.flush()


percent_good_fit_table.pprint(max_lines = 10, max_width = 500)
file3 = open("percent_good_fit.txt", "w+")
ascii.write(percent_good_fit_table, file3)
file3.flush()


mean_mag_table.pprint(max_lines = 10, max_width = 500)
file4 = open("mean_mag_table.txt", "w+")
ascii.write(mean_mag_table, file4)
file4.flush()


std_mag_table.pprint(max_lines = 10, max_width = 500)
file5 = open("std_mag_table.txt", "w+")
ascii.write(std_mag_table, file5)
file5.flush()


mean_day_table.pprint(max_lines = 10, max_width = 500)
file6 = open("mean_day_table.txt", "w+")
ascii.write(mean_day_table, file6)
file6.flush()


std_day_table.pprint(max_lines = 10, max_width = 500)
file7 = open("std_day_table.txt", "w+")
ascii.write(std_day_table, file7)
file7.flush()
