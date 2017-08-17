"""This program resamples a template Type 1b/c supernova lightcurve through the OpSim database given one of
the proposed cadences. The template is adjusted to simulate the supernova occuring at different
positions in the sky, distances from Earth, and times during the survey. We examine data for 
one filter at a time. For those combinations with at least 4 observations on seperate nights
during the first 30 days of the template light curve, we attempt to fit a quadratic the nickel
peak (again the first 30 days). This program returns a list of ra, dec, filter, peak day, and 
peak mag of each combination along with the peak day and peak mag found for the quadratic and 
the number of observations made."""

# import our python modules
import lsst.sims.maf.db as db
import lsst.sims.maf.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.plots as plots
from astropy.table import Table, Column
from scipy import optimize
import math
from astropy.io import ascii

#asciiLC is the template light curve
asciiLC = 'supernova1b_template.dat'
filterNames = ['u', 'g', 'r', 'i', 'z']
colors = {'u':'purple','g':'g','r':'r','i':'blue','z':'m'}
#create list of peak days and magnitudes to iterate over
day_of_peak = np.arange(59580, 63232, 30)
mag_of_peak = np.arange(17,25,1)


# Set the database and query
runName = 'minion_1018'
opsdb = db.OpsimDatabase(runName + '_sqlite.db')

# Set the output directory
outDir = 'Observations Dictionary'
resultsDb = db.ResultsDb(outDir)


# This creates our database of observations. The pass metric just passes data straight through.
metric = metrics.PassMetric(cols=['expMJD','filter','fiveSigmaDepth'])
"""use slicer to restrict the ra and decs, use np.random.uniform to get random points, 
	first coordinate represents ra and second dec. Or, give a list of specific
	ra and decs - the second slicer is for the deep drilling fields. One must be commented out.""" 
#slicer = slicers.UserPointsSlicer(np.random.uniform(0,360,1000), np.random.uniform(-80,0,1000))
slicer = slicers.UserPointsSlicer([349.4,0.00,53.0,34.4,150.4],[-63.3,-45.5,-27.4,-5.1,2.8])
#sql is empty as there are no restrictions currently
sql = ''
bundle = metricBundles.MetricBundle(metric,slicer,sql)
bg =  metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bg.runAll()


def createdict_for_mjd_filter_depth(bundle):
    
    """This function returns a list of tables of exposure day, filter, 
    and five sigma depth for each ra and dec chosen"""

    number_of_coord = len(bundle.metricValues)
    listofDict = []
    for coord in range(len(bundle.metricValues)):
        if isinstance(bundle.metricValues[coord],np.ma.MaskedArray):
            number_of_coord -= 1
        else:
            t = Table(bundle.metricValues[coord])
            t.rename_column('expMJD', 'day')
            t = t[t['filter'] != 'y']
            t.sort('day')
            listofDict.append(t)
    return listofDict, number_of_coord


def read_lightcurve(asciifile, f):

    """Reads asciifile columns corresponding to each f - 3 columns (day, magnitude, error).
	This will need to be altered if a different formatted file is used"""

    skip = 0
    if f == 'u':
        columns = (0,1,2)
        skip = 40
    elif f == 'g':
        columns = (3,4,5)
    elif f == 'r':
        columns = (6,7,8)
    elif f == 'i':
        columns = (9,10,11)
    elif f == 'z':
        columns = (12,13,14)
    else:
        pass
    lc = np.genfromtxt(asciifile, dtype=[('day', np.float), ('mag', np.float), ('error', np.float)],
                       skip_header = 1, skip_footer = skip, usecols = columns)
    return {f:lc}


def add_data_to_lc_table(asciiLC, template):

    '''This function returns a table with day, magnitude, and filter for a light curve read
    	in from an ascii file'''

    for f in filterNames:  
        curvedata = read_lightcurve(asciiLC, f)
        bdict = {key: curvedata[f][key] for key in ['day', 'mag', 'error']}
        t = Table(bdict)
        t['filter'] = f
        for row in range(len(t)):
	#The template lightcurve used was in terms of flux, here we convert it to magnitude
            t['mag'][row] = -2.5*math.log10(t['mag'][row])
            template.add_row(t[row])
    template.sort('day')
    return template


def normalize_template(template):

    #Shifts template so that peak in g filter occurs at (0,0)

    peaktable = peak_brightness(template)
    peakmag = peaktable['g'][1]
    template['mag'] -= peakmag
    return template


def peak_brightness(template):
    
    """This function returns a table of peak magnitude and the day it occurs 
    for each filter from the read in lightcurve"""
    
    peak_brightness = {}
    for f in filterNames:
        fMatch = np.where(template['filter'] == f)
        maxmag = np.amin(template['mag'][fMatch])
        location = np.argmin(template['mag'][fMatch])
        maxday = template['day'][fMatch][location]
        peak_brightness[f] = [maxday, maxmag]
    
    peak = Table(peak_brightness)
    peak[' '] = ['day', 'mag']
    orderedPeak = peak[' ','u','g','r','i','z']
    return orderedPeak


def adjust_peak(template, peakday, peakmag):

    """This function finds the necessary adjustment needed to make the peak 
    of the red filter occur at the right place and adjusts all filters by that same ammount"""

    adjusted_template = template.copy()
    adjusted_template['day'] += peakday
    adjusted_template['mag'] += peakmag
    
    return adjusted_template


def adjust_opsim_table(opsim, adjusted_template):

    """Cuts off parts of opsim that we will not use - those outside the range of the adjusted_template"""

    new_opsim = opsim.copy()
    new_opsim = new_opsim[new_opsim['day']< adjusted_template['day'].max()]
    new_opsim = new_opsim[new_opsim['day']> adjusted_template['day'].min()]
    return new_opsim


def interpolate_lightcurve(adjusted_template, new_opsim):
    
    #for each filter interpolate the read in light curve to the days of the opsim in all filters
    
    lc = {}
    for f in filterNames:
        fMatch = np.where(adjusted_template['filter'] == f)
        lc[f] = np.interp(new_opsim['day'], adjusted_template['day'][fMatch], adjusted_template['mag'][fMatch])
    lightcurve = Table(lc)
    lightcurve['day'] = new_opsim['day']
    lightcurve = lightcurve['day', 'u', 'g', 'r', 'i', 'z']
    return lightcurve


def resample_lightcurve(lightcurve, new_opsim):
    
    """Add to the opsim table a magnitude column where the magnitude is taken 
    from the index of the interpolated light curve where both the filter and day matched the opsim"""
    
    new_opsim['magnitude'] = 0.
    for row in range(len(new_opsim)):
        filterName = new_opsim['filter'][row]
        new_opsim['magnitude'][row] = lightcurve[filterName][row]
    return new_opsim


def calculate_error(new_opsim):
    
    #Calculate the error of the magnitude at each point from the magnitude and fiveSigmaDepth stored in opsim
    
    snr = 5.*10.**(-0.4*(new_opsim['magnitude'] - new_opsim['fiveSigmaDepth']))
    lc_err = 2.5/(np.log(10)*snr)
    new_opsim['error'] = lc_err
    
    return new_opsim


def magnitude_distribution(new_opsim):

    """Let magnitude of the resampled points be on the normal distribution calculated using the exact magnitude
        and the error"""

    new_opsim['magnitude'] = np.random.normal(new_opsim['magnitude'], new_opsim['error'])
    return new_opsim


def func(x, a, b, c, x0):
    #Function used for polynomial fitting
    return a*(x-x0)**2 + b*(x-x0) + c


def fit_curve(opsim_fmatch, template_fmatch2):

    #Use curve_fit to find a, b, c, x0 from the above function

    opsim_fmatch = opsim_fmatch[opsim_fmatch['day'] <= template_fmatch2['day'].max()]
    peak_day = opsim_fmatch['day'][opsim_fmatch['magnitude'].argmin()]
    peak_mag = opsim_fmatch['magnitude'].min()
    x_min = opsim_fmatch['day'].min()
    y_min = opsim_fmatch['magnitude'][opsim_fmatch['day'].argmin()]
    x_max = opsim_fmatch['day'].max()
    y_max = opsim_fmatch['magnitude'][opsim_fmatch['day'].argmax()]
    if x_min != peak_day:
        a = (y_min - peak_mag) / ((x_min - peak_day)**2)
    else:
        a = (y_max - peak_mag) / ((x_max - peak_day)**2)
    initial_parameters = [a, 0, peak_mag, peak_day]
    try:
        popt, pcov = optimize.curve_fit(func, opsim_fmatch['day'], opsim_fmatch['magnitude'], 
                                        p0 = initial_parameters, sigma = opsim_fmatch['error'])
    except RuntimeError:
        print("Optimal Parameters Not Found")
        popt = [0,0,0,0]
    return popt


def create_curve(f, ra, dec, peakday, peakmag, opsim_fmatch2, template_fmatch2, xdata, popt):

    #plot points, template, and polynomial fitting

    plt.plot(xdata, func(xdata, *popt), 'teal', label = 'poly')
    plt.errorbar(opsim_fmatch2['day'], opsim_fmatch2['magnitude'], 
                   yerr = opsim_fmatch2['error'], fmt = 'o', color=colors[f], label=f)
    plt.plot(template_fmatch2['day'], template_fmatch2['mag'], 
                     color=colors[f], label=f)
    plt.xlabel('day')
    plt.ylabel('magnitude')
    plt.ylim(22,15)
    plt.legend(numpoints = 1)
    plt.title('Lightcurve at ra = %r and dec = %r, peakday = %r and peakmag = %r'
              %(round(ra,3), round(dec,3), peakday, peakmag))
    plt.show()


survey, number_of_coord = createdict_for_mjd_filter_depth(bundle)


template = Table(names=('day', 'mag', 'error', 'filter'), dtype=('float', 'float', 'float', 'string'))
template = add_data_to_lc_table(asciiLC, template)
peaktable = peak_brightness(template)
template = normalize_template(template)


final_data = Table(names=('ra', 'dec', 'filter', 'actual peak day', 'actual peak mag', 'guess peak day', 'guess peak mag',
                          'number of observations', 'meets requirements'), dtype=('float', 'float', 'string', 'float', 'float', 'float', 'float', 'int', 'bool'))


for f in filterNames:
    for coord in range(number_of_coord):
        opsim = survey[coord]
        ra = np.degrees(opsim['fieldRA'][0])
        dec = np.degrees(opsim['fieldDec'][0])
        for peakday in day_of_peak:
            for peakmag in mag_of_peak:
                meets_requirements = False
                adjusted_template = adjust_peak(template, peakday, peakmag)
                new_opsim = adjust_opsim_table(opsim, adjusted_template)
                if len(new_opsim) != 0:
                    lightcurve = interpolate_lightcurve(adjusted_template, new_opsim)
                    new_opsim = resample_lightcurve(lightcurve, new_opsim)
                    new_opsim = calculate_error(new_opsim)
                    new_opsim = magnitude_distribution(new_opsim)
                    final_opsim = new_opsim.copy()
                    final_opsim = final_opsim[final_opsim['day'] <= (30 + adjusted_template['day'].min())]
                    fMatch = np.where(final_opsim['filter'] == f)
                    opsim_fmatch = final_opsim[fMatch]
                    opsim_rounded = opsim_fmatch.copy()
                    opsim_rounded['day'] = np.round(opsim_rounded['day'])

		    """We classify a lightcurve as well sampled if there are at least 4 observations in the first 30 days and
			if the minimum magnitude of these observations does not correspond to the first or last day of observation.
			We only fit a curve to those combination that meet these requirements"""
                    if (len(np.unique(opsim_rounded['day'])) >= 4
                        and np.round(opsim_fmatch['day'][opsim_fmatch['magnitude'].argmin()]) !=
                                                         np.round(opsim_fmatch['day'].max())
                        and np.round(opsim_fmatch['day'][opsim_fmatch['magnitude'].argmin()]) !=
                                                         np.round(opsim_fmatch['day'].min())):
                        fMatch2 = np.where(new_opsim['filter'] == f)
                        opsim_fmatch2 = new_opsim[fMatch2]
                        fMatch3 = np.where(adjusted_template['filter'] == f)
                        template_fmatch = adjusted_template[fMatch3]
                        xdata = np.arange(template_fmatch['day'].min(), template_fmatch['day'].max(), 0.1)
                        popt = fit_curve(opsim_fmatch, template_fmatch)
                        guess_peakmag = func(xdata, *popt).min()
                        guess_peakday = np.float(func(xdata, *popt).argmin())/10 + xdata.min()
                        meets_requirements = True
                        final_data.add_row([ra,dec,f,peakday,peakmag,guess_peakday,guess_peakmag,
                                           len(np.unique(opsim_rounded['day'])), meets_requirements])
                    else:
                        final_data.add_row([ra,dec,f,peakday,peakmag,0,0,len(np.unique(opsim_rounded['day'])), meets_requirements])


file1 = open("minion_1018_dd.txt", "w+")
ascii.write(final_data, file1)
file1.flush()
