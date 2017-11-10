#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 19:43:03 2017

@author: wellenbu
"""

import datetime
import ee
import sys
from scipy import stats
import numpy
import pandas as pd
import matplotlib.pyplot as plt
#from PIL import ImageTk
#import ee.mapclient


ee.Initialize()


def addNDVI(image):
    ndvi = image.normalizedDifference(['nir','red']).rename(['nd'])
    #ndvi = image.normalizedDifference(['nir', 'red'])
    #return image.addBands(ndvi)
    return ndvi.set("system:time_start",image.get("system:time_start"))
    
def addNDVI2(image):
    ndvi = image.normalizedDifference(['B5','B4']).rename(['nd'])
    #ndvi = image.normalizedDifference(['nir', 'red'])
    #return image.addBands(ndvi)
    return ndvi.set("system:time_start",image.get("system:time_start"))
    
def lsCloudMask(img):
  blank = ee.Image(0)
  scored = ee.Algorithms.Landsat.simpleCloudScore(img)
  clouds = blank.where(scored.select(['cloud']).lte(cloudThresh),1)
  return img.updateMask(clouds).set("system:time_start",img.get("system:time_start"))
  
# A helper to apply an expression and linearly rescale the output 
def rescale(img, exp, thresholds):
  return img.expression(exp, {'img': img})\
      .subtract(thresholds[0]).divide(thresholds[1] - thresholds[0])

def s2CloudMask(img):
  #Compute several indicators of cloudiness and take the minimum of them.
  #assigns a baseline value of 1 to each pixel
  score = ee.Image(1.0)
  
   #Clouds are reasonably bright in the blue band. 1900
  score = score.min(rescale(img, 'img.B2', [2000, 3000]))
 
  #Clouds are reasonably bright in all visible bands.19
  score = score.min(rescale(img, 'img.B4 + img.B3 + img.B2', [1900, 8000]))
   
  #Clouds are reasonably bright in all infrared bands.
  score = score.min(rescale(img, 'img.B8 + img.B11 + img.B12', [2000, 8000]))

  #However, clouds are not snow.
  ndsi = img.normalizedDifference(['B3', 'B11'])
  score =  score.min(rescale(ndsi, 'img', [0.8, 0.6]))
  
  #Cirrus masking350
  score = score.min(rescale(img, 'img.B10', [10,100]))
 
  score = score.multiply(100).byte().lt(cloudThresh)
  s2Clouds = img.select(['QA60']).lt(1024)
  finalClouds =score.Or(s2Clouds).rename(['cloudMask'])
  img = img.updateMask(finalClouds)
  return img.divide(10000).set("system:time_start",img.get("system:time_start"))
  
  
def mergeOptical(studyArea,t1,t2):
    le7rename = le7.filterBounds(studyArea)\
                    .filterDate(t1,t2)\
                    .filterMetadata('CLOUD_COVER','less_than',metadataCloudCoverMax)\
                    .map(lsCloudMask)\
                    .select(['B1','B2','B3','B4','B5','B7'],\
                            ['blue','green','red','nir','swir1','swir2'])
    lc8rename = lc8.filterBounds(studyArea)\
                    .filterDate(t1,t2)\
                    .filterMetadata('CLOUD_COVER','less_than',metadataCloudCoverMax)\
                    .map(lsCloudMask)\
                    .select(['B2','B3','B4','B5','B6','B7'],\
                            ['blue','green','red','nir','swir1','swir2'])
    st2rename = s2.filterBounds(studyArea)\
                    .filterDate(t1,t2)\
                    .filterMetadata('CLOUD_COVERAGE_ASSESSMENT','less_than',metadataCloudCoverMax)\
                    .map(s2CloudMask)\
                    .select(['B2','B3','B4','B8','B11','B12'],\
                            ['blue','green','red','nir','swir1','swir2'])\
                    .map(bandPassAdjustment)
                    
    out = ee.ImageCollection(le7rename.merge(lc8rename).merge(st2rename))
    
    return out


def bandPassAdjustment(img):
    bands = ['blue','green','red','nir','swir1','swir2']
    #linear regression coefficients for adjustment
    gain = ee.Array([[0.977], [1.005], [0.982], [1.001], [1.001], [0.996]])
    bias = ee.Array([[-0.00411],[-0.00093],[0.00094],[-0.00029],[-0.00015],[-0.00097]])
    #Make an Array Image, with a 1-D Array per pixel.
    arrayImage1D = img.select(bands).toArray()
  
    #Make an Array Image with a 2-D Array per pixel, 6x1.
    arrayImage2D = arrayImage1D.toArray(1)
    
    #Get rid of the extra dimensions. and Get a multi-band image with named bands.
    componentsImage = ee.Image(gain).multiply(arrayImage2D).add(ee.Image(bias))\
        .arrayProject([0])\
        .arrayFlatten([bands])\
        .float()
    
    return componentsImage.set('system:time_start',img.get('system:time_start'))

def smooth(x,window_len= 11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y


    
'''
def simpleTDOM2(collection,zScoreThresh,shadowSumThresh,dilatePixels):
    shadowSumBands = ['nir','swir1']
    #Get some pixel-wise stats for the time series
    irStdDev = collection.select(shadowSumBands).reduce(ee.Reducer.stdDev())
    irMean = collection.select(shadowSumBands).mean()
    def function(img):
        zScore = img.select(shadowSumBands).subtract(irMean).divide(irStdDev)
        irSum = img.select(shadowSumBands).reduce(ee.Reducer.sum())
        TDOMMask = zScore.lt(zScoreThresh).reduce(ee.Reducer.sum()).eq(2).And(irSum.lt(shadowSumThresh)).Not()
        TDOMMask = TDOMMask.focal_min(dilatePixels)
        return img.addBands(TDOMMask.rename('TDOMMask'))
    collection = collection.map(function(img))
    #Mask out dark dark outliers
    
    collection = collection.map(function(img){
    var zScore = img.select(shadowSumBands).subtract(irMean).divide(irStdDev);
    var irSum = img.select(shadowSumBands).reduce(ee.Reducer.sum());
    var TDOMMask = zScore.lt(zScoreThresh).reduce(ee.Reducer.sum()).eq(2)
        .and(irSum.lt(shadowSumThresh)).not();
    TDOMMask = TDOMMask.focal_min(dilatePixels);
    return img.addBands(TDOMMask.rename('TDOMMask'))
    return collection
'''  
iniDate = '2010-01-01'
endDate = '2016-12-31'

metadataCloudCoverMax = 75
cloudThresh = 10
dilatePixels = 2
cloudHeights = ee.List.sequence(200,5000,500)
zScoreThresh = -0.75
shadowSumThresh = 0.35
windowSize = 15
                    
geo1 = ee.Geometry.Point([-93.61124038696289,42.162003755958466])
geo2 = ee.Geometry.Point([-93.49742889404297,42.145078043817534])

lc8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_RT_TOA')
le7 = ee.ImageCollection('LANDSAT/LE07/C01/T1_RT_TOA')
s2 = ee.ImageCollection('COPERNICUS/S2')

optical = mergeOptical(geo1, iniDate, endDate)
#sys.exit()
#optical = simpleTDOM2(optical,zScoreThresh,shadowSumThresh,dilatePixels);

    
raw = lc8.filterDate('2015-01-01', '2015-10-30')#.filterBounds(geometry)

vi = optical.map(addNDVI).select('nd')
vi2 = raw.map(addNDVI2).select('nd')



nd = pd.DataFrame()

#sys.exit()


i = 0
for geo in [geo1, geo2]:
    
    #optical = mergeOptical(geo, iniDate, endDate)
    
    #vi = optical.map(addNDVI).select('nd')
    # region values as generated by getRegion
    region = vi.getRegion(geo, 30).getInfo()
    
    # stuff the values in a dataframe for convenience      
    df = pd.DataFrame.from_records(region[1:len(region)])
    # use the first list item as column names
    df.columns = region[0]
    
    # divide the time field by 1000 as in milliseconds
    # while datetime takes seconds to convert unix time
    # to dates
    df.time = df.time / 1000
    df['time'] = pd.to_datetime(df['time'], unit = 's')
    df.rename(columns = {'time': 'date'}, inplace = True)
    df.sort_values(by = 'date')
    
    df.index = df['date']
    df['interp'] = df['nd'].interpolate('nearest')


    sys.exit()
    
    nd['date_'+str(i)] = df['date']
    nd['nd_'+str(i)] = df['nd']
    
    i += 1

    
#nd[['nd_0','nd_1']].plot(linestyle = '-')

#statsmodels.tsa.stattools.grangercausalitytests(nd[['nd_0','nd_1']], 5, addconst=True, verbose=True)

#filled in nans 
ndfill = nd.fillna(method='ffill')




'''

var optical = mergeOptical(geo, iniDate, endDate);
optical = simpleTDOM2(optical,zScoreThresh,shadowSumThresh,dilatePixels);

var rgb_vis = {min:0, max:0.3, bands:['B4','B3','B2']};
function addNDVI(image) {
  var ndvi = image.normalizedDifference(['B5', 'B4']);
  return image.addBands(ndvi);
}
var filtered = L8.filterDate('2015-01-01', '2015-10-30');
var with_ndvi = filtered.map(addNDVI);
var greenest = with_ndvi.qualityMosaic('nd');
Map.addLayer(filtered.median(), rgb_vis, 'RGB (median)');
Map.addLayer(greenest, rgb_vis, 'RGB (greenest pixel)');
print(Chart.image.series(with_ndvi.select('nd'), roi));
'''