#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 19:43:03 2017

@author: wellenbu
"""

import math
import datetime
import ee
import sys
from scipy import stats
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.tsa.stattools
from scipy import signal
#from PIL import ImageTk
#import ee.mapclient


ee.Initialize()


def addNDVI(image):
    ndvi = image.normalizedDifference(['nir','red']).rename(['nd'])
    #ndvi = image.normalizedDifference(['nir', 'red'])
    #return image.addBands(ndvi)
    return ndvi.set("system:time_start",image.get("system:time_start"))
    
def addEVI(image):
   evi = image.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
    {'NIR': image.select(['nir']),
    'RED': image.select(['red']),
    'BLUE': image.select(['blue'])}).float().rename(['nd'])
   return evi.set("system:time_start",image.get("system:time_start"))
    
def lsCloudMask(img):
  blank = ee.Image(0)
  scored = ee.Algorithms.Landsat.simpleCloudScore(img)
  clouds = blank.where(scored.select(['cloud']).lte(cloudThresh),1)
  return img.updateMask(clouds).set("system:time_start",img.get("system:time_start"))
'''
def viirsQuality(img):
    def getQABits(image, start, end, newName):
        # Compute the bits we need to extract.
        pattern = 0;
        for i in range(start,end+1):
           pattern += math.pow(2, i);
        
        return image.select([0], [newName])\
                      .bitwiseAnd(int(pattern))\
                      .rightShift(start)
                      
    qf = img.select('QF1')
    quality = getQABits(qf,0,3,'quality')
    return img.updateMask(quality.eq(3))
''' 
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
    viirsrename = viirs.filterBounds(studyArea)\
                    .filterDate(t1,t2)\
                    .select(['M5','M5','M5','M7','M5','M5'],['blue','green','red','nir','swir1','swir2'])
                    #.map(viirsQuality)
           
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
                    
    out = ee.ImageCollection(le7rename.merge(lc8rename))#.merge(st2rename))#.merge(viirsrename))
    out = ee.ImageCollection(st2rename)#.merge(lc8rename))
    out = ee.ImageCollection(viirsrename)
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


    

def simpleTDOM2(collection,zScoreThresh,shadowSumThresh,dilatePixels):
    def bustShadows(img):
        zScore = img.select(shadowSumBands).subtract(irMean).divide(irStdDev);
        irSum = img.select(shadowSumBands).reduce(ee.Reducer.sum());
        TDOMMask = zScore.lt(zScoreThresh).reduce(ee.Reducer.sum()).eq(2)\
            .And(irSum.lt(shadowSumThresh)).Not();
        TDOMMask = TDOMMask.focal_min(dilatePixels);
        return img.addBands(TDOMMask.rename(['TDOMMask']))
    
    shadowSumBands = ['nir','swir1']
    #Get some pixel-wise stats for the time series
    irStdDev = collection.select(shadowSumBands).reduce(ee.Reducer.stdDev())
    irMean = collection.select(shadowSumBands).mean()

    collection = collection.map(bustShadows)
    #Mask out dark dark outliers
    
    return collection

# add VV/VH channel to the SAR image
def MWVImap(img):
  vv = img.select('sum');
  #var vvvh = vv.rename(['vegindex']);
  vvvh = vv.multiply(0.0679).add(1.1221).rename(['nd']);
  return img.addBands(vvvh);

def toNatural(img):
  return ee.Image(10.0).pow(img.select(0).divide(10.0))

def toDB(img):
  return ee.Image(img).log10().multiply(10.0)
    
def despeckle(img):
  t = ee.Date(img.get('system:time_start'))
  angles = img.select('angle')
  # The RL speckle filter
  img = toNatural(img)
  # img must be in natural units, i.e. not in dB!
  # Set up 3x3 kernels 
  weights3 = ee.List.repeat(ee.List.repeat(1,3),3)
  kernel3 = ee.Kernel.fixed(3,3, weights3, 1, 1, False)

  mean3 = img.reduceNeighborhood(ee.Reducer.mean(), kernel3)
  variance3 = img.reduceNeighborhood(ee.Reducer.variance(), kernel3)

  # Use a sample of the 3x3 windows inside a 7x7 windows to determine gradients and directions
  sample_weights = ee.List([[0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0],\
                            [0,1,0,1,0,1,0], [0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0]])

  sample_kernel = ee.Kernel.fixed(7,7, sample_weights, 3,3, False)

  # Calculate mean and variance for the sampled windows and store as 9 bands
  sample_mean = mean3.neighborhoodToBands(sample_kernel) 
  sample_var = variance3.neighborhoodToBands(sample_kernel)

  # Determine the 4 gradients for the sampled windows
  gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs()
  gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs())
  gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs())
  gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs())

  # And find the maximum gradient amongst gradient bands
  max_gradient = gradients.reduce(ee.Reducer.max())

  # Create a mask for band pixels that are the maximum gradient
  gradmask = gradients.eq(max_gradient)

  # duplicate gradmask bands: each gradient represents 2 directions
  gradmask = gradmask.addBands(gradmask)

  # Determine the 8 directions
  directions = sample_mean.select(1).subtract(sample_mean.select(4)).gt(sample_mean.select(4).\
                                  subtract(sample_mean.select(7))).multiply(1)
  
  directions = directions.addBands(sample_mean.select(6).subtract(sample_mean.select(4)).\
                                   gt(sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2))
  
  directions = directions.addBands(sample_mean.select(3).subtract(sample_mean.select(4)).\
                                   gt(sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3))
  
  directions = directions.addBands(sample_mean.select(0).subtract(sample_mean.select(4)).\
                                   gt(sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4))
  # The next 4 are the not() of the previous 4
  directions = directions.addBands(directions.select(0).Not().multiply(5))
  directions = directions.addBands(directions.select(1).Not().multiply(6))
  directions = directions.addBands(directions.select(2).Not().multiply(7))
  directions = directions.addBands(directions.select(3).Not().multiply(8))

  # Mask all values that are not 1-8
  directions = directions.updateMask(gradmask)

  # "collapse" the stack into a singe band image (due to masking, each pixel has just one value (1-8) in it's directional band, and is otherwise masked)
  directions = directions.reduce(ee.Reducer.sum())  

  sample_stats = sample_var.divide(sample_mean.multiply(sample_mean))

  # Calculate localNoiseVariance
  sigmaV = sample_stats.toArray().arraySort().arraySlice(0,0,5).arrayReduce(ee.Reducer.mean(), [0])

  # Set up the 7*7 kernels for directional statistics
  rect_weights = ee.List.repeat(ee.List.repeat(0,7),3).cat(ee.List.repeat(ee.List.repeat(1,7),4))

  diag_weights = ee.List([[1,0,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,1,0,0,0,0],\
                          [1,1,1,1,0,0,0], [1,1,1,1,1,0,0], [1,1,1,1,1,1,0], [1,1,1,1,1,1,1]]);

  rect_kernel = ee.Kernel.fixed(7,7, rect_weights, 3, 3, False)
  diag_kernel = ee.Kernel.fixed(7,7, diag_weights, 3, 3, False)

  # Create stacks for mean and variance using the original kernels. Mask with relevant direction.
  dir_mean = img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1))
  dir_var = img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1))

  dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2)))
  dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2)))

  # and add the bands for rotated kernels
  i = 1
  while i < 4:
    dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
    dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
    dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))
    dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))
    i+=1
  # "collapse" the stack into a single band image (due to masking, each pixel has just one value in it's directional band, and is otherwise masked)
  dir_mean = dir_mean.reduce(ee.Reducer.sum())
  dir_var = dir_var.reduce(ee.Reducer.sum())

  # A finally generate the filtered value
  varX = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigmaV)).divide(sigmaV.add(1.0))

  b = varX.divide(dir_var)
  
  # Get a multi-band image bands.
  result = dir_mean.add(b.multiply(img.subtract(dir_mean))).arrayProject([0])\
    .arrayFlatten([['sum']])\
    .float()


  return toDB(result).addBands(angles).set('system:time_start',t.millis())
'''
# Funtion to normalize backscatter values to a nominal angle
# using methods from Nguyen et al., 2015 doi:10.3390/rs71215808
def normalizeBackscatter(img):
  samp = img.sample({
    'region': img.geometry(),
    'scale': 30,
    'numPixels': 250
  });
  angles = ee.Array(samp.aggregate_array(['angle']));
  dB = ee.Array(samp.aggregate_array(sarBands[0]));

  coefs = fitOLS(angles,dB);
  print(coefs);

  norm = ee.Image(30);
  angleRatio = img.select(['angle']).subtract(norm).multiply(coefs.get([0]));
  normBackscatter = img.select(['VV']).subtract(angleRatio);
  return img.select(['angle']).addBands(normBackscatter);
'''

 
def normalizeBackscatter(img):
  norm = ee.Image(30)
  angleRatio = norm.divide(img.select(['angle']))
  normBackscatter = img.select(['sum']).divide(angleRatio)

  return img.select(['angle']).addBands(normBackscatter)
  
def microwv(geo,start,end, pol):
    sar = s1.filterBounds(geo).filterDate(start,end).select([pol,'angle'])\
                .filterMetadata('orbitProperties_pass','equals','ASCENDING').select([pol,'angle'])\
                .map(despeckle).map(normalizeBackscatter).map(MWVImap)

#              .filterMetadata('transmitterReceiverPolarisation','equals',['VV','VH'])\
    #sar = sar.cast({'nd':'float', 'angle':'float', 'sum':'float'}, ['nd', 'angle', 'sum'])
    return sar

##############################################################################
    
iniDate = '2015-01-01'
endDate = '2016-12-31'

metadataCloudCoverMax = 75
cloudThresh = 10
dilatePixels = 2
cloudHeights = ee.List.sequence(200,5000,500)
zScoreThresh = -0.75
shadowSumThresh = 0.35
windowSize = 15

geoms = pd.read_csv('geo.csv')
cols = geoms.columns[3:]
col = 1

                 
#geo1 = ee.Geometry.Point([-93.61124038696289,42.162003755958466])
#geo2 = ee.Geometry.Point([-93.49742889404297,42.145078043817534])

viirs = ee.ImageCollection('NOAA/VIIRS/VNP09GA/001')
lc8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_RT_TOA')
le7 = ee.ImageCollection('LANDSAT/LE07/C01/T1_RT_TOA')
s2 = ee.ImageCollection('COPERNICUS/S2')
s1 = ee.ImageCollection('COPERNICUS/S1_GRD')

#optical = mergeOptical(geo1, iniDate, endDate)
#mwvVi = microwv(geo,iniDate, endDate).select(sarBand)

#sys.exit()
#optical = simpleTDOM2(optical,zScoreThresh,shadowSumThresh,dilatePixels)

#vi = optical.map(addNDVI).select('nd')
#mwvVi = microwv.select('vegindex')

#imgSeries = ee.ImageCollection(vi.merge(mwvVi))
#sys.exit(1)

  
#raw = lc8.filterDate('2015-01-01', '2015-10-30')#.filterBounds(geometry)
#vi2 = raw.map(addNDVI2).select('nd')


nd = pd.DataFrame()

#sys.exit()


i = 0
for row in geoms.loc[geoms[cols[col]] == 1][['Lat','Lon']].iterrows():
#for geo in [geo1]:
    
    geo = ee.Geometry.Point([row[1]['Lon'],row[1]['Lat']])
    
    optical = mergeOptical(geo, iniDate, endDate)
    optical = simpleTDOM2(optical,zScoreThresh,shadowSumThresh,dilatePixels)
    optVi = optical.map(addNDVI).select('nd')
    
    mwvVi = microwv(geo,iniDate,endDate,'VH').select('nd')
    
    vi = ee.ImageCollection(optVi.merge(mwvVi))
    
    #sys.exit(0)
    
    
    #optical = mergeOptical(geo, iniDate, endDate)
    
    #vi = optical.map(addNDVI).select('nd')
    # region values as generated by getRegion
    regionO = optVi.getRegion(geo, 30).getInfo()
    regionM = mwvVi.getRegion(geo, 30).getInfo()
    #region = vi.getRegion(geo, 30).getInfo()
    
    #sys.exit()
    
    
    # stuff the values in a dataframe for convenience      
    df1 = pd.DataFrame.from_records(regionO[1:len(regionO)])
    # use the first list item as column names
    df1.columns = regionO[0]

    df2 = pd.DataFrame.from_records(regionM[1:len(regionM)])
    # use the first list item as column names
    df2.columns = regionM[0]
    #sys.exit()
    
    # divide the time field by 1000 as in milliseconds
    # while datetime takes seconds to convert unix time
    # to dates
    df1.time = df1.time / 1000
    df1['time'] = pd.to_datetime(df1['time'], unit = 's')
    df1.rename(columns = {'time': 'date'}, inplace = True)
    df1 = df1.sort_values(by = 'date')
    #sys.exit()
    del df1['id']
    df1 = df1.groupby(df1['date']).mean()
    #df1.index = df1['date']
    #df1 = df1.drop_duplicates(keep='last')
    #df1 = df1.groupby(pd.TimeGrouper(freq='d')).mean()
    
    df2.time = df2.time / 1000
    df2['time'] = pd.to_datetime(df2['time'], unit = 's')
    df2.rename(columns = {'time': 'date'}, inplace = True)
    df2 = df2.sort_values(by = 'date')
    del df2['id']
    df2 = df2.groupby(df2['date']).mean()
    #df2.index = df2['date']
    #df2 = df2.drop_duplicates(keep='last')
    #df2 = df2.groupby(pd.TimeGrouper(freq='d')).mean()
        
    ## Regress backscatter to evi   
    a = pd.DataFrame(signal.detrend(df1['nd'].dropna()), index = df1['nd'].dropna().index)
    c = pd.DataFrame(signal.detrend(df2['nd'].dropna()), index = df2['nd'].dropna().index)
    
    mo = pd.concat([a.iloc[:,0],c.iloc[:,0]],1).groupby(pd.TimeGrouper(freq='M')).mean().dropna()
    plt.scatter(mo.iloc[:,0],mo.iloc[:,1])
    m,b,r,p,e = linregress(mo.iloc[:,1].values,mo.iloc[:,0].values)
    print r**2
    #sys.exit()
    
    mo = pd.concat([df1['nd'],df2['nd']],1).groupby(pd.TimeGrouper(freq='M')).mean().dropna()
    plt.scatter(mo.iloc[:,0],mo.iloc[:,1], color = 'red')
    
    plt.plot([-0.5,1],[-0.5,1],'r--')
    m,b,r,p,e = linregress(mo.iloc[:,1].values,mo.iloc[:,0].values)
    print r**2
    print '\n'
    #sys.exit()
    
    #df.iloc[:,1]*m+b
    #sys.exit()
    ser_no = df1['nd']
    ser_no = pd.DataFrame(signal.detrend(df1['nd'].dropna()), index = df1['nd'].dropna().index)
    ser_no.columns = ['nd']

    ser = pd.concat([df1['nd'],df2['nd']*m+b])
    
    df = pd.DataFrame(ser_no)
    
    #df['date_'+str(i)] = df['date']
    #del df['date']
    df['nd_'+str(i)] = df['nd']
    del df['nd']
    #sys.exit()  
    
    if i == 0:
        nd = df
    else:
        nd = pd.concat([nd,df],axis=1)
    #sys.exit()
    
    i += 1
plist = []
for j in np.arange(i):  
    nd['interp_'+str(j)] = nd['nd_'+str(j)].interpolate('nearest')
for j in np.arange(i):
    nd['smooth_'+str(j)] = smooth(nd['interp_'+str(j)], window_len=30)[15:-14]
       
    plist.append('smooth_'+str(j))
        
nd.iloc[:,12:].plot()

       
       
#nd[['nd_0','nd_1']].plot(linestyle = '-')

#statsmodels.tsa.stattools.grangercausalitytests(nd[['nd_0','nd_1']], 5, addconst=True, verbose=True)

#filled in nans 
# = nd.fillna(method='ffill')
'''
nd['smooth_0'].plot(marker = '.')
nd['smooth_1'].plot(marker = '.')
'''
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

'''
    b = []
    i = 0
    while i < len(regionM[1:]):
        a = regionM[i+1][3]
        j = 0
        while j < len(regionO[1:]):
            if float(regionM[i+1][3]) > regionO[j+1][3]-864000000.0 and a < regionO[j+1][3]+864000000.0:
                print regionO[j+1][3]
                b.append([regionM[i+1][4],regionO[j+1][4]])
                #print a,regionO[j+1][3]
                #sys.exit()
            j+=1   
        i+=1
    '''