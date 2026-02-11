import ee
import os, json, datetime as dt
import geopandas as gpd
import geemap
import pandas as pd
import datetime as dt
import rasterio as rio
from rasterio.mask import mask
from rasterio.features import shapes
import numpy as np
from shapely.geometry import shape

##helper functions
def create_gee_featureCollection(geometry):
    ##converting district shapefile to suitable format for gee
    #converting district geo to 4326
    features = []
    gj = json.loads(geometry.to_json())
    for feat in gj["features"]:
                    # Keep only lightweight properties
                    props = feat.get("properties", {})
                    geom = feat["geometry"]
                    features.append(ee.Feature(ee.Geometry(geom), props))

    sites_fc = ee.FeatureCollection(features)
    region_geom = sites_fc.first().geometry()
    print(f'Converted geometry to GEE FeatureCollection')
    return region_geom

def get_cropland_mask(district_4326, esa_cci_lulc_fp, buffer_width = 0.005):
    cropland_class = [10, 20, 30]
    src = rio.open(esa_cci_lulc_fp) 

    # IMPORTANT: capture both data and transform
    cropland_data, cropland_transform = mask(src, district_4326.geometry, crop=True)

    # Convert landcover values into 0/1 cropland mask
    cropland_mask = np.isin(cropland_data, cropland_class).astype(np.uint8)

    # Remove the band dimension (convert from (1, H, W) → (H, W))
    cropland_mask_2d = cropland_mask.squeeze()

    # Convert raster mask into polygons
    polygons = []
    for geom, value in shapes(cropland_mask_2d, mask=cropland_mask_2d==1, transform=cropland_transform):
        if value == 1:  # only cropland
            polygons.append(shape(geom))

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")

    # Dissolve all cropland polygons into 1 MultiPolygon
    cropland_single = gdf.dissolve()   # dissolves all rows into one geometry

    #buffer cropland
    cropland_single = cropland_single.buffer(buffer_width)
    # If you want a clean geometry object:
    cropland_geom = cropland_single.geometry.iloc[0]

    # Optional: save to shapefile / geojson
    # cropland_single.to_file("ernakulam_cropland_single.geojson", driver="GeoJSON")

    return cropland_single