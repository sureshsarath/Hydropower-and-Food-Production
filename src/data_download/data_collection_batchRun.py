import geopandas as gpd
import matplotlib.pyplot as plt
import os
import ee
from tqdm import tqdm
import warnings
import traceback
import pandas as pd
from gee_helpers import create_gee_featureCollection, get_cropland_mask

warnings.filterwarnings("ignore")

# ee.Authenticate()
ee.Initialize()

sites_fp = '../Outputs/study_sites_v1_CA_stats.geojson'
downstream_reach_geom_fp_struct = '../Outputs/CommandArea/{}_{}_command_area.geojson'
esa_cci_folder = '../Scripts/Core/revisions_GEN_Exam/supporting_data/ESA_CCI'
esa_cci_file_format = 'ESACCI-LC-L4-LCCS-Map-300m-P1Y-{}-v2.0.7.tif'
save_folder = '../Scripts/Core/revisions_GEN_Exam/outputs/revisions_gpp_ndvi'

start_processing_counter = 0 #th site
processing_batch_length = 30 #sites


def _landast_scale(img):
            scaled = img.multiply(0.0000275).add(-0.2)
            # copy props (returns Element) then cast back to Image before returning
            scaled = ee.Image(scaled).copyProperties(img, img.propertyNames())
            return ee.Image(scaled)
        
def _add_ndwi_landsat(img, band_info):
            scaled = _landast_scale(img.select(band_info[:2]))
            ndwi = scaled.normalizedDifference(band_info[:2]).rename('NDWI')
            return img.addBands(ndwi)
        



sites = gpd.read_file(sites_fp)
#data extraction dates
start_date = "2001-01-01"
end_date = "2022-12-30"
landsat_scale = 500 #m

##batch run logic
start_year = pd.to_datetime(start_date).year
end_year = pd.to_datetime(end_date).year
years = list(range(start_year, end_year + 1))

modis_collection_id = 'CAS/IGSNRR/PML/V2_v018'
l8_collection_id = 'LANDSAT/LC08/C02/T1_L2'
l7_collection_id = 'LANDSAT/LE07/C02/T1_L2'
l8_bands = ['SR_B3', 'SR_B6', 'QA_PIXEL']
l7_bands = ['SR_B2', 'SR_B5', 'QA_PIXEL']

error_list = {}
site_counter = 1
for study_site in sites[start_processing_counter:start_processing_counter+processing_batch_length].iterrows():
    dam_name = study_site[1]['DAM_NAME_left']
    dam_grand_ID = study_site[1]['GRAND_ID']
    
    save_fp = os.path.join(save_folder, f'{dam_grand_ID}{dam_name}.csv')
    print(f'Processing site: {dam_name} (ID: {dam_grand_ID})')
    if os.path.exists(save_fp):
        print(f'Saved file already exists for site {dam_name}. Skipping processing.')
        continue
    try:
        downstream_reach_geom = gpd.read_file(downstream_reach_geom_fp_struct.format(dam_grand_ID,dam_name))
        downstream_reach_geom_4326 = downstream_reach_geom.to_crs(epsg=4326)
        
        data_monthly_allYears = pd.DataFrame()
        
        for year in years:
            
            esa_cci_file = os.path.join(esa_cci_folder, esa_cci_file_format.format(year)) if year <=2015 else os.path.join(esa_cci_folder, esa_cci_file_format.format(2015))

            cropland_mask = get_cropland_mask(downstream_reach_geom_4326, esa_cci_file, buffer_width = 0.0001)
            cropland_mask_geometry_simplified = cropland_mask.simplify(tolerance=0.001, preserve_topology=True)

            cropland_geeFeature = create_gee_featureCollection(cropland_mask_geometry_simplified)

            #obtaining gee based modis and landsat datasets
            start_date_year = f"{year}-01-01"
            end_date_year = f"{year}-12-31"
            
            modis_gpp = (ee.ImageCollection(modis_collection_id)
                            .filterDate(start_date_year, end_date_year)
                            .filterBounds(cropland_geeFeature)
                            .select(['GPP'])
                            .map(lambda img: img.clip(cropland_geeFeature)))
            
            if year < 2014:
                print(f'[[{dam_name}]] Before 2014. Using Landsat 7.')
                landsat_base = (ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
                    .filterDate(start_date_year, end_date_year)
                    .filterBounds(cropland_geeFeature)
                    .select(l7_bands)   
                    .map(lambda img: img.clip(cropland_geeFeature))) 
                

                
            else:
                print(f'[[{dam_name}]] After 2014. Using Landsat 8.')

                landsat_base = (ee.ImageCollection(l8_collection_id)
                    .filterDate(start_date_year, end_date_year)
                    .filterBounds(cropland_geeFeature)
                    .select(l8_bands)
                    .map(lambda img: img.clip(cropland_geeFeature)))
            
            landsat_with_ndwi = landsat_base.map(lambda img: _add_ndwi_landsat(img, l8_bands) if year >=2014 else _add_ndwi_landsat(img, l7_bands))
            # landsat_with_cloud = landsat_with_ndwi.map(_get_cloud_cover_percentage)
            print(f'[[{dam_name}]] Landsat images with NDWI added.')
            
            rows = []  # collect per-month records here
            start = ee.Date.fromYMD(year, 1, 1)
            end   = ee.Date.fromYMD(year + 1, 1, 30) 
            n_months = end.difference(start, 'month')
            months = ee.List.sequence(0, 11)

            print(f'[[{dam_name}]] Starting monthly loop')

            for i in months.getInfo():
                i = ee.Number(i)
                m_start = start.advance(i, 'month')
                m_end   = m_start.advance(1, 'month')
                date_str = m_start.format('YYYY-MM').getInfo()  # monthly label
                print(f'[[{dam_name}]] ----------------')
                print(f'[[{dam_name}]] Month: {date_str}')
                print(f'[[{dam_name}]] ----------------')

                
                #modis processing
                monthly_img = modis_gpp.filterDate(m_start, m_end).median()
                gpp_mean = monthly_img.reduceRegion(
                            reducer = ee.Reducer.mean(),
                            geometry = cropland_geeFeature,
                            scale = 500,         # 500 m PML resolution
                            maxPixels = 1e14
                        ).get('GPP')

                gpp_mean_val = gpp_mean.getInfo()
                gpp_mean_month = gpp_mean_val*30*10
                print(f'[[{dam_name}]] Computed MODIS GPP for month. GPP = {gpp_mean_month:.2f} gC/m2/month')
                #landsat processing
                ndwi_month = landsat_with_ndwi.filterDate(m_start, m_end).select('NDWI')
                if ndwi_month.size().getInfo() == 0:
                    print(f'{date_str}: No images')
                    rows.append({
                        'Date': date_str,
                        'water_area_km2': None,
                        'modis_gpp': gpp_mean_month,
                        'landsat_cloud_pct': None   
                    })
                    continue
                ndwi_med = ndwi_month.median()
                print(f'[[{dam_name}]] Loaded NDWI median for month. Computing water area...')

                water_mask = ndwi_med.gte(0)
                water_area = ee.Algorithms.If(
                    ndwi_month.size().gt(0),
                    ee.Image.pixelArea().updateMask(water_mask).reduceRegion(
                        reducer=ee.Reducer.sum(), geometry=cropland_geeFeature,
                        scale=landsat_scale, maxPixels=1e15, bestEffort=False, tileScale=2
                    ).get('area'),
                    None
                )
                water_area_m2 = water_area.getInfo()  # may be None
                water_area_km2 = (water_area_m2 / 1e6) if water_area_m2 is not None else None
                
                #cloud cover
                landsat_month = landsat_with_ndwi.filterDate(m_start, m_end)

                cloud_pct = ee.Algorithms.If(
                    landsat_month.size().gt(0),
                    landsat_month.aggregate_mean('CLOUD_COVER_LAND'),  # or aggregate_mean
                    None
                )
                cloud_pct_val = cloud_pct.getInfo() 
                
                print(f'[[{dam_name}]] Computed water area for month. Area  = {water_area_km2:.2f} km2')
                print(f'[[{dam_name}]] Average cloud cover for the month = {cloud_pct_val:.2f} %')
                rows.append({
                        'Date': date_str,
                        'water_area_km2': water_area_km2,
                        'modis_gpp': gpp_mean_month,
                        'landsat_cloud_pct': cloud_pct_val   
                    })
            
            df = pd.DataFrame(rows).sort_values('Date').reset_index(drop=True)
            df['date'] = df['Date'].astype(str) + '-01'
            df['date'] = pd.to_datetime(df['date'])
            df.drop(columns=['Date'], inplace=True)
            data_monthly_allYears = pd.concat([data_monthly_allYears, df], axis=0).reset_index(drop=True)   
        #save data
        data_monthly_allYears.to_csv(save_fp, index=False)
            
    except Exception as e:
        error_list[dam_name] = traceback.format_exc()
        print(f'Error processing site {dam_name}: {e}')
        

    
    