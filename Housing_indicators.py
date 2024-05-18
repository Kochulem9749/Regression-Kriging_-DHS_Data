

#Unet1 environment
import os
from os import path as op
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler


import statsmodels.api as sm


from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR



from sklearn.model_selection import train_test_split


from sklearn import metrics

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


import rasterio
import rasterio as rio

from rasterio.features import rasterize
from rasterio.transform import from_origin

import patchify
from patchify import patchify,unpatchify


import geopandas as gpd
from shapely.geometry import Point


from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt

from scipy.optimize import minimize

# Specify the path of the directory you want to set as the working directory
directory_path = r"D:\OneDrive - United Nations\Attachments\Projects\DHS_Data"

# Change the current working directory to the specified directory
os.chdir(directory_path)

# Read the Excel file
df = pd.read_excel('Training_data_with_covariates_V1.xlsx')

# Replace -9999 with NaN in the entire DataFrame
#df1 = df[df != -9999].dropna()

##Then drop all the rows that have NAN values
##df1 = df1.dropna()

print(df.columns)



training_data_coords_df=df[['LATNUM', 'LONGNUM']]

##The response variable- Y-variable
y_var_name='Total_Impr'

##Total_Impr
##Electric_1
##Improved_w

DF_Subset=df[[y_var_name,'Poverty_in', 'Pregnancy',
       'Pop_den', 'Birth_Atte', 'Postnatal', 'Antenatal', 'Literacy_F',
       'Births_Pro']]

print(DF_Subset.columns)



#Extract dependent and numerical independent variables
y=DF_Subset[y_var_name]


##This represent numerical independent variables
x_columns=DF_Subset.columns[1:]
x_num =DF_Subset.loc[:,x_columns]

#print(x_num)

# perform a standardization transform on the numerical independent variables
# define the pipeline

pipeline = StandardScaler()
df_scaled=pipeline.fit_transform(x_num.values)

# convert the array back to a dataframe
#Transfromed numerical variables through standardization transform
transformed_numeric_df = pd.DataFrame(df_scaled,columns=x_num.columns)


##Concatenate the both dependent variable,standardized numerical values and hot-encoded categorical values
combine_data=pd.concat([y,transformed_numeric_df], axis=1)


##drop any NAN values that might be existing in any of the rows
Transformed_training_data=combine_data.dropna()

print(Transformed_training_data)

##Create x and y variables from the preprocessed data
y_vars=Transformed_training_data[y_var_name]

x_cols=Transformed_training_data.columns[1:]
X_vars=Transformed_training_data.loc[:,x_cols]


import statsmodels.api as sm

# Assuming you have your DataFrame named 'data' with 'y' as the dependent variable and 'X' as the independent variables

# Add a constant term to the independent variables for the intercept term in the OLS model

X = sm.add_constant(X_vars)
y=y_vars
# Fit the OLS model
model = sm.OLS(y, X).fit()

# Print the summary to see the p-values and other statistics
#print(model.summary())

# Access the p-values of the model parameters
p_values = model.pvalues

# Create a DataFrame from the p-values
p_values_df = pd.DataFrame(p_values, columns=['P-value'])

# Display DataFrame
print(p_values_df)


# Assuming p_values_df is your DataFrame containing the p-values
# Filter variables with p-values less than 0.05
significant_variables = p_values_df[p_values_df['P-value'] < 0.05]



# Extract variable names from the index of the DataFrame
variable_names = significant_variables.index.tolist()

X_Significant_vars=X_vars[variable_names[1:]]

# Display variable names
print("significacant variables are: {0}".format(X_Significant_vars.columns))






#Split data into training and testing sets
X_train,X_test,Y_train,Y_test=train_test_split(X_Significant_vars.values,y_vars.values,test_size=0.3,random_state=1)



#"""
#Create a new model by applying the best parameters obtained above
#rf_regr = RandomForestRegressor(n_estimators=5000)
rf_regr = RandomForestRegressor(n_estimators=5000)



# Train the model on the training data
rf_regr.fit(X_train, Y_train)

# Make predictions on the test data
y_predictions = rf_regr.predict(X_test)


#print(y_predictions)

#Calculating the various accuracy measures rf
mae = metrics.mean_absolute_error(Y_test, y_predictions)
mse = metrics.mean_squared_error(Y_test, y_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, y_predictions)
mape = np.mean(np.abs((Y_test - y_predictions)/Y_test))*100


#printing the accuracy measures, rounded to 2 dp
print("Results of Random Forest Model:")
mae = round(mae, 2)
mse = round(mse, 2)
rmse = round(rmse, 2)
r2 = round(r2, 2)
mape = round(mape, 2)

print("RF_MAE:",mae)
print("RF_MSE:", mse)
print("RF_RMSE:", rmse)
print("RF_R-Squared:", r2)
print("RF_MAPE:", mape)





# Creating SVR model
##svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_rbf = SVR(kernel='rbf',C=100)

# Training the SVR model
svr_rbf.fit(X_train, Y_train)

# Predicting on training data
y_pred_svr = svr_rbf.predict(X_test)



#Calculating the various accuracy measures rf
mae_SVR = metrics.mean_absolute_error(Y_test, y_pred_svr)
mse_SVR = metrics.mean_squared_error(Y_test, y_pred_svr)
rmse_SVR = np.sqrt(mse_SVR)
r2_SVR = r2_score(Y_test, y_pred_svr)
mape_SVR = np.mean(np.abs((Y_test - y_pred_svr)/Y_test))*100


#printing the accuracy measures, rounded to 2 dp
print("Results of Random Forest Model:")
mae_SVR = round(mae_SVR, 2)
mse_SVR = round(mse_SVR, 2)
rmse_SVR = round(rmse_SVR, 2)
r2_SVR = round(r2_SVR, 2)
mape_SVR = round(mape_SVR, 2)

print("SVR_MAE:",mae_SVR)
print("SVR_MSE:", mse_SVR)
print("SVR_RMSE:", rmse_SVR)
print("SVR_R-Squared:", r2_SVR)
print("SVR_MAPE:", mape_SVR)


Births_pp = op.join(directory_path)+'/Covariates/Projected_covariates/KEN_births_pp_v2_2015.tif'
Literacy_Female =op.join(directory_path)+'/Covariates/Projected_covariates/KEN_literacy_Female.tif'
Antenatal_care = op.join(directory_path)+'/Covariates/Projected_covariates/KEN_MNH_antenatal_care.tif'
Postnatal_care = op.join(directory_path)+'/Covariates/Projected_covariates/KEN_MNH_postnatal_care.tif'
Skilled_birth_attendance =op.join(directory_path)+'/Covariates/Projected_covariates/KEN_MNH_skilled_birth_attendance.tif'
Population_density = op.join(directory_path)+'/Covariates/Projected_covariates/ken_population_density_2020_1km_UNadj.tif'
Pregnancy = op.join(directory_path)+'/Covariates/Projected_covariates/KEN_pregnancy_proportion_v2_2015.tif'
Poverty = op.join(directory_path)+'/Covariates/Projected_covariates/ken08povmpi_Multidimensional.tif'





#X_train,X_test,Y_train,Y_test
#Ordinary Kriging of regression results residuals
y_pred_RF = rf_regr.predict(X_Significant_vars.values)


residuals_RF_df = pd.DataFrame(y_vars.values - y_pred_RF)



residuals_RF_df.columns=['residuals']

Coords_Residuals_df=pd.concat([training_data_coords_df,residuals_RF_df],axis=1)

Coords_Residuals_df.columns=['latitude', 'longitude','residuals']


#print(Coords_Residuals_df)
#Function to calculate haversine distance between two points



# Function to calculate haversine distance between two points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    return distance

# Empirical variogram calculation
def empirical_variogram(df, lag_bins):
    distances = []
    residual_diff = []
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            dist = haversine(df['latitude'].iloc[i], df['longitude'].iloc[i], df['latitude'].iloc[j], df['longitude'].iloc[j])
            distances.append(dist)
            residual_diff.append((df['residuals'].iloc[i] - df['residuals'].iloc[j])**2)
    lag_distance_indices = np.digitize(distances, lag_bins)
    semivariance = [np.mean([residual_diff[k] for k in range(len(residual_diff)) if lag_distance_indices[k] == i]) for i in range(1, len(lag_bins))]
    return semivariance

# Variogram model
def variogram_model(params, lag_distances):
    range_, sill, nugget = params
    return nugget + sill * (1.0 - np.exp(-3.0 * lag_distances / (range_ / 3.0)))

# Objective function for parameter estimation
def objective_function(params, lag_distances, empirical_semivariance):
    model_semivariance = variogram_model(params, lag_distances)
    return np.sum((empirical_semivariance - model_semivariance)**2)

# Initial guess for parameters
initial_guess = [10, 1, 0.1]  # range, sill, nugget

# Binning logic for variogram
lag_bins = np.arange(0, 100, 10)

# Calculate empirical variogram
empirical_semivariance = empirical_variogram(Coords_Residuals_df, lag_bins)

# Fit variogram model using optimization
result = minimize(objective_function, initial_guess, args=(lag_bins[1:], empirical_semivariance))
best_params = result.x

# Plot empirical and fitted variograms
plt.plot(lag_bins[1:], empirical_semivariance, label='Empirical Variogram')
plt.plot(lag_bins[1:], variogram_model(best_params, lag_bins[1:]), label='Fitted Variogram')
plt.xlabel('Lag Distance')
plt.ylabel('Semivariance')
plt.title('Variogram')
plt.legend()
plt.show()

# Manual or algorithmic selection of simplified parameters based on the plot
# You can adjust these parameters based on your observations
simplified_params = [round(best_params[0]), best_params[1], best_params[2]]

# Use the simplified parameters for ordinary kriging at new locations
ok = OrdinaryKriging(
    x=Coords_Residuals_df['longitude'],  # longitudes
    y=Coords_Residuals_df['latitude'],   # latitudes
    z=Coords_Residuals_df['residuals'],
    variogram_model='exponential',#Adjust according to the shapeof the variograms optaions are; exponential,Gaussian,Circular,Spherical,Linear
    variogram_parameters={'sill': simplified_params[1], 'range': simplified_params[0], 'nugget': simplified_params[2]},
    ##variogram_parameters={'sill': 1070, 'range': 65, 'nugget': 850},
    verbose=False
)


# variogram_model='exponential
#sill -1070
#range-65
#nugget - 850

# Perform ordinary kriging at the new locations
kriging_results, kriging_variance = ok.execute('points', [Coords_Residuals_df['longitude']], [Coords_Residuals_df['latitude']])


print(kriging_results)
# Step 4: Add kriging results to regression prediction to obtain final prediction
final_prediction_OK = y_pred_RF + kriging_results

# Print or use final_prediction as needed
#print("final_prediction_OK:", final_prediction_OK)


final_prediction_OK_df=pd.DataFrame(final_prediction_OK)
final_prediction_OK_df.columns=['REG_OK']



y_pred_RF_df=pd.DataFrame(y_pred_RF)

y_pred_RF_df.columns=['rf_pred']


Combined_coords_residuals_ok_df=pd.concat([Coords_Residuals_df,y,y_pred_RF_df,final_prediction_OK_df],axis=1)





Residuals_csv=op.join(directory_path)+'/Covariates/Projected_covariates/Outputs/Ordinary_kriging_residuals_'+y_var_name+'.csv'
# Export the GeoDataFrame to a CSV
Combined_coords_residuals_ok_df.to_csv(Residuals_csv,index=False)




#"""
raster_files = [Poverty,Pregnancy,Population_density,Skilled_birth_attendance,Postnatal_care,Antenatal_care,Literacy_Female,Births_pp]



# Initialize an empty list to store individual raster data arrays
raster_data = []

# Loop through raster files, open, and append data to the list
for ind,file_path in enumerate(raster_files):
  with rio.open(file_path) as src:
        data_type = src.meta['dtype']
        nodata_value=src.nodata
        #scale_factor = src.meta['scale_factor']
        data = src.read(1)  # Replace '1' with the desired band index (e.g., 1, 2, 3...)


        raster_data.append(data)



##Extract the xy coordinates for each pixel value center
cord_list=[]
# Open the raster dataset
with rio.open(raster_files[0]) as src:
    # Read the image as an array
    data = src.read()

    # Get the transformation matrix
    transform = src.transform

    # Get the width and height of the raster dataset
    width = src.width
    height = src.height

    crs_input = src.crs

    # Print the CRS
    print("CRS:", crs_input)

    epsg_code = int(crs_input.to_epsg())

    print("epsg_code:", epsg_code)

    # Loop through each pixel and get its corresponding latitude and longitude
    for y in range(height):
        for x in range(width):
            # Convert pixel coordinates to coordinates in the CRS
            Coords_val = src.xy(y, x)
            cord_list.append(Coords_val)






# Create DataFrame for pixel xy location
grid_coordinates = pd.DataFrame(cord_list)

grid_coordinates.columns=['Long','Lat']


# Stack the individual data arrays (covariate values) along the band axis (axis=0)
stacked_data = np.stack(raster_data, axis=0)


# Define the file path where you want to save the array
Ind_vars_array=op.join(directory_path)+'/Covariates/Projected_covariates/Outputs/ind_vars_array.npy'

# Save the NumPy array to the file
np.save(Ind_vars_array, stacked_data)

# Load the .npz file
ind_vars_array_load = np.load(Ind_vars_array)



#Rasterio has shape of count of bands, height, width;
#I want it to be height,width, band count
#Use moveaxis function to move data from one dimension to destination dimension
Large_image=np.moveaxis(ind_vars_array_load,0,2)
#print(Large_image.shape)



#Extract the Numerical values that of the independent variables that requares standardization
a=(Large_image[:,:,0]).ravel() #Poverty
b=(Large_image[:,:,1]).ravel()#Pregnancy
c=(Large_image[:,:,2]).ravel()#Population_density
d=(Large_image[:,:,3]).ravel()#Skilled_birth_attendance
e=(Large_image[:,:,4]).ravel()#Postnatal_care
f=(Large_image[:,:,5]).ravel()#Antenatal_care
g=(Large_image[:,:,6]).ravel()#Literacy_Female
h=(Large_image[:,:,7]).ravel()#Births_pp


stack_X_numerical=np.column_stack([a,b,c,d,e,f,g,h])#Arrange the pixel values of all the bands columnwise

Ind_numr_vars=pd.DataFrame(stack_X_numerical)


Ind_numr_vars.columns=['Poverty_in', 'Pregnancy','Pop_den', 'Birth_Atte', 'Postnatal', 'Antenatal', 'Literacy_F','Births_Pro']





#print(Ind_numr_vars)

# Assuming Stack_img is a DataFrame
# Replace NaN values with the mean of each column
Ind_numr_vars.fillna(Ind_numr_vars.mean(), inplace=True)



#Apply standard scaler similar to what was used during preprocessing of training data
Ind_vars_numer_scaled=pipeline.transform(Ind_numr_vars.values)

Ind_vars_numer_scaled_df=pd.DataFrame(Ind_vars_numer_scaled)

Ind_vars_numer_scaled_df.columns=['Poverty_in', 'Pregnancy','Pop_den', 'Birth_Atte', 'Postnatal', 'Antenatal', 'Literacy_F','Births_Pro']



# Select columns with indices 2 to 5 (columns 3 to 6 in Python zero-based indexing)
Ind_vars_numer_scaled_signif_df = Ind_vars_numer_scaled_df.loc[:, X_Significant_vars.columns]


#print(Ind_vars_numer_scaled_signif_df)

# Clip values to a specified range
clip_min = -1e10  # Adjust the minimum clipping value as needed
clip_max = 1e10   # Adjust the maximum clipping value as needed

##Ind_vars_numer_scaled1 = np.clip(Ind_vars_numer_scaled_signif_df.values, clip_min, clip_max)


Ind_vars_numer_scaled1=Ind_vars_numer_scaled_signif_df.values

img_RF_pred=rf_regr.predict(Ind_vars_numer_scaled1)



predi_RF_df=pd.DataFrame(img_RF_pred)
final_gpd=pd.concat([grid_coordinates,predi_RF_df],axis=1)
final_gpd.columns=['Long','Lat',y_var_name]





# Assuming df is your DataFrame containing latitude and longitude columns
# Create a geometry column from latitude and longitude coordinates
geometry = [Point(xy) for xy in zip(final_gpd['Long'], final_gpd['Lat'])]

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(final_gpd, geometry=geometry)


gdf.crs = 'EPSG:'+str(epsg_code)

#gdf = gdf.to_crs(epsg=4326)

Y_output_shp=op.join(directory_path)+'/Covariates/Projected_covariates/Outputs/RF_'+y_var_name+'.shp'

# Export the GeoDataFrame to a shapefile
gdf.to_file(Y_output_shp, driver='ESRI Shapefile')




# Define the output raster file path
output_tif_path = op.join(directory_path)+'/Covariates/Projected_covariates/Outputs/RF_'+y_var_name + '.tif'

print("Export RF outputs")

# Open a raster dataset to use its metadata (assuming you have an existing raster file)
with rasterio.open(raster_files[0]) as src:
    # Define raster dimensions and transformation
    height, width = src.shape
    transform = src.transform

    # Create a blank array to rasterize into
    rasterized_array = np.zeros((height, width), dtype=np.float32)

    # Rasterize the GeoDataFrame into the blank array using Improved_w values
    rasterized_array = rasterize(
        [(geom, value) for geom, value in zip(gdf['geometry'], gdf[y_var_name])],
        out_shape=(height, width),
        transform=transform,
        fill=np.nan,
        dtype=rasterized_array.dtype,
        all_touched=True
    )

    # Write the rasterized array to a new raster file
    with rasterio.open(
        output_tif_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,  # Number of bands
        dtype=rasterized_array.dtype,  # Data type
        crs=src.crs,  # CRS
        transform=transform,  # Transformation
    ) as dst:
        dst.write(rasterized_array, 1)  # Write the array to the raster file





##RF+OK combined
print("Export RF-OK outputs")
# Perform ordinary kriging at the new locations
#kriging_results_rf, kriging_variance_rf = ok.execute('points', [grid_coordinates['Long']], [grid_coordinates['Lat']])
# Define the number of subsections and the size of each subsection
num_subsections = 500
subsection_size = len(grid_coordinates) // num_subsections

# Initialize lists to store kriging results
kriging_results = []

# Loop over subsections
for i in range(num_subsections):
    # Define the start and end indices for the current subsection
    start_idx = i * subsection_size
    end_idx = (i + 1) * subsection_size if i < num_subsections - 1 else len(grid_coordinates)

    # Extract the subsection of the DataFrame
    subset_df = grid_coordinates.iloc[start_idx:end_idx]


    # Perform kriging at the new locations
    kriging_result, _  = ok.execute('points', subset_df['Long'], subset_df['Lat'])

    # Append the kriging result to the list
    kriging_results.append(kriging_result)
    #print(i)

# Combine kriging results from all subsections
kriging_results_rf = np.concatenate(kriging_results)





##kriging_results, kriging_variance = ok.execute('points', [Coords_Residuals_df['longitude']], [Coords_Residuals_df['latitude']])

# Add kriging results to regression prediction to obtain final prediction
final_prediction_RF_OK = img_RF_pred + kriging_results_rf



predi_RF_OK_df=pd.DataFrame(final_prediction_RF_OK)
final_RF_OK_gpd=pd.concat([grid_coordinates,predi_RF_OK_df],axis=1)
final_RF_OK_gpd.columns=['Long','Lat',y_var_name]





# Assuming df is your DataFrame containing latitude and longitude columns
# Create a geometry column from latitude and longitude coordinates
geometry_RF_OK = [Point(xy) for xy in zip(final_RF_OK_gpd['Long'], final_RF_OK_gpd['Lat'])]

# Create a GeoDataFrame
gdf_RF_OK = gpd.GeoDataFrame(final_RF_OK_gpd, geometry=geometry_RF_OK)


gdf_RF_OK.crs = 'EPSG:'+str(epsg_code)


Y_RF_OK_shp=op.join(directory_path)+'/Covariates/Projected_covariates/Outputs/RF_OK_'+y_var_name+'.shp'

# Export the GeoDataFrame to a shapefile
gdf_RF_OK.to_file(Y_RF_OK_shp, driver='ESRI Shapefile')




# Define the output raster file path
RF_OK_tif_path = op.join(directory_path)+'/Covariates/Projected_covariates/Outputs/RF_OK_'+y_var_name + '.tif'



# Open a raster dataset to use its metadata (assuming you have an existing raster file)
with rasterio.open(raster_files[0]) as src:
    # Define raster dimensions and transformation
    height, width = src.shape
    transform = src.transform

    # Create a blank array to rasterize into
    rasterized_array = np.zeros((height, width), dtype=np.float32)

    # Rasterize the GeoDataFrame into the blank array using Improved_w values
    rasterized_array = rasterize(
        [(geom, value) for geom, value in zip(gdf_RF_OK['geometry'], gdf_RF_OK[y_var_name])],
        out_shape=(height, width),
        transform=transform,
        fill=np.nan,
        dtype=rasterized_array.dtype,
        all_touched=True
    )

    # Write the rasterized array to a new raster file
    with rasterio.open(
        RF_OK_tif_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,  # Number of bands
        dtype=rasterized_array.dtype,  # Data type
        crs=src.crs,  # CRS
        transform=transform,  # Transformation
    ) as dst:
        dst.write(rasterized_array, 1)  # Write the array to the raster file











##SVR Results
print("Export SVR outputs")
##Apply the SVR model to entire study area
img_svr_pred=svr_rbf.predict(Ind_vars_numer_scaled1)


predi_SVR_df=pd.DataFrame(img_svr_pred)
final_SVR_gpd=pd.concat([grid_coordinates,predi_SVR_df],axis=1)
final_SVR_gpd.columns=['Long','Lat',y_var_name]







# Assuming df is your DataFrame containing latitude and longitude columns
# Create a geometry column from latitude and longitude coordinates
geometry_SVR = [Point(xy) for xy in zip(final_SVR_gpd['Long'], final_SVR_gpd['Lat'])]

# Create a GeoDataFrame
gdf_SVR = gpd.GeoDataFrame(final_SVR_gpd, geometry=geometry)


gdf_SVR.crs = 'EPSG:'+str(epsg_code)


Y_output_SVR_shp=op.join(directory_path)+'/Covariates/Projected_covariates/Outputs/SVR_'+y_var_name+'.shp'

# Export the GeoDataFrame to a shapefile
gdf_SVR.to_file(Y_output_SVR_shp, driver='ESRI Shapefile')




# Define the output raster file path
output_SVR_tif_path = op.join(directory_path)+'/Covariates/Projected_covariates/Outputs/SVR_'+y_var_name + '.tif'



# Open a raster dataset to use its metadata (assuming you have an existing raster file)
with rasterio.open(raster_files[0]) as src:
    # Define raster dimensions and transformation
    height, width = src.shape
    transform = src.transform

    # Create a blank array to rasterize into
    rasterized_array = np.zeros((height, width), dtype=np.float32)

    # Rasterize the GeoDataFrame into the blank array using Improved_w values
    rasterized_array = rasterize(
        [(geom, value) for geom, value in zip(gdf_SVR['geometry'], gdf_SVR[y_var_name])],
        out_shape=(height, width),
        transform=transform,
        fill=np.nan,
        dtype=rasterized_array.dtype,
        all_touched=True
    )

    # Write the rasterized array to a new raster file
    with rasterio.open(
        output_SVR_tif_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,  # Number of bands
        dtype=rasterized_array.dtype,  # Data type
        crs=src.crs,  # CRS
        transform=transform,  # Transformation
    ) as dst:
        dst.write(rasterized_array, 1)  # Write the array to the raster file




#"""

