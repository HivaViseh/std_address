import pandas as pd
import googlemaps
from tqdm import tqdm
import time, logging
import pdb
import os
from geopy.geocoders import GoogleV3


input_file_path = r"C:\Users\hviseh\projects\Standardizing_Address"
output_file_path = r"C:\Users\hviseh\projects\Standardizing_Address"
Input_file_name = "CO_incidents.csv"
Output_file_name = "CO_incidents_std_address.csv"

#for i in range (len(df)/100)
df = pd.read_csv(os.path.join(input_file_path, Input_file_name))
df['City'] = df['City'].astype(str)
df['Full address'] = df['Full address'].astype(str)
df['Postal Code'] = df['Postal Code'].astype(str)
#df['postal_code'] = df['PostalCode'].astype(str)
#df['city_name'] = df['City'].astype(str)

#df["Address"] = df['st_num'] + ',' + df['st_name'] + ',' + df['street_type'] + ',' + df['postal_code'] + ',' + df['city_name'] + ',' +'BC' + ',' +'Canada'

df["Address"] = df['Full address'] + ',' + df['Postal Code'] + ',' + df['City'] + ',' +'BC' + ',' +'Canada'


google_api_key = "google_api_key"

def get_address_details_using_google(address):
    try:
        geolocator = GoogleV3(api_key=google_api_key)
        location = geolocator.geocode(address, language='en')

        if location:
            # Get the first result in case 'address_components' is a list
            address_components = location.raw

            
            postal_code = next((item.get('long_name', None) for item in address_components.get('address_components', []) if 'postal_code' in item.get('types', [])), None)
            city_name = next((item.get('long_name', None) for item in address_components.get('address_components', []) if 'locality' in item.get('types', [])), None)
            street_name = next((item.get('short_name', None) for item in address_components.get('address_components', []) if 'route' in item.get('types', [])), None)
            street_number = next((item.get('long_name', None) for item in address_components.get('address_components', []) if 'street_number' in item.get('types', [])), None)
            latitude = location.latitude if location else None
            longitude = location.longitude if location else None


            return street_number, street_name, postal_code, city_name, latitude, longitude
  
        else:
            logging.error(f"Failed to get address details: {address}")
            return None, None, None, None
    except Exception as e:
        logging.error(f"Error geocoding address: {address}. {e}")
        return None, None, None, None
    
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    if row["Address"]:
        result = get_address_details_using_google(row["Address"])
        if result is not None:
            if len(result) >= 6:
                df.at[index, "street_number"] = result[0]
                df.at[index, "street_name"] = result[1]
                df.at[index, "postal_code"] = result[2]
                df.at[index, "city"] = result[3]
                df.at[index, "lat"] = result[4]
                df.at[index, "long"] = result[5]
            else:
                # Log the length of the result tuple if it has fewer than 6 elements
                logging.error(f"Result tuple has fewer than 6 elements: {len(result)}")
                logging.error(f"Failed to process row {index}: {row['Address']}, result: {result}")
        else:
            # Handle the case where geocoding fails completely
            logging.error(f"Failed to process row {index}: {row['Address']}")
    else:
        # Handle the case where the address is empty or None
        logging.warning(f"Skipping empty address in row {index}")


#columns_to_drop = ['st_num', 'st_name', 'street_type', 'postal_code', 'city_name', "Address"]
#df.drop(columns=columns_to_drop, inplace=True)
df.to_csv(os.path.join(output_file_path, Output_file_name), index=False)



