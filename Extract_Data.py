import pandas as pd
import numpy as np

df = pd.read_csv (r'Dublin_Bike_Data.csv')



df_time = df[df['TIME'] < '2020-02-25 00:00:00']

dame_street_df = df_time[df['NAME'].str.contains('DAME')]
print (dame_street_df)

pd.DataFrame.to_csv(dame_street_df,'Datasets/Dame_Street_Data.csv')

pearse_street_df = df_time[df['NAME'].str.contains('HERBERT')]
print (pearse_street_df)

pd.DataFrame.to_csv(pearse_street_df,'Datasets/Herbert_Place_Data.csv')

print ("Done")