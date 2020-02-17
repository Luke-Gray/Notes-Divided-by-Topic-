#kernal density plot removing the axis spines and set background style
sns.set_style('white')
sns.kdeplot(titanic['Age'], shade = True)
plt.xlabel('Age')
sns.despine(left = True, bottom =True)

# sns Facet Grid using row and hue conditions
g = sns.FacetGrid(titanic, col = 'Survived', row = 'Pclass',
                  hue = 'Sex', size = 3)
g.map(sns.kdeplot, 'Age', shade = True)
g.add_legend()
sns.despine(left = True, bottom = True)
plt.show()

#using basemap to create 2d scatter of map
from mpl_toolkits.basemap import Basemap

fig, ax = plt.subplots(figsize = (15,20))
plt.title("Scaled Up Earth With Coastlines")
m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180)
longitudes = airports["longitude"].tolist()
latitudes = airports["latitude"].tolist()
x, y = m(longitudes, latitudes)
m.scatter(x, y, s=1)
m.drawcoastlines()
plt.show()

##complex mapping using draw_great_circles. requires a starting long-lat and an ending long-lat
## in this case, we are using airport(starting) and their destinations(ending)
fig, ax = plt.subplots(figsize=(15,20))
m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180)
m.drawcoastlines()

def create_great_circles(df):
    for index, row in df.iterrows():
        end_lat, start_lat = row['end_lat'], row['start_lat']
        end_lon, start_lon = row['end_lon'], row['start_lon']
        
        if abs(end_lat - start_lat) < 180:
            if abs(end_lon - start_lon) < 180:
                m.drawgreatcircle(start_lon, start_lat, end_lon, end_lat)

dfw = geo_routes[geo_routes['source'] == "DFW"]
create_great_circles(dfw)
plt.show()

#setting color before matplotlib
color  = list(map(lambda x: 'b' if x==0 else 'r', y_train))	

## basemap with river and specific colors
from mpl_toolkits.basemap import Basemap
m = Basemap(
    projection='merc', 
    llcrnrlat=40.496044, 
    urcrnrlat=40.915256, 
    llcrnrlon=-74.255735, 
    urcrnrlon=-73.700272,
    resolution='i'
)

m.drawmapboundary(fill_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawrivers(color='#6D5F47', linewidth=.4)

longitudes = combined['lon'].tolist()
latitudes = combined['lat'].tolist()

m.scatter(longitudes, latitudes, s = 20, zorder = 2,
          latlon=True, c = combined['ell_percent'], cmap='summer')
plt.show()

#Plotting mean, std of a distribution
import matplotlib.pyplot as plt
houses['SalePrice'].plot.kde(xlim = (houses['SalePrice'].min(),
                                    houses['SalePrice'].max()
                                    )
                            )

st_dev = houses['SalePrice'].std(ddof = 0)
mean = houses['SalePrice'].mean()
plt.axvline(mean, color = 'Black', label = 'Mean')
plt.axvline(mean + st_dev, color = 'Red', label = 'Standard deviation')
plt.axvline(220000, color = 'Orange', label = '220000')
plt.legend()