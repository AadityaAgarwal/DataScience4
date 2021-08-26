import csv 
import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
rows = []

with open("final_data.csv", "r") as f:
  csvreader = csv.reader(f)
  for row in csvreader: 
    rows.append(row)
df=pd.read_csv("final_data.csv")
headers = rows[0]
star_data_rows = rows[1:]
print(headers)
print(star_data_rows[9])

df['Radius']=df['Radius'].apply(lambda x:x.replace('$','').replace(',','')).astype('float')

star_gravity=[]
star_mass=df['Mass'].to_list()
star_radius=df['Radius'].to_list()
star_name=df['Star_name'].to_list()
star_distance=df['Distance'].to_list()

def convert_to_si(radius,mass):
  for i in range(0,len(radius)-1):
    radius[i]=radius[i]*6.957e+8
    mass[i]=mass[i]*1.989e+30
convert_to_si(star_radius,star_mass)

def gravity_calc(radius,mass):
  G=6.674e-11
  for index in range(0,len(mass)):
    g=(mass[index]*G)/((radius[index])**2)
    star_gravity.append(g)
gravity_calc(star_radius,star_mass)
df['Gravity']=star_gravity

print(star_gravity)

  
fig = px.scatter(x=star_radius, y=star_mass, size=star_gravity, hover_data=[star_name])
fig.show()

fig2=px.scatter(x=star_mass,y=star_gravity).show()


X=[]

for index,star_masses in enumerate(star_mass):
  temp_list=[
             star_radius[index],
             star_masses
  ]

  X.append(temp_list)

wcss=[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state = 42)
    kmeans.fit(X)
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), wcss, marker='o', color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

temp_star_dist=[]

for index,planet in enumerate(star_distance):
    print(index,planet)
    star_distance.remove(planet)
    if (float(planet)) <=100:
        temp_star_dist.append(planet)
    else:
        temp_star_dist.remove(planet)
print(temp_star_dist)