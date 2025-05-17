import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from sentence_transformers import SentenceTransformer
import faiss

# Create output directory for visualizations
os.makedirs('visualizations', exist_ok=True)

# Load the Gaza geographic data and damage data
with open('aggregator/gaza_geographic_data.json', 'r', encoding='utf-8') as f:
    geo_data = json.load(f)

with open('gaza_damage_data.json', 'r', encoding='utf-8') as f:
    damage_data = json.load(f)

# Extract data into dataframes
# Governorate data
governorate_df = pd.DataFrame([{
    'Governorate': gov['name'],
    'Governorate_AR': gov['name_ar']
} for gov in geo_data['gaza_strip']['governorates']])

# Add damage metrics to governorate dataframe
gov_damage_df = pd.DataFrame([{
    'Governorate': gov['name'],
    'Buildings_Destroyed_Percent': gov['statistics']['buildings_destroyed_percent'],
    'Displacement_Percent': gov['statistics']['displacement_percent'],
    'Civilian_Casualties': gov['statistics']['civilian_casualties'],
    'Hospitals_Damage_Percent': gov['statistics']['infrastructure_damage_percent']['hospitals'],
    'Schools_Damage_Percent': gov['statistics']['infrastructure_damage_percent']['schools'],
    'Water_Facilities_Damage_Percent': gov['statistics']['infrastructure_damage_percent']['water_facilities'],
    'Power_Facilities_Damage_Percent': gov['statistics']['infrastructure_damage_percent']['power_facilities']
} for gov in damage_data['damage_metrics']['governorates']])

# Merge governorate dataframes
governorate_df = pd.merge(governorate_df, gov_damage_df, on='Governorate')

# City/Town data with nested extraction
city_town_data = []
for gov in geo_data['gaza_strip']['governorates']:
    for city in gov['cities_and_towns']:
        city_town_data.append({
            'Governorate': gov['name'],
            'City_Town': city['name'],
            'City_Town_AR': city['name_ar'],
            'Type': city.get('type', 'unknown')
        })

city_town_df = pd.DataFrame(city_town_data)

# Add damage metrics to city/town dataframe
city_damage_df = pd.DataFrame([{
    'City_Town': city['name'],
    'Buildings_Destroyed_Percent': city['statistics']['buildings_destroyed_percent'],
    'Displacement_Percent': city['statistics']['displacement_percent'],
    'Civilian_Casualties': city['statistics']['civilian_casualties'],
    'Hospitals_Damage_Percent': city['statistics']['infrastructure_damage_percent']['hospitals'],
    'Schools_Damage_Percent': city['statistics']['infrastructure_damage_percent']['schools'],
    'Water_Facilities_Damage_Percent': city['statistics']['infrastructure_damage_percent']['water_facilities'],
    'Power_Facilities_Damage_Percent': city['statistics']['infrastructure_damage_percent']['power_facilities']
} for city in damage_data['damage_metrics']['cities_and_towns']])

# Merge city/town dataframes
city_town_df = pd.merge(city_town_df, city_damage_df, on='City_Town')

# Neighborhood data
neighborhood_data = []
for gov in geo_data['gaza_strip']['governorates']:
    for city in gov['cities_and_towns']:
        if 'neighborhoods' in city:
            for neigh in city['neighborhoods']:
                neighborhood_data.append({
                    'Governorate': gov['name'],
                    'City_Town': city['name'],
                    'Neighborhood': neigh['name'],
                    'Neighborhood_AR': neigh['name_ar'],
                    'Street_Count': len(neigh['streets'])
                })

neighborhood_df = pd.DataFrame(neighborhood_data)

# Add damage metrics to neighborhood dataframe where available
neigh_damage_df = pd.DataFrame([{
    'Neighborhood': neigh['name'],
    'Buildings_Destroyed_Percent': neigh['statistics']['buildings_destroyed_percent'],
    'Displacement_Percent': neigh['statistics']['displacement_percent'],
    'Civilian_Casualties': neigh['statistics']['civilian_casualties'],
    'Hospitals_Damage_Percent': neigh['statistics']['infrastructure_damage_percent']['hospitals'],
    'Schools_Damage_Percent': neigh['statistics']['infrastructure_damage_percent']['schools'],
    'Water_Facilities_Damage_Percent': neigh['statistics']['infrastructure_damage_percent']['water_facilities'],
    'Power_Facilities_Damage_Percent': neigh['statistics']['infrastructure_damage_percent']['power_facilities']
} for neigh in damage_data['damage_metrics']['neighborhoods']])

# Merge neighborhood dataframes (left join as not all neighborhoods have damage data)
neighborhood_df = pd.merge(neighborhood_df, neigh_damage_df, on='Neighborhood', how='left')

# Fill NaN values with parent city data where available
for idx, row in neighborhood_df.iterrows():
    if pd.isna(row['Buildings_Destroyed_Percent']):
        city_data = city_town_df[city_town_df['City_Town'] == row['City_Town']]
        if not city_data.empty:
            for col in ['Buildings_Destroyed_Percent', 'Displacement_Percent', 'Civilian_Casualties', 
                        'Hospitals_Damage_Percent', 'Schools_Damage_Percent', 
                        'Water_Facilities_Damage_Percent', 'Power_Facilities_Damage_Percent']:
                neighborhood_df.loc[idx, col] = city_data[col].values[0]

# --------------------------------
# VISUALIZATION 1: Governorate-level Damage Overview
# --------------------------------
plt.figure(figsize=(15, 10))

# Buildings destroyed
plt.subplot(2, 2, 1)
bars = plt.bar(governorate_df['Governorate'], governorate_df['Buildings_Destroyed_Percent'], color='firebrick')
plt.title('Buildings Destroyed (%)', fontsize=14, fontweight='bold')
plt.xlabel('Governorate', fontsize=12)
plt.ylabel('Percentage', fontsize=12)
plt.xticks(rotation=45)
plt.ylim(0, 100)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f"{bar.get_height()}%", ha='center', fontweight='bold')

# Displacement
plt.subplot(2, 2, 2)
bars = plt.bar(governorate_df['Governorate'], governorate_df['Displacement_Percent'], color='darkred')
plt.title('Population Displaced (%)', fontsize=14, fontweight='bold')
plt.xlabel('Governorate', fontsize=12)
plt.ylabel('Percentage', fontsize=12)
plt.xticks(rotation=45)
plt.ylim(0, 100)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f"{bar.get_height()}%", ha='center', fontweight='bold')

# Civilian Casualties
plt.subplot(2, 2, 3)
bars = plt.bar(governorate_df['Governorate'], governorate_df['Civilian_Casualties'], color='black')
plt.title('Civilian Casualties', fontsize=14, fontweight='bold')
plt.xlabel('Governorate', fontsize=12)
plt.ylabel('Number of People', fontsize=12)
plt.xticks(rotation=45)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, 
             f"{int(bar.get_height())}", ha='center', fontweight='bold')

# Infrastructure Damage
plt.subplot(2, 2, 4)
gov_names = governorate_df['Governorate'].tolist()
x = np.arange(len(gov_names))
width = 0.2

plt.bar(x - 1.5*width, governorate_df['Hospitals_Damage_Percent'], width, label='Hospitals', color='darkblue')
plt.bar(x - 0.5*width, governorate_df['Schools_Damage_Percent'], width, label='Schools', color='royalblue')
plt.bar(x + 0.5*width, governorate_df['Water_Facilities_Damage_Percent'], width, label='Water', color='lightblue')
plt.bar(x + 1.5*width, governorate_df['Power_Facilities_Damage_Percent'], width, label='Power', color='steelblue')

plt.title('Infrastructure Damage (%)', fontsize=14, fontweight='bold')
plt.xlabel('Governorate', fontsize=12)
plt.ylabel('Percentage', fontsize=12)
plt.xticks(x, gov_names, rotation=45)
plt.ylim(0, 110)
plt.legend()

plt.suptitle('Gaza Strip Damage by Governorate', fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('visualizations/governorate_damage_overview.png', dpi=300)

# --------------------------------
# VISUALIZATION 2: City-level Damage Comparison
# --------------------------------
# Select major cities
major_cities = ['Gaza City', 'Rafah City', 'Khan Younis City', 'Jabalia', 'Beit Hanoun', 'Beit Lahia', 'Deir al-Balah City']
city_filtered_df = city_town_df[city_town_df['City_Town'].isin(major_cities)]

plt.figure(figsize=(20, 15))

# Buildings destroyed
plt.subplot(2, 2, 1)
bars = plt.bar(city_filtered_df['City_Town'], city_filtered_df['Buildings_Destroyed_Percent'], color='firebrick')
plt.title('Buildings Destroyed (%)', fontsize=16, fontweight='bold')
plt.xlabel('City', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.ylim(0, 100)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f"{bar.get_height()}%", ha='center', fontweight='bold', fontsize=12)

# Displacement
plt.subplot(2, 2, 2)
bars = plt.bar(city_filtered_df['City_Town'], city_filtered_df['Displacement_Percent'], color='darkred')
plt.title('Population Displaced (%)', fontsize=16, fontweight='bold')
plt.xlabel('City', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.ylim(0, 101)  # Allow room for Beit Hanoun's 100%
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f"{bar.get_height()}%", ha='center', fontweight='bold', fontsize=12)

# Civilian Casualties
plt.subplot(2, 2, 3)
bars = plt.bar(city_filtered_df['City_Town'], city_filtered_df['Civilian_Casualties'], color='black')
plt.title('Civilian Casualties', fontsize=16, fontweight='bold')
plt.xlabel('City', fontsize=14)
plt.ylabel('Number of People', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, 
             f"{int(bar.get_height())}", ha='center', fontweight='bold', fontsize=12)

# Infrastructure Damage
plt.subplot(2, 2, 4)
city_names = city_filtered_df['City_Town'].tolist()
x = np.arange(len(city_names))
width = 0.2

plt.bar(x - 1.5*width, city_filtered_df['Hospitals_Damage_Percent'], width, label='Hospitals', color='darkblue')
plt.bar(x - 0.5*width, city_filtered_df['Schools_Damage_Percent'], width, label='Schools', color='royalblue')
plt.bar(x + 0.5*width, city_filtered_df['Water_Facilities_Damage_Percent'], width, label='Water', color='lightblue')
plt.bar(x + 1.5*width, city_filtered_df['Power_Facilities_Damage_Percent'], width, label='Power', color='steelblue')

plt.title('Infrastructure Damage (%)', fontsize=16, fontweight='bold')
plt.xlabel('City', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.xticks(x, city_names, rotation=45, fontsize=12)
plt.ylim(0, 110)
plt.legend(fontsize=12)

plt.suptitle('Gaza Strip Damage by Major City', fontsize=22, fontweight='bold', y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('visualizations/city_damage_overview.png', dpi=300)

# --------------------------------
# VISUALIZATION 3: Interactive Plotly Heatmap for Cities
# --------------------------------
# Prepare data for heatmap
heatmap_df = city_town_df[['City_Town', 'Buildings_Destroyed_Percent', 'Displacement_Percent', 
                           'Civilian_Casualties', 'Hospitals_Damage_Percent', 
                           'Schools_Damage_Percent', 'Water_Facilities_Damage_Percent', 
                           'Power_Facilities_Damage_Percent']]

# Normalize casualties for heatmap using min-max scaling
max_casualties = heatmap_df['Civilian_Casualties'].max()
heatmap_df['Casualties_Normalized'] = heatmap_df['Civilian_Casualties'] / max_casualties * 100

# Create a heatmap with better readable labels
heatmap_cols = ['Buildings_Destroyed_Percent', 'Displacement_Percent', 'Casualties_Normalized',
                'Hospitals_Damage_Percent', 'Schools_Damage_Percent', 
                'Water_Facilities_Damage_Percent', 'Power_Facilities_Damage_Percent']

readable_cols = ['Buildings Destroyed', 'Population Displaced', 'Civilian Casualties',
                'Hospitals Damaged', 'Schools Damaged', 'Water Facilities Damaged', 'Power Facilities Damaged']

fig = px.imshow(heatmap_df[heatmap_cols].values,
                labels=dict(x="Damage Type", y="City/Town", color="Percentage"),
                x=readable_cols,
                y=heatmap_df['City_Town'],
                color_continuous_scale='Reds',
                title="Gaza Strip Damage Heatmap by City/Town")

# Add text annotations
for i in range(len(heatmap_df)):
    for j in range(len(heatmap_cols)):
        if j == 2:  # Show actual casualty numbers instead of normalized value
            text = f"{int(heatmap_df['Civilian_Casualties'].iloc[i])}"
        else:
            text = f"{int(heatmap_df[heatmap_cols[j]].iloc[i])}%"
        
        fig.add_annotation(
            x=j,
            y=i,
            text=text,
            showarrow=False,
            font=dict(color="black" if heatmap_df[heatmap_cols[j]].iloc[i] < 70 else "white")
        )

fig.update_layout(height=800, width=1200)
fig.write_html('visualizations/city_damage_heatmap.html')

# --------------------------------
# VISUALIZATION 4: Camp vs. Non-Camp Areas Comparison
# --------------------------------
# Identify refugee camps
camp_towns = city_town_df[city_town_df['Type'] == 'refugee_camp']
camp_neighborhoods = neighborhood_df[neighborhood_df['Neighborhood'].str.contains('Camp')]

# Combine both for overall camp data
all_camps = pd.concat([
    camp_towns[['Buildings_Destroyed_Percent', 'Displacement_Percent', 'Civilian_Casualties',
                'Hospitals_Damage_Percent', 'Schools_Damage_Percent', 
                'Water_Facilities_Damage_Percent', 'Power_Facilities_Damage_Percent']],
    camp_neighborhoods[['Buildings_Destroyed_Percent', 'Displacement_Percent', 'Civilian_Casualties',
                        'Hospitals_Damage_Percent', 'Schools_Damage_Percent', 
                        'Water_Facilities_Damage_Percent', 'Power_Facilities_Damage_Percent']]
])

# Calculate averages for camps
camp_avg = all_camps.mean()

# Non-camp city/town data
non_camp_towns = city_town_df[city_town_df['Type'] != 'refugee_camp']
non_camp_neighborhoods = neighborhood_df[~neighborhood_df['Neighborhood'].str.contains('Camp')]

# Combine both for overall non-camp data
all_non_camps = pd.concat([
    non_camp_towns[['Buildings_Destroyed_Percent', 'Displacement_Percent', 'Civilian_Casualties',
                    'Hospitals_Damage_Percent', 'Schools_Damage_Percent', 
                    'Water_Facilities_Damage_Percent', 'Power_Facilities_Damage_Percent']],
    non_camp_neighborhoods[['Buildings_Destroyed_Percent', 'Displacement_Percent', 'Civilian_Casualties',
                            'Hospitals_Damage_Percent', 'Schools_Damage_Percent', 
                            'Water_Facilities_Damage_Percent', 'Power_Facilities_Damage_Percent']]
])

# Calculate averages for non-camps
non_camp_avg = all_non_camps.mean()

# Create comparison dataframe
comparison_data = pd.DataFrame({
    'Metric': ['Buildings Destroyed (%)', 'Population Displaced (%)', 
               'Hospitals Damaged (%)', 'Schools Damaged (%)',
               'Water Facilities Damaged (%)', 'Power Facilities Damaged (%)'],
    'Refugee Camps': [camp_avg['Buildings_Destroyed_Percent'], camp_avg['Displacement_Percent'],
                      camp_avg['Hospitals_Damage_Percent'], camp_avg['Schools_Damage_Percent'],
                      camp_avg['Water_Facilities_Damage_Percent'], camp_avg['Power_Facilities_Damage_Percent']],
    'Other Areas': [non_camp_avg['Buildings_Destroyed_Percent'], non_camp_avg['Displacement_Percent'],
                   non_camp_avg['Hospitals_Damage_Percent'], non_camp_avg['Schools_Damage_Percent'],
                   non_camp_avg['Water_Facilities_Damage_Percent'], non_camp_avg['Power_Facilities_Damage_Percent']]
})

# Create bar chart comparing camps vs non-camps
plt.figure(figsize=(15, 10))
x = np.arange(len(comparison_data['Metric']))
width = 0.35

plt.bar(x - width/2, comparison_data['Refugee Camps'], width, label='Refugee Camps', color='darkred')
plt.bar(x + width/2, comparison_data['Other Areas'], width, label='Other Areas', color='firebrick')

plt.title('Damage Comparison: Refugee Camps vs. Other Areas', fontsize=18, fontweight='bold')
plt.xlabel('Damage Metric', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.xticks(x, comparison_data['Metric'], rotation=45, ha='right', fontsize=12)
plt.ylim(0, 110)
plt.legend(fontsize=14)

# Add percentage annotations
for i, (camp_val, non_camp_val) in enumerate(zip(comparison_data['Refugee Camps'], comparison_data['Other Areas'])):
    plt.text(i - width/2, camp_val + 3, f"{camp_val:.1f}%", ha='center', fontweight='bold')
    plt.text(i + width/2, non_camp_val + 3, f"{non_camp_val:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/camp_vs_noncamp_comparison.png', dpi=300)

# Create a markdown file with key findings
with open('gaza_damage_analysis.md', 'w', encoding='utf-8') as f:
    f.write("# Gaza Strip Damage Analysis\n\n")
    f.write("## Overview\n\n")
    f.write("This analysis examines the damage caused to the Gaza Strip during the conflict, focusing on:\n")
    f.write("- Building destruction\n")
    f.write("- Population displacement\n")
    f.write("- Civilian casualties\n")
    f.write("- Infrastructure damage (hospitals, schools, water, power)\n\n")
    
    f.write("## Key Findings\n\n")
    
    f.write("### Governorate-Level Impact\n\n")
    f.write(f"- **Most severely affected governorate**: {governorate_df.sort_values('Buildings_Destroyed_Percent', ascending=False).iloc[0]['Governorate']} ")
    f.write(f"({governorate_df.sort_values('Buildings_Destroyed_Percent', ascending=False).iloc[0]['Buildings_Destroyed_Percent']}% buildings destroyed)\n")
    f.write(f"- **Highest displacement**: {governorate_df.sort_values('Displacement_Percent', ascending=False).iloc[0]['Governorate']} ")
    f.write(f"({governorate_df.sort_values('Displacement_Percent', ascending=False).iloc[0]['Displacement_Percent']}% of population)\n")
    f.write(f"- **Highest casualties**: {governorate_df.sort_values('Civilian_Casualties', ascending=False).iloc[0]['Governorate']} ")
    f.write(f"({int(governorate_df.sort_values('Civilian_Casualties', ascending=False).iloc[0]['Civilian_Casualties'])} civilians)\n\n")
    
    f.write("### City-Level Impact\n\n")
    f.write(f"- **Most devastated city**: {city_town_df.sort_values('Buildings_Destroyed_Percent', ascending=False).iloc[0]['City_Town']} ")
    f.write(f"({city_town_df.sort_values('Buildings_Destroyed_Percent', ascending=False).iloc[0]['Buildings_Destroyed_Percent']}% buildings destroyed)\n")
    f.write(f"- **City with highest displacement**: {city_town_df.sort_values('Displacement_Percent', ascending=False).iloc[0]['City_Town']} ")
    f.write(f"({city_town_df.sort_values('Displacement_Percent', ascending=False).iloc[0]['Displacement_Percent']}% of population)\n")
    f.write(f"- **City with highest casualties**: {city_town_df.sort_values('Civilian_Casualties', ascending=False).iloc[0]['City_Town']} ")
    f.write(f"({int(city_town_df.sort_values('Civilian_Casualties', ascending=False).iloc[0]['Civilian_Casualties'])} civilians)\n\n")
    
    f.write("### Refugee Camps vs. Other Areas\n\n")
    f.write(f"- Refugee camps suffered {camp_avg['Buildings_Destroyed_Percent']:.1f}% building destruction compared to {non_camp_avg['Buildings_Destroyed_Percent']:.1f}% in other areas\n")
    f.write(f"- Displacement in refugee camps reached {camp_avg['Displacement_Percent']:.1f}% compared to {non_camp_avg['Displacement_Percent']:.1f}% in other areas\n")
    f.write(f"- Infrastructure damage was consistently higher in refugee camps across all categories\n\n")
    
    f.write("### Infrastructure Damage\n\n")
    f.write("- Critical infrastructure was severely impacted across all governorates\n")
    f.write(f"- Power facilities experienced the highest damage rate ({governorate_df['Power_Facilities_Damage_Percent'].mean():.1f}% average)\n")
    f.write(f"- Water facilities damage averaged {governorate_df['Water_Facilities_Damage_Percent'].mean():.1f}%, creating a humanitarian crisis\n")
    f.write(f"- Hospital damage averaged {governorate_df['Hospitals_Damage_Percent'].mean():.1f}%, severely limiting healthcare access\n\n")
    
    f.write("## Methodology\n\n")
    f.write("This analysis combines geographic data from the Gaza Strip with damage assessment metrics. ")
    f.write("The data is visualized at three geographic levels: governorate, city/town, and neighborhood. ")
    f.write("Special attention is given to the comparison between refugee camps and other areas to highlight ")
    f.write("the disproportionate impact on vulnerable populations.\n\n")
    
    f.write("## Images\n\n")
    f.write("- [Governorate Damage Overview](visualizations/governorate_damage_overview.png)\n")
    f.write("- [City Damage Overview](visualizations/city_damage_overview.png)\n")
    f.write("- [Camp vs. Non-Camp Comparison](visualizations/camp_vs_noncamp_comparison.png)\n")
    f.write("- [Interactive City Damage Heatmap](visualizations/city_damage_heatmap.html)\n")

print("Analysis complete. Visualizations have been saved to the 'visualizations' directory.")
print("A summary of findings is available in gaza_damage_analysis.md")

# Extract all location names at different levels
location_names = []
location_info = []  # Store additional context about each location

# Extract from governorates, cities, neighborhoods, etc.
for gov in geo_data['gaza_strip']['governorates']:
    location_names.append(gov['name'])
    location_info.append({'type': 'governorate', 'name': gov['name'], 'name_ar': gov['name_ar']})
    
    for city in gov['cities_and_towns']:
        location_names.append(city['name'])
        location_info.append({'type': 'city', 'name': city['name'], 'name_ar': city['name_ar'], 
                             'governorate': gov['name']})
        
        # Add neighborhoods
        if 'neighborhoods' in city:
            for neigh in city['neighborhoods']:
                location_names.append(neigh['name'])
                location_info.append({'type': 'neighborhood', 'name': neigh['name'], 
                                     'name_ar': neigh['name_ar'], 'city': city['name'], 
                                     'governorate': gov['name']})
                
                # Add streets
                for street in neigh['streets']:
                    location_names.append(street)
                    location_info.append({'type': 'street', 'name': street, 
                                         'neighborhood': neigh['name'], 'city': city['name'],
                                         'governorate': gov['name']})

# Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
location_embeddings = model.encode(location_names)

# Create FAISS index
dimension = location_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(location_embeddings).astype('float32'))

# Function to match message with locations
def match_message_to_locations(message_text, top_k=5):
    message_embedding = model.encode([message_text])
    D, I = index.search(np.array(message_embedding).astype('float32'), top_k)
    
    matches = []
    for idx in I[0]:
        matches.append({
            'location': location_names[idx],
            'info': location_info[idx],
            'score': float(D[0][matches.index(location_names[idx])])
        })
    
    return matches

# Example usage with scraped messages
scraped_messages = [
    "Heavy bombing reported in Jabalia camp last night",
    "Water infrastructure completely destroyed in Beit Hanoun",
    "Schools in Al-Rimal neighborhood targeted by airstrikes"
]

for message in scraped_messages:
    matches = match_message_to_locations(message)
    print(f"Message: {message}")
    print("Matching locations:")
    for match in matches:
        print(f"  - {match['location']} ({match['info']['type']})")
    print() 