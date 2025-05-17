import json
import pandas as pd
from collections import Counter, defaultdict

# Load the Gaza geographic data
with open('gaza_geographic_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize counts
governorate_counts = Counter()
city_town_counts = Counter()
neighborhood_counts = Counter()
street_counts = Counter()

# Track streets by geographic hierarchy
streets_by_governorate = defaultdict(list)
streets_by_city = defaultdict(list)
streets_by_neighborhood = defaultdict(list)

# Process the data
gaza_data = data['gaza_strip']
for governorate in gaza_data['governorates']:
    governorate_name = governorate['name']
    governorate_counts[governorate_name] = 0
    
    for city_town in governorate['cities_and_towns']:
        city_name = city_town['name']
        city_type = city_town.get('type', 'unknown')
        city_town_counts[(city_name, city_type)] = 0
        
        # Check if the city/town has neighborhoods
        if 'neighborhoods' in city_town:
            for neighborhood in city_town['neighborhoods']:
                neighborhood_name = neighborhood['name']
                neighborhood_counts[neighborhood_name] = len(neighborhood['streets'])
                governorate_counts[governorate_name] += len(neighborhood['streets'])
                city_town_counts[(city_name, city_type)] += len(neighborhood['streets'])
                
                # Add streets to the counters
                for street in neighborhood['streets']:
                    street_counts[street] += 1
                    streets_by_governorate[governorate_name].append(street)
                    streets_by_city[city_name].append(street)
                    streets_by_neighborhood[neighborhood_name].append(street)
        
        # If the town/village doesn't have neighborhoods but has streets directly
        elif 'streets' in city_town:
            city_town_counts[(city_name, city_type)] += len(city_town['streets'])
            governorate_counts[governorate_name] += len(city_town['streets'])
            
            # Add streets to the counters
            for street in city_town['streets']:
                street_counts[street] += 1
                streets_by_governorate[governorate_name].append(street)
                streets_by_city[city_name].append(street)

# Create DataFrames for analysis
governorate_df = pd.DataFrame([
    {'Governorate': gov, 'Street Count': count}
    for gov, count in governorate_counts.items()
]).sort_values('Street Count', ascending=False)

city_town_df = pd.DataFrame([
    {'City/Town': city, 'Type': type_, 'Street Count': count}
    for (city, type_), count in city_town_counts.items()
]).sort_values('Street Count', ascending=False)

neighborhood_df = pd.DataFrame([
    {'Neighborhood': neigh, 'Street Count': count}
    for neigh, count in neighborhood_counts.items()
]).sort_values('Street Count', ascending=False)

street_df = pd.DataFrame([
    {'Street': street, 'Count': count}
    for street, count in street_counts.items()
]).sort_values('Count', ascending=False)

# Save results to CSV files
governorate_df.to_csv('gaza_governorate_counts.csv', index=False)
city_town_df.to_csv('gaza_city_town_counts.csv', index=False)
neighborhood_df.to_csv('gaza_neighborhood_counts.csv', index=False)
street_df.to_csv('gaza_street_counts.csv', index=False)

# Print summary statistics
print(f"Total Gaza Governorates: {len(governorate_counts)}")
print(f"Total Cities/Towns: {len(city_town_counts)}")
print(f"Total Neighborhoods: {len(neighborhood_counts)}")
print(f"Total Unique Streets: {len(street_counts)}")
print("\nTop 5 Governorates by Street Count:")
print(governorate_df.head(5))
print("\nTop 5 Cities/Towns by Street Count:")
print(city_town_df.head(5))
print("\nTop 5 Neighborhoods by Street Count:")
print(neighborhood_df.head(5))
print("\nMost Common Street Names (may appear in multiple locations):")
print(street_df.head(10))

# Find duplicate street names across different neighborhoods/cities
duplicated_streets = [street for street, count in street_counts.items() if count > 1]
print(f"\nNumber of street names that appear in multiple locations: {len(duplicated_streets)}")

# Save a summary report
with open('gaza_geographic_summary.txt', 'w', encoding='utf-8') as f:
    f.write("Gaza Geographic Data Summary\n")
    f.write("===========================\n\n")
    f.write(f"Total Gaza Governorates: {len(governorate_counts)}\n")
    f.write(f"Total Cities/Towns: {len(city_town_counts)}\n")
    f.write(f"Total Neighborhoods: {len(neighborhood_counts)}\n")
    f.write(f"Total Unique Streets: {len(street_counts)}\n\n")
    
    f.write("Streets by Governorate:\n")
    for gov, count in governorate_counts.most_common():
        f.write(f"  {gov}: {count} streets\n")
    
    f.write("\nStreets by City/Town (Top 10):\n")
    for (city, type_), count in sorted(city_town_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        f.write(f"  {city} ({type_}): {count} streets\n")
    
    f.write("\nStreets by Neighborhood (Top 10):\n")
    for neigh, count in sorted(neighborhood_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        f.write(f"  {neigh}: {count} streets\n")

print("\nAnalysis complete. CSV files and summary report have been generated.") 