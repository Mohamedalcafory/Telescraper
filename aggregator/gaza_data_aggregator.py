import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import faiss
from sentence_transformers import SentenceTransformer
import os
import csv
from tqdm import tqdm

# Load the Gaza geographic data
with open('aggregator/gaza_geographic_data.json', 'r', encoding='utf-8') as f:
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
os.makedirs('output', exist_ok=True)
governorate_df.to_csv('output/gaza_governorate_counts.csv', index=False)
city_town_df.to_csv('output/gaza_city_town_counts.csv', index=False)
neighborhood_df.to_csv('output/gaza_neighborhood_counts.csv', index=False)
street_df.to_csv('output/gaza_street_counts.csv', index=False)

# Print summary statistics
print(f"Total Gaza Governorates: {len(governorate_counts)}")
print(f"Total Cities/Towns: {len(city_town_counts)}")
print(f"Total Neighborhoods: {len(neighborhood_counts)}")
print(f"Total Unique Streets: {len(street_counts)}")

# Find duplicate street names across different neighborhoods/cities
duplicated_streets = [street for street, count in street_counts.items() if count > 1]
print(f"\nNumber of street names that appear in multiple locations: {len(duplicated_streets)}")

# Save a summary report
with open('output/gaza_geographic_summary.txt', 'w', encoding='utf-8') as f:
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

# ------------------------
# Vector Index Building and Message Processing
# ------------------------

print("\nBuilding vector index for geographic data...")

# Load pre-trained language model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare geographic data for vector indexing
geo_locations = []
geo_metadata = []

# Extract all locations with their hierarchical information
for governorate in gaza_data['governorates']:
    gov_name = governorate['name']
    gov_name_ar = governorate.get('name_ar', '')
    
    # Add governorate as a location
    geo_locations.append(f"{gov_name} {gov_name_ar}")
    geo_metadata.append({
        'level': 'governorate',
        'name': gov_name,
        'name_ar': gov_name_ar,
        'hierarchy': {'governorate': gov_name}
    })
    
    for city_town in governorate['cities_and_towns']:
        city_name = city_town['name']
        city_name_ar = city_town.get('name_ar', '')
        city_type = city_town.get('type', 'unknown')
        
        # Add city/town as a location
        geo_locations.append(f"{city_name} {city_name_ar}")
        geo_metadata.append({
            'level': 'city_town',
            'name': city_name,
            'name_ar': city_name_ar,
            'type': city_type,
            'hierarchy': {
                'governorate': gov_name,
                'city_town': city_name
            }
        })
        
        # Process neighborhoods if they exist
        if 'neighborhoods' in city_town:
            for neighborhood in city_town['neighborhoods']:
                neigh_name = neighborhood['name']
                neigh_name_ar = neighborhood.get('name_ar', '')
                
                # Add neighborhood as a location
                geo_locations.append(f"{neigh_name} {neigh_name_ar}")
                geo_metadata.append({
                    'level': 'neighborhood',
                    'name': neigh_name,
                    'name_ar': neigh_name_ar,
                    'hierarchy': {
                        'governorate': gov_name,
                        'city_town': city_name,
                        'neighborhood': neigh_name
                    }
                })
                
                # Process streets if they exist
                if 'streets' in neighborhood:
                    for street in neighborhood['streets']:
                        # Add street as a location
                        geo_locations.append(street)
                        geo_metadata.append({
                            'level': 'street',
                            'name': street,
                            'hierarchy': {
                                'governorate': gov_name,
                                'city_town': city_name,
                                'neighborhood': neigh_name,
                                'street': street
                            }
                        })
        
        # Process streets directly under city/town if they exist
        elif 'streets' in city_town:
            for street in city_town['streets']:
                # Add street as a location
                geo_locations.append(street)
                geo_metadata.append({
                    'level': 'street',
                    'name': street,
                    'hierarchy': {
                        'governorate': gov_name,
                        'city_town': city_name,
                        'street': street
                    }
                })

print(f"Generating embeddings for {len(geo_locations)} geographic locations...")

# Generate embeddings for all locations
geo_embeddings = model.encode(geo_locations, show_progress_bar=True)

# Normalize embeddings for cosine similarity
geo_embeddings = geo_embeddings / np.linalg.norm(geo_embeddings, axis=1, keepdims=True)

# Create Faiss index (using IndexFlatIP for inner product, which corresponds to cosine similarity with normalized vectors)
dimension = geo_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(geo_embeddings)

print("Vector index built successfully!")

# Function to find most similar geographic locations for a text
def find_geographic_match(text, top_k=3):
    # Generate embedding for the query text
    text_embedding = model.encode([text])[0]
    
    # Normalize the embedding
    text_embedding = text_embedding / np.linalg.norm(text_embedding)
    
    # Search for similar locations
    scores, indices = index.search(np.array([text_embedding]), top_k)
    
    results = []
    for i in range(top_k):
        if i < len(indices[0]):
            idx = indices[0][i]
            score = scores[0][i]
            if idx < len(geo_metadata):
                result = {
                    'location': geo_locations[idx],
                    'similarity': float(score),
                    'metadata': geo_metadata[idx]
                }
                results.append(result)
    
    return results

# Process messages and create geographic mapping
def process_messages(messages_csv_path, output_path='output/message_geo_mapping.csv'):
    print(f"Processing messages from {messages_csv_path}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Open the output file
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        fieldnames = ['message_id', 'text', 'top_match', 'similarity_score', 
                      'governorate', 'city_town', 'neighborhood', 'street']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process the messages
        with open(messages_csv_path, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader)  # Skip header
            
            # Find the indices of relevant columns
            id_idx = header.index('id') if 'id' in header else 0
            text_idx = header.index('text') if 'text' in header else 1
            
            for row in tqdm(reader, desc="Processing messages"):
                try:
                    message_id = row[id_idx]
                    message_text = row[text_idx]
                    
                    # Skip empty messages
                    if not message_text:
                        continue
                    
                    # Find geographic matches
                    matches = find_geographic_match(message_text)
                    
                    if matches:
                        top_match = matches[0]
                        hierarchy = top_match['metadata']['hierarchy']
                        
                        # Write to output file
                        writer.writerow({
                            'message_id': message_id,
                            'text': message_text[:150],  # Truncate long messages
                            'top_match': top_match['location'],
                            'similarity_score': top_match['similarity'],
                            'governorate': hierarchy.get('governorate', ''),
                            'city_town': hierarchy.get('city_town', ''),
                            'neighborhood': hierarchy.get('neighborhood', ''),
                            'street': hierarchy.get('street', '')
                        })
                except Exception as e:
                    print(f"Error processing message: {e}")
                    continue
    
    print(f"Message processing complete. Results saved to {output_path}")

# Process the messages
if os.path.exists('scraper/muthanapress84_messages.csv'):
    process_messages('scraper/muthanapress84_messages.csv', 'output/message_geo_mapping.csv')
else:
    print("Messages file not found. Please provide the path to your messages CSV file.")

print("All operations completed successfully!") 