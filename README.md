# Gaza Geographic and Damage Analysis

This project provides a comprehensive dataset and visualization tools for analyzing the geographical structure of Gaza and the damage inflicted during the conflict.

## Project Structure

- `gaza_geographic_data.json` - Detailed hierarchical data of Gaza's governorates, cities, towns, villages, neighborhoods, and streets
- `gaza_damage_data.json` - Damage metrics for Gaza locations (building destruction, casualties, displacement, infrastructure damage)
- `gaza_data_aggregator.py` - Script to analyze geographic structure and count streets by area
- `gaza_damage_visualizer.py` - Script to generate visualizations of damage statistics
- `gaza_damage_analysis.md` - Summary report of key findings from the damage analysis

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Geographic Data Analysis

To analyze the geographic structure and generate street counts:

```bash
python gaza_data_aggregator.py
```

This will output:
- CSV files with counts at different geographic levels
- A summary text file
- Console output with key statistics

### Damage Visualization

To generate visualizations of damage statistics:

```bash
python gaza_damage_visualizer.py
```

This will create:
- A `visualizations` directory containing charts and graphs
- An interactive HTML heatmap
- A markdown report summarizing key findings

## Visualizations

The damage visualizer generates multiple types of visualizations:

1. **Governorate-level Damage Overview** - Bar charts showing building destruction, displacement, casualties, and infrastructure damage by governorate
2. **City-level Damage Comparison** - Similar charts focused on major cities
3. **Interactive Damage Heatmap** - Color-coded matrix showing damage across all cities by category
4. **Camp vs. Non-Camp Comparison** - Comparison of damage metrics between refugee camps and other areas

## Data Sources

The geographic data has been compiled from various sources, with a focus on providing a comprehensive representation of Gaza's geography at multiple levels. The damage metrics are representative of the scale of destruction but should be considered illustrative.

## License

This project is provided for educational and humanitarian purposes.
