import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

def generate_house_price_dataset(n_samples=1000):
    """
    Generate a synthetic house price dataset for machine learning training.
    
    Features:
    - size_sqft: Size of house in square feet (500-5000)
    - bedrooms: Number of bedrooms (1-6)
    - floors: Number of floors (1-3)
    - age_years: Age of house in years (0-50)
    
    Target:
    - price_inr: Price in Indian Rupees (2,500,000 - 100,000,000)
    
    Returns:
    - X: numpy array of shape (n_samples, 4) with features
    - y: numpy array of shape (n_samples,) with target prices
    - feature_names: list of feature names
    """
    
    # Generate input features
    size_sqft = np.random.uniform(500, 5000, n_samples)
    bedrooms = np.random.randint(1, 7, n_samples)  # 1-6 bedrooms
    floors = np.random.randint(1, 4, n_samples)    # 1-3 floors
    age_years = np.random.uniform(0, 50, n_samples)
    
    # Generate realistic price based on features with some noise
    # Base price calculation with realistic coefficients
    base_price = (
        size_sqft * 15000 +           # ₹15,000 per sq ft base rate
        bedrooms * 500000 +           # ₹5 lakh per bedroom
        floors * 800000 +             # ₹8 lakh per floor
        (50 - age_years) * 50000      # Depreciation: ₹50k per year of age
    )
    
    # Add some random noise (±20% variation)
    noise = np.random.normal(1.0, 0.2, n_samples)
    price_inr = base_price * noise
    
    # Ensure prices are within the specified range
    price_inr = np.clip(price_inr, 2500000, 100000000)
    
    # Create feature matrix X and target vector y
    X = np.column_stack([
        size_sqft.round(0).astype(int),
        bedrooms,
        floors,
        age_years.round(1)
    ])
    
    y = price_inr.round(0).astype(int)
    
    feature_names = ['size_sqft', 'bedrooms', 'floors', 'age_years']
    
    return X, y, feature_names

def display_dataset_info(X, y, feature_names):
    """Display basic information about the dataset."""
    print("House Price Dataset Information")
    print("=" * 40)
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"\nFeature ranges:")
    for i, name in enumerate(feature_names):
        print(f"{name}: {X[:, i].min()} - {X[:, i].max()}")
    print(f"Price (INR): ₹{y.min():,} - ₹{y.max():,}")
    
    print(f"\nFirst 5 samples:")
    print("Features (size_sqft, bedrooms, floors, age_years):")
    print(X[:5])
    print("Prices (INR):")
    print(y[:5])
    
    print(f"\nDataset statistics:")
    print("Feature means:", X.mean(axis=0))
    print("Feature std:", X.std(axis=0))
    print(f"Price mean: ₹{y.mean():,.0f}")
    print(f"Price std: ₹{y.std():,.0f}")

def plot_price_distribution(X, y, feature_names):
    """Plot the distribution of house prices."""
    plt.figure(figsize=(12, 8))
    
    # Price distribution
    plt.subplot(3, 2, 1)
    plt.hist(y / 1000000, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Price (Crores INR)')
    plt.ylabel('Frequency')
    plt.title('Distribution of House Prices')
    plt.grid(True, alpha=0.3)
    
    # Price vs Size
    plt.subplot(3, 2, 2)
    plt.scatter(X[:, 0], y / 1000000, alpha=0.6, color='green')
    plt.xlabel('Size (sq ft)')
    plt.ylabel('Price (Crores INR)')
    plt.title('Price vs House Size')
    plt.grid(True, alpha=0.3)
    
    # Price vs Age
    plt.subplot(3, 2, 3)
    plt.scatter(X[:, 3], y / 1000000, alpha=0.6, color='orange')
    plt.xlabel('Age (years)')
    plt.ylabel('Price (Crores INR)')
    plt.title('Price vs House Age')
    plt.grid(True, alpha=0.3)
    
    # Price vs Bedrooms (box plot)
    plt.subplot(3, 2, 4)
    bedroom_values = np.unique(X[:, 1])
    bedroom_groups = [y[X[:, 1] == bedroom] / 1000000 for bedroom in bedroom_values]
    plt.boxplot(bedroom_groups, labels=bedroom_values.astype(int))
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Price (Crores INR)')
    plt.title('Price Distribution by Bedrooms')
    plt.grid(True, alpha=0.3)
    
    # price vs floors
    plt.subplot(3, 2, 5)
    floor_values = np.unique(X[:, 2])
    floor_groups = [y[X[:, 2] == floor] / 1000000 for floor in floor_values]
    plt.boxplot(floor_groups, labels=floor_values.astype(int))
    plt.xlabel('Number of Floors')
    plt.ylabel('Price (Crores INR)')
    
    plt.tight_layout()
    plt.show()

def save_dataset_to_csv(X, y, feature_names, filename='house_price_dataset.csv'):
    """Save the dataset to a CSV file."""
    # Combine features and target
    data = np.column_stack([X, y])
    header = ','.join(feature_names + ['price_inr'])
    
    np.savetxt(filename, data, delimiter=',', header=header, comments='', fmt='%d')
    print(f"\nDataset saved to {filename}")

if __name__ == "__main__":
    # Generate the dataset
    print("Generating house price dataset...")
    X, y, feature_names = generate_house_price_dataset(n_samples=1000)
    
    # Display dataset information
    display_dataset_info(X, y, feature_names)
    
    # Plot visualizations
    plot_price_distribution(X, y, feature_names)
    
    # Save to CSV
    # save_dataset_to_csv(X, y, feature_names)
    
    print("\nDataset generation complete!")
    print("You can now use this dataset for machine learning training.")
    print("\nTo access the data programmatically:")
    print("  X, y, feature_names = generate_house_price_dataset()")
    print("  # X is the feature matrix (n_samples, 4)")
    print("  # y is the target vector (n_samples,)")
    print("  # feature_names = ['size_sqft', 'bedrooms', 'floors', 'age_years']")
