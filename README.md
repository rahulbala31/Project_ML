    import pandas as pd
    import numpy as np
    from datetime import datetime
    from math import radians, cos, sin, sqrt, atan2
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from colorama import Fore, Style
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler  # For normalization
    
    # Haversine function to calculate distance between two coordinates
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0  # Earth's radius in kilometers
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c
    
    # Load dataset
    try:
        data = pd.read_excel('ev_data_with_random_valuess.xlsx', engine='openpyxl')
    except FileNotFoundError:
        print(Fore.RED + "Error: File not found. Please check the file path.")
        exit()
    
    # Data Preprocessing
    data = data.drop(['Station ID', 'STATION name', 'Unnamed: 2', 'Unnamed: 11'], axis=1, errors='ignore')
    
    # Function to extract numeric values from strings
    def extract_numeric(value):
        try:
            return float(str(value).split()[0])
        except ValueError:
            return np.nan  # Return NaN instead of None for consistency
    
    data['capacity'] = data['capacity'].apply(extract_numeric)
    
    # Function to convert time string to float hours
    def convert_time_to_hour(time_str):
        try:
            hours, minutes = map(int, time_str.split(':'))
            return hours + minutes / 60.0
        except:
            return np.nan  # Return NaN instead of None for consistency
    
    # Function to convert cost strings to float
    def convert_cost_to_float(cost_str):
        try:
            return float(cost_str.split()[0].replace('₹', '').strip())
        except:
            return np.nan  # Return NaN instead of None for consistency
    
    data['cost_per_unit'] = data['cost_per_unit'].apply(convert_cost_to_float)
    data = data.dropna()  # Drop rows with missing values
    
    # Feature Selection
    X = data[['latitude', 'longitude', 'capacity', 'current_battery_level', 'charge_required']]
    y = data['cost_per_unit']
    
    # Ensure correct data types
    X = X.astype(float)
    y = y.astype(float)
    
    # Split data into training and testing sets before normalization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models before normalization
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    nn_model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42)
    nn_model.fit(X_train, y_train)
    
    # Evaluate Models before normalization
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2, y_pred
    
    # Evaluate Models before normalization
    rf_mse, rf_r2, rf_pred = evaluate_model(rf_model, X_test, y_test)
    lr_mse, lr_r2, lr_pred = evaluate_model(lr_model, X_test, y_test)
    nn_mse, nn_r2, nn_pred = evaluate_model(nn_model, X_test, y_test)
    
    print(Fore.BLACK + "Random Forest Evaluation (Before Normalization):")
    print(Fore.GREEN + f"MSE: {rf_mse:.2f}, R²: {rf_r2:.2f}")
    
    print(Fore.BLACK + "\nLinear Regression Evaluation (Before Normalization):")
    print(Fore.GREEN + f"MSE: {lr_mse:.2f}, R²: {lr_r2:.2f}")
    
    print(Fore.BLACK + "\nNeural Network Evaluation (Before Normalization):")
    print(Fore.GREEN + f"MSE: {nn_mse:.2f}, R²: {nn_r2:.2f}")
    
    # Normalize features using Z-score normalization
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Split normalized data into training and testing sets
    X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
    
    # Train models after normalization
    rf_model.fit(X_train_norm, y_train_norm)
    lr_model.fit(X_train_norm, y_train_norm)
    nn_model.fit(X_train_norm, y_train_norm)
    
    # Evaluate Models after normalization
    rf_mse_norm, rf_r2_norm, rf_pred_norm = evaluate_model(rf_model, X_test_norm, y_test_norm)
    lr_mse_norm, lr_r2_norm, lr_pred_norm = evaluate_model(lr_model, X_test_norm, y_test_norm)
    nn_mse_norm, nn_r2_norm, nn_pred_norm = evaluate_model(nn_model, X_test_norm, y_test_norm)
    
    print(Fore.BLUE + "\nRandom Forest Evaluation (After Normalization):")
    print(Fore.GREEN + f"MSE: {rf_mse_norm:.2f}, R²: {rf_r2_norm:.2f}")
    
    print(Fore.BLUE + "\nLinear Regression Evaluation (After Normalization):")
    print(Fore.GREEN + f"MSE: {lr_mse_norm:.2f}, R²: {lr_r2_norm:.2f}")
    
    print(Fore.BLUE + "\nNeural Network Evaluation (After Normalization):")
    print(Fore.GREEN + f"MSE: {nn_mse_norm:.2f}, R²: {nn_r2_norm:.2f}")
    
    # Plot Comparison: Actual vs Predicted Costs for Each Algorithm
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))  # Create 3 subplots vertically
    
    # Plot for Random Forest
    axes[0].plot(y_test.values, label='Actual Costs', linestyle='-', color='black', linewidth=2)
    axes[0].plot(rf_pred_norm, label='RF Predicted (Normalized)', linestyle='-', color='green', linewidth=2)
    axes[0].set_title('Random Forest: Actual vs Predicted Costs')
    axes[0].set_xlabel('Test Data Points')
    axes[0].set_ylabel('Cost per Unit (₹)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot for Linear Regression
    axes[1].plot(y_test.values, label='Actual Costs', linestyle='-', color='black', linewidth=2)
    axes[1].plot(lr_pred_norm, label='LR Predicted (Normalized)', linestyle='-', color='blue', linewidth=2)
    axes[1].set_title('Linear Regression: Actual vs Predicted Costs')
    axes[1].set_xlabel('Test Data Points')
    axes[1].set_ylabel('Cost per Unit (₹)')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot for Neural Network
    axes[2].plot(y_test.values, label='Actual Costs', linestyle='-', color='black', linewidth=2)
    axes[2].plot(nn_pred_norm, label='NN Predicted (Normalized)', linestyle='-', color='red', linewidth=2)
    axes[2].set_title('Neural Network (MLP): Actual vs Predicted Costs')
    axes[2].set_xlabel('Test Data Points')
    axes[2].set_ylabel('Cost per Unit (₹)')
    axes[2].legend()
    axes[2].grid(True)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Display the plots
    plt.show()
    
    
    # Plotting Normalized Evaluation Metrics for All Algorithms
    def plot_normalized_metrics(rf_mse, rf_r2, lr_mse, lr_r2, nn_mse, nn_r2):
        # Prepare data for plotting
        algorithms = ['Random Forest', 'Linear Regression', 'Neural Network']
        mse_values = [rf_mse, lr_mse, nn_mse]
        r2_values = [rf_r2, lr_r2, nn_r2]
    
        # Normalize MSE values for better comparison with R² scores
        mse_values_normalized = [mse / max(mse_values) for mse in mse_values]
    
        # Create the bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
    
        # Bar width and positions
        x = np.arange(len(algorithms))
        bar_width = 0.35
    
        # Plot bars for Normalized MSE and R² Score
        bars1 = ax.bar(x - bar_width / 2, mse_values_normalized, bar_width, label='Normalized MSE', color='skyblue')
        bars2 = ax.bar(x + bar_width / 2, r2_values, bar_width, label='R² Score', color='lightgreen')
    
        # Add labels, title, and legend
        ax.set_xlabel('Algorithms')
        ax.set_ylabel('Metrics (Normalized)')
        ax.set_title('Normalized Evaluation Metrics for All Algorithms')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        ax.legend()
    
        # Display the plot
        plt.show()
    
    # Call the function with the evaluation metrics
    plot_normalized_metrics(
        rf_mse_norm, rf_r2_norm, 
        lr_mse_norm, lr_r2_norm, 
        nn_mse_norm, nn_r2_norm
    )
    
    
    
    # Charging Recommendation System
    def calculate_charging_cost(battery_capacity, battery_level, cost_per_unit):
        energy_needed_kWh = battery_capacity * (100 - battery_level) / 100
        return energy_needed_kWh * cost_per_unit
    
    def calculate_charging_time(battery_capacity, battery_level, charging_speed):
        energy_needed_kWh = battery_capacity * (100 - battery_level) / 100
        if charging_speed <= 0:
            return None  
        return round(energy_needed_kWh / charging_speed, 2)
    
    def recommend_stations(user_lat, user_lon, battery_level, battery_capacity, top_n=3, max_distance_km=50):
        data['distance_km'] = data.apply(lambda row: haversine(user_lat, user_lon, row['latitude'], row['longitude']), axis=1)
        nearby_stations = data[data['distance_km'] <= max_distance_km]
        if nearby_stations.empty:
            print(Fore.RED + "No stations found within the specified distance.")
            return pd.DataFrame(), None
    
        top_stations = nearby_stations.sort_values(by='distance_km').head(top_n)
        top_stations['predicted_cost'] = top_stations['cost_per_unit'].apply(
            lambda cost: calculate_charging_cost(battery_capacity, battery_level, cost)
        )
        return top_stations, top_stations['predicted_cost'].mean()
    
    def find_best_station(stations):
        if stations.empty:
            print(Fore.RED + "No stations available for recommendations.")
            return None
    
        stations['score'] = (0.7 * stations['predicted_cost'] / stations['predicted_cost'].max() +
                             0.3 * stations['distance_km'] / stations['distance_km'].max())
        return stations.loc[stations['score'].idxmin()]
    
    
    
    #INPUT -GUINDY, CHENNAI
    latitude = 13.0067
    longitude = 80.2206
    battery_level = 49.9  
    battery_capacity = 55.0
    
    recommended_stations, avg_cost = recommend_stations(latitude, longitude, battery_level, battery_capacity)
    
    energy_needed_kWh = battery_capacity * (100 - battery_level) / 100
    charging_speed = 25.0  
    charging_time = calculate_charging_time(battery_capacity, battery_level, charging_speed)
    
    if battery_level < 30:
        print(Fore.YELLOW + "\nPREDICTED AVERAGE COST OF CHARGING 'EV' :")
        print(Fore.RED + f"₹{avg_cost:.2f} ")
        print(Fore.YELLOW + f"\nESTIMATED CHARGING TIME TO FULL FROM ({battery_level}% to 100%) : ")
        print(Fore.RED + f"{charging_time} hours")
        
        if not recommended_stations.empty:
            print(Fore.BLUE + "RECOMMENDED CHARGING STATIONS:")
            print(recommended_stations[['address', 'city', 'capacity', 'cost_per_unit', 'distance_km', 'predicted_cost']])
        
        best_station = find_best_station(recommended_stations)
        if best_station is not None:
            print(Fore.YELLOW + "\nBEST USER/COST-FRIENDLY CHARGING STATION:")
            print(Fore.GREEN + f"Address: {best_station['address']}")
            print(Fore.GREEN + f"City: {best_station['city']}")
            print(Fore.GREEN + f"Capacity: {best_station['capacity']} kW")
            print(Fore.GREEN + f"Cost per Unit: ₹{best_station['cost_per_unit']:.2f}")
            print(Fore.GREEN + f"Distance: {best_station['distance_km']:.2f} km")
            print(Fore.GREEN + f"Predicted Cost: ₹{best_station['predicted_cost']:.2f}")
    
    else:
        print(Fore.YELLOW + "\nPREDICTED AVERAGE COST OF CHARGING 'EV' :")
        print(Fore.RED + f"₹{avg_cost:.2f} ")
        print(Fore.YELLOW + f"\nESTIMATED CHARGING TIME TO FULL FROM ({battery_level}% to 100%) : ")
        print(Fore.RED + f"{charging_time} hours")
    
        if not recommended_stations.empty:
            best_station = find_best_station(recommended_stations)
            if best_station is not None:
                print(Fore.GREEN + f"\nBEST STATION:\n{best_station[['address', 'city', 'capacity', 'predicted_cost', 'distance_km']]}")
    
