import pandas as pd

def get_enhanced_price(district, commodity, season="Rabi"):
    """
    Enhanced price retrieval with multiple fallback strategies and price trends
    """
    try:
        prices_df = pd.read_csv("mandi_price.csv")
        prices_df = prices_df[['District', 'Commodity', 'Price (Rs/Quintal)', 'Season']].copy()
    except FileNotFoundError:
        # Use price ranges as fallback
        price_ranges = {
            'Rice (Paddy)': {'min': 1347.98, 'max': 2266.93, 'mean': 1806.08},
            'Maize': {'min': 1233.58, 'max': 2246.11, 'mean': 1693.93},
            'Wheat': {'min': 1623.35, 'max': 2532.89, 'mean': 2038.65},
            'Arhar (Tur)': {'min': 5354.83, 'max': 8391.58, 'mean': 6765.03},
            'Sugarcane': {'min': 249.25, 'max': 393.07, 'mean': 315.55}
        }
        price_info = price_ranges.get(commodity, {'mean': 2000})
        return price_info['mean'], "estimated", 0.5
    
    # District name mapping
    district_mapping = {
        'Saraikela-Kharsawan': 'Saraikela Kharsawan'
    }
    mandi_district = district_mapping.get(district, district)
    
    # Strategy 1: Exact match (district + commodity + season)
    exact_match = prices_df[
        (prices_df['District'] == mandi_district) & 
        (prices_df['Commodity'] == commodity) &
        (prices_df['Season'] == season)
    ]
    
    if not exact_match.empty:
        price = float(exact_match['Price (Rs/Quintal)'].iloc[0])
        return price, "district_specific", 1.0
    
    # Strategy 2: District + commodity (any season)
    district_commodity = prices_df[
        (prices_df['District'] == mandi_district) & 
        (prices_df['Commodity'] == commodity)
    ]
    
    if not district_commodity.empty:
        price = float(district_commodity['Price (Rs/Quintal)'].mean())
        return price, "district_average", 0.8
    
    # Strategy 3: Commodity + season (state average)
    commodity_season = prices_df[
        (prices_df['Commodity'] == commodity) &
        (prices_df['Season'] == season)
    ]
    
    if not commodity_season.empty:
        price = float(commodity_season['Price (Rs/Quintal)'].mean())
        return price, "state_average", 0.7
    
    # Strategy 4: Commodity only (all seasons, all districts)
    commodity_only = prices_df[prices_df['Commodity'] == commodity]
    
    if not commodity_only.empty:
        price = float(commodity_only['Price (Rs/Quintal)'].mean())
        return price, "national_average", 0.6
    
    # Strategy 5: Default fallback
    return 2000.0, "default_estimate", 0.3

def get_price_volatility(commodity, season="Rabi"):
    """
    Calculate price volatility/risk for a commodity
    """
    try:
        prices_df = pd.read_csv("mandi_price.csv")
        prices_df = prices_df[['District', 'Commodity', 'Price (Rs/Quintal)', 'Season']].copy()
        
        commodity_prices = prices_df[
            (prices_df['Commodity'] == commodity) &
            (prices_df['Season'] == season)
        ]['Price (Rs/Quintal)']
        
        if len(commodity_prices) > 1:
            std_dev = float(commodity_prices.std())
            mean_price = float(commodity_prices.mean())
            cv = (std_dev / mean_price) * 100  # Coefficient of variation
            
            risk_level = "Low" if cv < 10 else "Medium" if cv < 20 else "High"
            
            return {
                "standard_deviation": round(std_dev, 2),
                "coefficient_of_variation": round(cv, 2),
                "risk_level": risk_level,
                "min_price": float(commodity_prices.min()),
                "max_price": float(commodity_prices.max()),
                "price_range": float(commodity_prices.max() - commodity_prices.min())
            }
        else:
            return {
                "standard_deviation": 0,
                "coefficient_of_variation": 0,
                "risk_level": "Unknown",
                "min_price": 0,
                "max_price": 0,
                "price_range": 0
            }
    except:
        return {
            "standard_deviation": 0,
            "coefficient_of_variation": 0,
            "risk_level": "Unknown",
            "min_price": 0,
            "max_price": 0,
            "price_range": 0
        }

def get_seasonal_price_comparison(commodity):
    """
    Compare prices across seasons for a commodity
    """
    try:
        prices_df = pd.read_csv("mandi_price.csv")
        prices_df = prices_df[['District', 'Commodity', 'Price (Rs/Quintal)', 'Season']].copy()
        
        seasonal_prices = prices_df[
            prices_df['Commodity'] == commodity
        ].groupby('Season')['Price (Rs/Quintal)'].agg(['mean', 'count']).round(2)
        
        result = {}
        for season in seasonal_prices.index:
            result[season.lower()] = {
                "average_price": float(seasonal_prices.loc[season, 'mean']),
                "data_points": int(seasonal_prices.loc[season, 'count'])
            }
        
        # Determine best season
        if len(result) > 1:
            best_season = max(result.keys(), key=lambda x: result[x]['average_price'])
            result['best_season'] = best_season
            result['price_advantage'] = round(
                result[best_season]['average_price'] - 
                min([result[s]['average_price'] for s in result.keys() if s != 'best_season']), 2
            )
        
        return result
        
    except:
        return {"error": "Unable to calculate seasonal comparison"}

def calculate_profit_margins(price, crop, input_costs=None):
    """
    Calculate estimated profit margins for a crop
    """
    # Default input costs per acre (in rupees) - these are rough estimates
    default_input_costs = {
        'rice': 15000,
        'wheat': 12000,
        'maize': 10000,
        'pulses': 8000,
        'sugarcane': 25000
    }
    
    # Expected yields (quintals per acre)
    yields = {
        'rice': 25,
        'wheat': 20,
        'maize': 18,
        'pulses': 8,
        'sugarcane': 300
    }
    
    crop_lower = crop.lower()
    input_cost = input_costs or default_input_costs.get(crop_lower, 10000)
    expected_yield = yields.get(crop_lower, 15)
    
    gross_revenue = price * expected_yield
    net_profit = gross_revenue - input_cost
    profit_margin = (net_profit / gross_revenue) * 100 if gross_revenue > 0 else 0
    roi = (net_profit / input_cost) * 100 if input_cost > 0 else 0
    
    return {
        "gross_revenue_per_acre": round(gross_revenue, 2),
        "input_costs_per_acre": input_cost,
        "net_profit_per_acre": round(net_profit, 2),
        "profit_margin_percentage": round(profit_margin, 1),
        "return_on_investment_percentage": round(roi, 1),
        "profitability_category": "High" if roi >= 50 else "Medium" if roi >= 25 else "Low"
    }

# Test the enhanced pricing functions
if __name__ == "__main__":
    print("=== Testing Enhanced Price Functions ===")
    
    # Test enhanced price retrieval
    commodities = ['Rice (Paddy)', 'Wheat', 'Maize', 'Arhar (Tur)', 'Sugarcane']
    districts = ['Ranchi', 'Dhanbad', 'Dumka']
    
    for commodity in commodities[:2]:  # Test first 2 commodities
        print(f"\n--- {commodity} Analysis ---")
        
        # Price volatility
        volatility = get_price_volatility(commodity)
        print(f"Price Risk: {volatility['risk_level']} (CV: {volatility['coefficient_of_variation']}%)")
        
        # Seasonal comparison
        seasonal = get_seasonal_price_comparison(commodity)
        if 'best_season' in seasonal:
            print(f"Best Season: {seasonal['best_season']} (+₹{seasonal['price_advantage']}/quintal)")
        
        # District-wise pricing
        for district in districts[:2]:  # Test first 2 districts
            price, source, confidence = get_enhanced_price(district, commodity)
            crop_name = [k for k, v in {'Rice': 'Rice (Paddy)', 'Wheat': 'Wheat', 
                        'Maize': 'Maize', 'Pulses': 'Arhar (Tur)', 
                        'Sugarcane': 'Sugarcane'}.items() if v == commodity][0]
            
            profit_analysis = calculate_profit_margins(price, crop_name)
            
            print(f"  {district}: ₹{price:.2f}/quintal ({source}, confidence: {confidence:.1f})")
            print(f"    ROI: {profit_analysis['return_on_investment_percentage']}% "
                  f"({profit_analysis['profitability_category']} profitability)")