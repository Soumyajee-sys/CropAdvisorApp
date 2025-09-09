from main import app
from mandiPrice import (
    get_enhanced_price, get_price_volatility, 
    get_seasonal_price_comparison, calculate_profit_margins
)
import pandas as pd
import numpy as np

# Additional utility functions for the integrated system

def get_crop_suitability_matrix(district, soil_data):
    """
    Generate a comprehensive suitability matrix for all crops
    """
    try:
        from main import get_ai_predictions_with_confidence, CROP_MAPPING
        
        ai_predictions = get_ai_predictions_with_confidence(soil_data)
        
        suitability_matrix = {}
        
        for crop_key, commodity in CROP_MAPPING.items():
            ai_confidence = ai_predictions.get(crop_key, 0.5)
            
            # Get market data
            rabi_price, rabi_source, rabi_confidence = get_enhanced_price(district, commodity, "Rabi")
            kharif_price, kharif_source, kharif_confidence = get_enhanced_price(district, commodity, "Kharif")
            
            # Get price volatility
            volatility = get_price_volatility(commodity)
            
            # Calculate profit margins
            rabi_profits = calculate_profit_margins(rabi_price, crop_key)
            kharif_profits = calculate_profit_margins(kharif_price, crop_key)
            
            suitability_matrix[crop_key.title()] = {
                "ai_suitability": {
                    "confidence": round(ai_confidence * 100, 1),
                    "recommendation": "High" if ai_confidence >= 0.7 else "Medium" if ai_confidence >= 0.4 else "Low"
                },
                "market_analysis": {
                    "rabi_season": {
                        "price": round(rabi_price, 2),
                        "source_confidence": rabi_confidence,
                        "roi_percentage": rabi_profits['return_on_investment_percentage']
                    },
                    "kharif_season": {
                        "price": round(kharif_price, 2),
                        "source_confidence": kharif_confidence,
                        "roi_percentage": kharif_profits['return_on_investment_percentage']
                    },
                    "price_risk": volatility['risk_level'],
                    "best_season": "Rabi" if rabi_price > kharif_price else "Kharif"
                },
                "overall_score": round(
                    (ai_confidence * 100 * 0.4) + 
                    (max(rabi_profits['return_on_investment_percentage'], 
                         kharif_profits['return_on_investment_percentage']) * 0.6), 1
                )
            }
        
        # Sort by overall score
        sorted_crops = sorted(suitability_matrix.items(), 
                            key=lambda x: x[1]['overall_score'], reverse=True)
        
        return dict(sorted_crops)
        
    except Exception as e:
        print(f"Error generating suitability matrix: {e}")
        return {}

def generate_farming_calendar(district, recommended_crops, lat, lon):
    """
    Generate a comprehensive farming calendar for recommended crops
    """
    try:
        calendar_df = pd.read_csv("crop_calendar.csv")
        calendar_df.columns = calendar_df.columns.str.lower()
    except:
        return {"error": "Crop calendar data not available"}
    
    farming_calendar = {}
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    
    for month in months:
        farming_calendar[month] = {
            "sowing_activities": [],
            "harvesting_activities": [],
            "maintenance_activities": []
        }
    
    for crop in recommended_crops[:3]:  # Top 3 recommended crops
        crop_data = calendar_df[
            (calendar_df['district'].str.lower() == district.lower()) &
            (calendar_df['crop'].str.lower() == crop.lower())
        ]
        
        if not crop_data.empty:
            for _, row in crop_data.iterrows():
                # Sowing activities
                sowing_months = [row['sowing_start_month'], row['sowing_end_month']]
                for month in sowing_months:
                    if month in farming_calendar:
                        farming_calendar[month]["sowing_activities"].append({
                            "crop": crop,
                            "activity": "Sowing",
                            "season": row['season'],
                            "water_requirement": row.get('water_requirement', 'Unknown')
                        })
                
                # Harvesting activities
                harvesting_months = [row['harvesting_start_month'], row['harvesting_end_month']]
                for month in harvesting_months:
                    if month in farming_calendar:
                        farming_calendar[month]["harvesting_activities"].append({
                            "crop": crop,
                            "activity": "Harvesting",
                            "season": row['season'],
                            "expected_yield": f"{row.get('growth_duration_days', 'Unknown')} days cycle"
                        })
    
    return farming_calendar

def calculate_portfolio_optimization(district, soil_data, total_acres=10):
    """
    Optimize crop portfolio for maximum returns with risk management
    """
    suitability_matrix = get_crop_suitability_matrix(district, soil_data)
    
    if not suitability_matrix:
        return {"error": "Unable to generate portfolio optimization"}
    
    # Get top crops with different risk profiles
    crops_data = []
    
    for crop, data in suitability_matrix.items():
        market_data = data['market_analysis']
        best_roi = max(market_data['rabi_season']['roi_percentage'],
                      market_data['kharif_season']['roi_percentage'])
        
        risk_score = {"Low": 1, "Medium": 2, "High": 3}.get(market_data['price_risk'], 2)
        
        crops_data.append({
            "crop": crop,
            "expected_roi": best_roi,
            "risk_score": risk_score,
            "overall_score": data['overall_score'],
            "ai_confidence": data['ai_suitability']['confidence']
        })
    
    # Sort by risk-adjusted returns
    crops_data.sort(key=lambda x: x['expected_roi'] / x['risk_score'], reverse=True)
    
    # Portfolio allocation strategy
    portfolio = []
    remaining_acres = total_acres
    
    if len(crops_data) >= 3:
        # Diversified portfolio
        primary_crop = crops_data[0]
        secondary_crop = crops_data[1]
        tertiary_crop = crops_data[2]
        
        # Allocate based on scores and risk management
        primary_allocation = min(0.6 * total_acres, remaining_acres)
        remaining_acres -= primary_allocation
        
        secondary_allocation = min(0.3 * total_acres, remaining_acres)
        remaining_acres -= secondary_allocation
        
        tertiary_allocation = remaining_acres
        
        portfolio = [
            {
                "crop": primary_crop["crop"],
                "acres": round(primary_allocation, 1),
                "percentage": round((primary_allocation / total_acres) * 100, 1),
                "expected_roi": primary_crop["expected_roi"],
                "risk_level": "Primary (Highest confidence)"
            },
            {
                "crop": secondary_crop["crop"],
                "acres": round(secondary_allocation, 1),
                "percentage": round((secondary_allocation / total_acres) * 100, 1),
                "expected_roi": secondary_crop["expected_roi"],
                "risk_level": "Secondary (Diversification)"
            },
            {
                "crop": tertiary_crop["crop"],
                "acres": round(tertiary_allocation, 1),
                "percentage": round((tertiary_allocation / total_acres) * 100, 1),
                "expected_roi": tertiary_crop["expected_roi"],
                "risk_level": "Tertiary (Risk management)"
            }
        ]
    else:
        # Simple allocation if fewer crops available
        for i, crop_data in enumerate(crops_data):
            allocation = total_acres / len(crops_data)
            portfolio.append({
                "crop": crop_data["crop"],
                "acres": round(allocation, 1),
                "percentage": round((allocation / total_acres) * 100, 1),
                "expected_roi": crop_data["expected_roi"],
                "risk_level": f"Rank {i+1}"
            })
    
    # Calculate portfolio metrics
    total_expected_return = sum([p["expected_roi"] * p["acres"] for p in portfolio])
    average_roi = total_expected_return / total_acres if total_acres > 0 else 0
    
    return {
        "total_acres": total_acres,
        "portfolio_allocation": portfolio,
        "portfolio_metrics": {
            "expected_average_roi": round(average_roi, 1),
            "diversification_level": len(portfolio),
            "risk_assessment": "Diversified" if len(portfolio) >= 3 else "Concentrated"
        }
    }

# Test the utility functions
if __name__ == "__main__":
    print("=== Testing Integrated System Utilities ===")
    
    # Sample soil data
    sample_soil = {
        "district": "Ranchi",
        "soil_type": "Red and Yellow",
        "soil_ph": 6.0,
        "n": 60,
        "p": 35,
        "k": 40,
        "soil_texture": "Loamy",
        "irrigation_type": "Canal"
    }
    
    # Test suitability matrix
    print("\n--- Crop Suitability Matrix ---")
    matrix = get_crop_suitability_matrix("Ranchi", sample_soil)
    for crop, data in list(matrix.items())[:3]:
        print(f"{crop}: Overall Score {data['overall_score']}")
        print(f"  AI Confidence: {data['ai_suitability']['confidence']}%")
        print(f"  Best Season: {data['market_analysis']['best_season']}")
    
    # Test portfolio optimization
    print("\n--- Portfolio Optimization ---")
    portfolio = calculate_portfolio_optimization("Ranchi", sample_soil, 10)
    if "portfolio_allocation" in portfolio:
        for allocation in portfolio["portfolio_allocation"]:
            print(f"{allocation['crop']}: {allocation['acres']} acres "
                  f"({allocation['percentage']}%) - ROI: {allocation['expected_roi']}%")
        print(f"Portfolio Average ROI: {portfolio['portfolio_metrics']['expected_average_roi']}%")