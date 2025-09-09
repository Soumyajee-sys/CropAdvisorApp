import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np

def train_enhanced_model():
    """
    Train an enhanced crop prediction model that supports probability predictions
    """
    
    # Load the crop prediction dataset
    try:
        df = pd.read_csv("crop_prediction_dataset.csv")
    except FileNotFoundError:
        print("Error: crop_prediction_dataset.csv not found")
        return
    
    print(f"Dataset shape: {df.shape}")
    print(f"Available crops: {df['Crop_Grown'].unique()}")
    
    # Features: soil and environmental factors
    feature_columns = ['Soil_Type', 'Soil_pH', 'N', 'P', 'K', 'Soil_Texture', 'Irrigation_Type', 'District']
    X = df[feature_columns]
    y = df['Crop_Grown']  # Target: crop type
    
    print(f"Features: {feature_columns}")
    print(f"Target classes: {y.unique()}")
    
    # Encode categorical columns
    label_encoders = {}
    X_encoded = X.copy()
    
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
            label_encoders[col] = le
            print(f"Encoded {col}: {len(le.classes_)} classes")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train enhanced Random Forest model with probability support
    model = RandomForestClassifier(
        n_estimators=200,  # More trees for better probability estimates
        max_depth=15,      # Controlled depth to prevent overfitting
        random_state=42,
        class_weight='balanced',  # Handle class imbalance
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"\n=== Model Performance ===")
    print(f"Training Accuracy: {train_accuracy:.3f}")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    
    # Test probability predictions
    print(f"\n=== Probability Support ===")
    print(f"Model supports predict_proba: {hasattr(model, 'predict_proba')}")
    
    if hasattr(model, 'predict_proba'):
        # Test with first sample
        sample_proba = model.predict_proba(X_test.iloc[:1])
        print(f"Sample probabilities shape: {sample_proba.shape}")
        print(f"Classes: {model.classes_}")
        print(f"Sample probabilities: {sample_proba[0]}")
    
    # Get detailed predictions for evaluation
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print(f"\n=== Detailed Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n=== Feature Importance ===")
    print(feature_importance)
    
    # Confidence analysis
    print(f"\n=== Confidence Analysis ===")
    max_probabilities = np.max(y_pred_proba, axis=1)
    print(f"Average prediction confidence: {np.mean(max_probabilities):.3f}")
    print(f"Min confidence: {np.min(max_probabilities):.3f}")
    print(f"Max confidence: {np.max(max_probabilities):.3f}")
    
    high_confidence = np.sum(max_probabilities >= 0.7)
    medium_confidence = np.sum((max_probabilities >= 0.5) & (max_probabilities < 0.7))
    low_confidence = np.sum(max_probabilities < 0.5)
    
    print(f"High confidence predictions (>=0.7): {high_confidence} ({high_confidence/len(max_probabilities)*100:.1f}%)")
    print(f"Medium confidence predictions (0.5-0.7): {medium_confidence} ({medium_confidence/len(max_probabilities)*100:.1f}%)")
    print(f"Low confidence predictions (<0.5): {low_confidence} ({low_confidence/len(max_probabilities)*100:.1f}%)")
    
    # Save the enhanced model and encoders
    model_filename = "enhanced_crop_prediction_model.pkl"
    encoders_filename = "enhanced_crop_label_encoders.pkl"
    
    joblib.dump(model, model_filename)
    joblib.dump(label_encoders, encoders_filename)
    
    print(f"\n=== Model Saved ===")
    print(f"Model saved as: {model_filename}")
    print(f"Encoders saved as: {encoders_filename}")
    
    # Test the saved model
    print(f"\n=== Testing Saved Model ===")
    loaded_model = joblib.load(model_filename)
    loaded_encoders = joblib.load(encoders_filename)
    
    # Test with sample data
    test_sample = X_test.iloc[:1]
    prediction = loaded_model.predict(test_sample)[0]
    probabilities = loaded_model.predict_proba(test_sample)[0]
    
    print(f"Test prediction: {prediction}")
    print(f"Test probabilities: {dict(zip(loaded_model.classes_, probabilities))}")
    
    return model, label_encoders

def test_model_integration():
    """
    Test the model integration with sample soil data
    """
    try:
        model = joblib.load("enhanced_crop_prediction_model.pkl")
        encoders = joblib.load("enhanced_crop_label_encoders.pkl")
    except FileNotFoundError:
        print("Enhanced model not found. Please run train_enhanced_model() first.")
        return
    
    # Sample soil data for testing
    sample_districts = ["Ranchi", "Dhanbad", "Dumka"]
    
    for district in sample_districts:
        print(f"\n=== Testing for {district} ===")
        
        # Create sample soil data
        soil_data = {
            "Soil_Type": "Red and Yellow",
            "Soil_pH": 6.0,
            "N": 60,
            "P": 35,
            "K": 40,
            "Soil_Texture": "Loamy",
            "Irrigation_Type": "Canal",
            "District": district
        }
        
        # Build features
        features = pd.DataFrame([soil_data])
        
        # Encode categorical features
        for col in ["Soil_Type", "Soil_Texture", "Irrigation_Type", "District"]:
            if col in features and col in encoders:
                le = encoders[col]
                try:
                    features[col] = le.transform([features[col][0]])
                except ValueError:
                    features[col] = le.transform([le.classes_[0]])
                    print(f"Warning: Unknown category for {col}, using default")
        
        # Get predictions
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        print(f"Top prediction: {prediction}")
        print("All crop probabilities:")
        
        # Sort by probability
        crop_probs = list(zip(model.classes_, probabilities))
        crop_probs.sort(key=lambda x: x[1], reverse=True)
        
        for crop, prob in crop_probs:
            print(f"  {crop}: {prob:.3f} ({prob*100:.1f}%)")

if __name__ == "__main__":
    print("=== Training Enhanced Crop Prediction Model ===")
    model, encoders = train_enhanced_model()
    
    print("\n" + "="*50)
    print("=== Testing Model Integration ===")
    test_model_integration()