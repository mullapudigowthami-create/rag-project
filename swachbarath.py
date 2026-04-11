# ============================================
# SWACHH BHARAT AI ECOSYSTEM
# Built by: [Your Name]
# Target: Generative AI
# Modules: Waste Classification, Plastic Tracker,
#          Cleanliness Monitor, Chatbot,
#          Disposal Finder, Violation Reporter,
#          Daily Announcements, Sentiment Analysis
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime
import random

print("=" * 50)
print("  SWACHH BHARAT AI ECOSYSTEM STARTED! 🚀")
print("=" * 50)

# ============================================
# MODULE 1 - WASTE CLASSIFICATION
# ============================================
print("\n📦 MODULE 1: WASTE CLASSIFICATION")
print("-" * 40)

# Sample waste data
waste_data = {
    'weight_kg': [0.5, 2.0, 0.1, 5.0, 1.5, 0.3, 3.0, 0.8, 4.0, 0.2],
    'is_liquid': [0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'is_biodegradable': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    'waste_type': ['Wet', 'Plastic', 'Liquid', 'Metal',
                   'Wet', 'Liquid', 'Plastic', 'Wet',
                   'Metal', 'Liquid']
}

df_waste = pd.DataFrame(waste_data)

# Encode waste types
waste_map = {'Wet': 0, 'Plastic': 1, 'Liquid': 2, 'Metal': 3}
df_waste['waste_encoded'] = df_waste['waste_type'].map(waste_map)

X = df_waste[['weight_kg', 'is_liquid', 'is_biodegradable']]
y = df_waste['waste_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

waste_model = DecisionTreeClassifier()
waste_model.fit(X_train, y_train)

def classify_waste(weight, is_liquid, is_biodegradable):
    prediction = waste_model.predict(
        [[weight, is_liquid, is_biodegradable]]
    )
    reverse_map = {0: 'Wet Waste', 1: 'Plastic Waste',
                   2: 'Liquid Waste', 3: 'Metal Waste'}
    return reverse_map[prediction[0]]

print("Testing Waste Classification:")
print(f"Sample 1: {classify_waste(0.5, 0, 1)}")
print(f"Sample 2: {classify_waste(2.0, 0, 0)}")
print(f"Sample 3: {classify_waste(0.1, 1, 1)}")

# ============================================
# MODULE 2 - PLASTIC TO PETROL TRACKER
# ============================================
print("\n⛽ MODULE 2: PLASTIC TO PETROL TRACKER")
print("-" * 40)

def plastic_to_petrol(plastic_kg):
    # 1 kg plastic = approx 0.8 litres of fuel
    fuel_litres = plastic_kg * 0.8
    co2_saved = plastic_kg * 2.5  # kg of CO2 saved
    money_saved = fuel_litres * 100  # Rs per litre approx
    return fuel_litres, co2_saved, money_saved

plastics_collected = [10, 25, 15, 30, 20]
total_plastic = sum(plastics_collected)
fuel, co2, money = plastic_to_petrol(total_plastic)

print(f"Total Plastic Collected: {total_plastic} kg")
print(f"Fuel Generated: {fuel} litres")
print(f"CO2 Saved: {co2} kg")
print(f"Money Saved: Rs {money}")

# ============================================
# MODULE 3 - CLEANLINESS MONITORING
# ============================================
print("\n🔍 MODULE 3: CLEANLINESS MONITORING")
print("-" * 40)

areas = ['Area A', 'Area B', 'Area C', 'Area D', 'Area E']
cleanliness_scores = [85, 45, 90, 30, 70]

for area, score in zip(areas, cleanliness_scores):
    if score >= 75:
        status = "✅ CLEAN"
    elif score >= 50:
        status = "⚠️ NEEDS ATTENTION"
    else:
        status = "❌ CRITICAL - ALERT SENT!"
    print(f"{area}: {score}% - {status}")

# ============================================
# MODULE 4 - SENTIMENT ANALYSIS
# ============================================
print("\n😊 MODULE 4: PUBLIC SENTIMENT ANALYSIS")
print("-" * 40)

try:
    sentiment_analyzer = pipeline("sentiment-analysis")
    public_opinions = [
        "Swachh Bharat is making our city beautiful!",
        "Still so much garbage on the streets!",
        "Cleanliness drives are working wonderfully!",
        "Nobody follows the rules it is very dirty!"
    ]

    for opinion in public_opinions:
        result = sentiment_analyzer(opinion)[0]
        emoji = "😊" if result['label'] == 'POSITIVE' else "😞"
        print(f"{emoji} {opinion[:40]}...")
        print(f"   Sentiment: {result['label']} "
              f"({round(result['score']*100, 1)}%)")
except:
    print("Sentiment model loading... please wait")

# ============================================
# MODULE 5 - DISPOSAL LOCATION FINDER
# ============================================
print("\n📍 MODULE 5: DISPOSAL LOCATION FINDER")
print("-" * 40)

disposal_locations = {
    'Hyderabad Central': (17.3850, 78.4867),
    'Secunderabad': (17.4399, 78.4983),
    'Kukatpally': (17.4849, 78.3995),
    'Madhapur': (17.4474, 78.3762),
    'LB Nagar': (17.3486, 78.5488)
}

print("Nearest Disposal Locations:")
for location, coords in disposal_locations.items():
    print(f"📍 {location}: Lat {coords[0]}, Lon {coords[1]}")

# ============================================
# MODULE 6 - VIOLATION REPORTER
# ============================================
print("\n⚖️ MODULE 6: VIOLATION REPORTER")
print("-" * 40)

violations = [
    {'area': 'Area B', 'violation': 'Dumping on roadside',
     'fine': 500},
    {'area': 'Area D', 'violation': 'Burning waste',
     'fine': 1000},
    {'area': 'Area A', 'violation': 'No waste segregation',
     'fine': 200},
]

for v in violations:
    print(f"🚨 {v['area']}: {v['violation']}")
    print(f"   Fine: Rs {v['fine']}")

# ============================================
# MODULE 7 - DAILY ANNOUNCEMENTS
# ============================================
print("\n📢 MODULE 7: DAILY ANNOUNCEMENTS")
print("-" * 40)

today = datetime.datetime.now()
day_of_week = today.strftime("%A")

announcements = {
    'Monday': "🧹 Weekly cleaning drive today! All residents participate!",
    'Tuesday': "♻️ Plastic collection day! Keep plastic waste ready!",
    'Wednesday': "🌱 Wet waste composting awareness program today!",
    'Thursday': "🚛 Municipal truck arrives at 8AM! Keep waste ready!",
    'Friday': "🏘️ Area inspection day! Keep surroundings clean!",
    'Saturday': "👨‍👩‍👧 Community cleanliness drive! Join your neighbors!",
    'Sunday': "📚 Cleanliness awareness program for children today!"
}

print(f"Today is {day_of_week}")
print(f"Announcement: {announcements.get(day_of_week, 'Stay Clean!')}")

# ============================================
# MODULE 8 - SWACHH BHARAT CHATBOT
# ============================================
print("\n🤖 MODULE 8: SWACHH BHARAT CHATBOT")
print("-" * 40)

def swachh_chatbot(question):
    question = question.lower()
    if 'dustbin' in question or 'disposal' in question:
        return "📍 Nearest dustbin is at Hyderabad Central. 500m away!"
    elif 'plastic' in question:
        return "♻️ Plastic waste goes in BLUE dustbin. We convert it to fuel!"
    elif 'wet' in question or 'food' in question:
        return "🌱 Wet/Food waste goes in GREEN dustbin for composting!"
    elif 'fine' in question or 'penalty' in question:
        return "⚖️ Dumping waste illegally = Rs 500 fine. Burning = Rs 1000!"
    elif 'clean' in question:
        return "✅ To keep your area clean, dispose waste properly."