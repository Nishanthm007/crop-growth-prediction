import streamlit as st
import joblib
import numpy as np

crop_info = {
    "rice": {
        "ideal_temp": "20–35°C",
        "soil_type": "Clay, Loamy",
        "growing_season": "Kharif (June–Nov)",
        "profit_tip": "Needs standing water; ideal for irrigated fields."
    },
    "maize": {
        "ideal_temp": "18–27°C",
        "soil_type": "Well-drained loamy",
        "growing_season": "All seasons",
        "profit_tip": "High-yielding and drought-tolerant."
    },
    "jute": {
        "ideal_temp": "24–37°C",
        "soil_type": "Well-drained loamy, alluvial",
        "growing_season": "Kharif",
        "profit_tip": "Needs warm, humid climate and standing water."
    },
    "cotton": {
        "ideal_temp": "21–30°C",
        "soil_type": "Black soil, well-drained",
        "growing_season": "Kharif",
        "profit_tip": "Requires long frost-free periods and lots of sunshine."
    },
    "coconut": {
        "ideal_temp": "27–32°C",
        "soil_type": "Sandy loam, coastal soil",
        "growing_season": "Perennial",
        "profit_tip": "Highly profitable in coastal tropical climates."
    },
    "papaya": {
        "ideal_temp": "25–30°C",
        "soil_type": "Loamy, well-drained",
        "growing_season": "All year",
        "profit_tip": "Fast fruiting crop; high demand."
    },
    "orange": {
        "ideal_temp": "15–30°C",
        "soil_type": "Sandy loam",
        "growing_season": "Nov–Feb",
        "profit_tip": "Requires cool winters; avoid water-logging."
    },
    "apple": {
        "ideal_temp": "15–20°C",
        "soil_type": "Loamy, well-drained",
        "growing_season": "Temperate zones only",
        "profit_tip": "Suited for hilly regions; requires chilling hours."
    },
    "muskmelon": {
        "ideal_temp": "25–35°C",
        "soil_type": "Sandy loam",
        "growing_season": "Summer",
        "profit_tip": "Quick-growing; needs high sunlight."
    },
    "watermelon": {
        "ideal_temp": "25–40°C",
        "soil_type": "Sandy loam",
        "growing_season": "Summer",
        "profit_tip": "Needs full sun and ample water."
    },
    "grapes": {
        "ideal_temp": "15–40°C",
        "soil_type": "Black clay loam",
        "growing_season": "Dec–Feb",
        "profit_tip": "Requires good pruning and drainage."
    },
    "mango": {
        "ideal_temp": "24–30°C",
        "soil_type": "Well-drained loamy",
        "growing_season": "Perennial",
        "profit_tip": "High value; needs long dry season before flowering."
    },
    "banana": {
        "ideal_temp": "25–35°C",
        "soil_type": "Alluvial, loamy",
        "growing_season": "Throughout year",
        "profit_tip": "Needs high rainfall or irrigation; wind protection needed."
    },
    "pomegranate": {
        "ideal_temp": "25–30°C",
        "soil_type": "Loamy, well-drained",
        "growing_season": "Feb–May",
        "profit_tip": "Can be grown in semi-arid regions; minimal water needed."
    },
    "lentil": {
        "ideal_temp": "18–30°C",
        "soil_type": "Loamy to clay",
        "growing_season": "Rabi",
        "profit_tip": "Short duration pulse crop; improves soil fertility."
    },
    "blackgram": {
        "ideal_temp": "25–35°C",
        "soil_type": "Loamy, well-drained",
        "growing_season": "Kharif & Summer",
        "profit_tip": "Nitrogen-fixing legume; good for intercropping."
    },
    "mungbean": {
        "ideal_temp": "25–35°C",
        "soil_type": "Sandy loam",
        "growing_season": "Summer",
        "profit_tip": "Short-duration; good for dry regions."
    },
    "mothbeans": {
        "ideal_temp": "25–35°C",
        "soil_type": "Light soils",
        "growing_season": "Kharif",
        "profit_tip": "Drought-tolerant legume; grows well in arid zones."
    },
    "pigeonpeas": {
        "ideal_temp": "20–30°C",
        "soil_type": "Loamy, well-drained",
        "growing_season": "Kharif",
        "profit_tip": "Deep-rooted; improves soil structure."
    },
    "kidneybeans": {
        "ideal_temp": "18–25°C",
        "soil_type": "Well-drained sandy loam",
        "growing_season": "Rabi",
        "profit_tip": "Rich in protein; suited for cooler regions."
    },
    "chickpea": {
        "ideal_temp": "21–26°C",
        "soil_type": "Sandy to clay loam",
        "growing_season": "Rabi",
        "profit_tip": "Popular pulse crop; good export demand."
    },
    "coffee": {
        "ideal_temp": "15–28°C",
        "soil_type": "Red soil, rich in organic matter",
        "growing_season": "Hilly areas (perennial)",
        "profit_tip": "Requires shade and humid climate; long-term profit."
    }
}

# Load model
model = joblib.load('./model/crop_model.pkl')

# App Title
st.title("🌾 Crop Recommendation System")

# Input fields
N = st.number_input("Nitrogen (N)", 0, 140)
P = st.number_input("Phosphorus (P)", 5, 145)
K = st.number_input("Potassium (K)", 5, 205)
temperature = st.number_input("Temperature (°C)", 10.0, 45.0)
humidity = st.number_input("Humidity (%)", 10.0, 100.0)
ph = st.number_input("Soil pH", 3.5, 9.5)
rainfall = st.number_input("Rainfall (mm)", 20.0, 300.0)

# Predict button
if st.button("Predict Best Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    st.success(f"✅ Recommended Crop: **{prediction[0]}**")

    # 🌿 Show Crop Info Card
    crop = prediction[0]

    if crop in crop_info:
        st.markdown("### 🌿 Crop Information")
        st.write(f"**🌡️ Ideal Temperature:** {crop_info[crop]['ideal_temp']}")
        st.write(f"**🌱 Suitable Soil:** {crop_info[crop]['soil_type']}")
        st.write(f"**📅 Growing Season:** {crop_info[crop]['growing_season']}")
        st.info(f"💡 Tip: {crop_info[crop]['profit_tip']}")
    else:
        st.warning("⚠️ Crop info not available.")
    
