import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS if available
CSS_PATH = os.path.join("static", "style.css")
if os.path.exists(CSS_PATH):
  with open(CSS_PATH, "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Dataset file check
DATASET_PATH = "maharashtra_crop_yield_500_records.csv"
if not os.path.exists(DATASET_PATH):
  st.error(
      f"Dataset file '{DATASET_PATH}' not found in your VS Code folder. Please"
      " make sure the CSV file is in the project folder."
  )
  st.stop()


# Helper function to auto-train model if .pkl is missing
@st.cache_resource
def load_or_train_model():
  MODEL_PATH = "crop_yield_model.pkl"
  DATA_PATH = "crop_yield_data.csv"

  df = pd.read_csv(DATASET_PATH)
  df.columns = df.columns.str.strip()
  df = df.drop_duplicates().reset_index(drop=True)

  target = "Yield_ton_per_hectare"
  features = [
      "State",
      "District",
      "Crop",
      "Season",
      "Area_hectares",
      "Rainfall_mm",
      "Temperature_C",
      "Soil_Moisture_%",
      "Fertilizer_kg",
      "Pesticide_kg",
  ]

  df = df.dropna(subset=[target]).copy()

  if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
      model = pickle.load(f)
  else:
    # Train model automatically if not present
    X = df[features].copy()
    y = df[target].copy()

    cat_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("numerical", "passthrough", num_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200, random_state=42, n_jobs=-1
                ),
            ),
        ]
    )
    model.fit(X, y)

    # Save for future runs
    with open(MODEL_PATH, "wb") as f:
      pickle.dump(model, f)
    df.to_csv(DATA_PATH, index=False)

  return model, df, features, target


model, df, features, target = load_or_train_model()

# Sidebar Navigation
st.sidebar.title("🌾 Crop Yield")
st.sidebar.markdown("### Prediction System")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Predict Yield", "Dataset Overview", "Insights & Graphs"],
)
st.sidebar.markdown("---")
st.sidebar.info(
    "This system uses historical agricultural data and a machine learning"
    " model to estimate crop yield."
)

if page == "Home":
  st.title("🌱 Crop Yield Prediction")
  st.write(
      "A machine learning based system that uses historical agricultural data"
      " to estimate crop yield and understand important farming patterns."
  )
  st.markdown("---")

  st.subheader("Explore the System")
  col1, col2, col3 = st.columns(3)
  with col1:
    st.markdown("### Predict Yield")
    st.write(
        "Enter crop, weather, soil and farming information to estimate the"
        " expected crop yield."
    )
  with col2:
    st.markdown("### Explore Dataset")
    st.write(
        "View the agricultural dataset, available crops, districts and seasons."
    )
  with col3:
    st.markdown("### Understand Patterns")
    st.write(
        "Explore graphs showing relationships between yield, rainfall,"
        " temperature, soil moisture and crops."
    )

  st.markdown("---")
  st.subheader("Project Overview")
  st.write(
      "The project uses historical crop and agricultural data from"
      " Maharashtra. A Random Forest machine learning model is used to predict"
      " crop yield based on factors such as rainfall, temperature, soil"
      " moisture, fertilizer, pesticide and cultivated area."
  )

  st.markdown("---")
  st.subheader("Dataset Summary")
  c1, c2, c3, c4 = st.columns(4)
  with c1:
    st.metric("Total Records", len(df))
  with c2:
    st.metric("Crops", df["Crop"].nunique())
  with c3:
    st.metric("Districts", df["District"].nunique())
  with c4:
    st.metric("Seasons", df["Season"].nunique())

  st.markdown("---")
  st.subheader("Crops Available")
  crops = sorted(df["Crop"].dropna().unique())
  st.write(" • ".join(crops))

elif page == "Predict Yield":
  st.title("🎯 Crop Yield Prediction")
  st.write(
      "Enter the agricultural and environmental information below to estimate"
      " crop yield."
  )
  st.markdown("---")

  col1, col2 = st.columns(2)
  with col1:
    st.subheader("🌾 Crop Information")
    state = st.selectbox("State", sorted(df["State"].dropna().unique()))
    district = st.selectbox(
        "District", sorted(df["District"].dropna().unique())
    )
    crop = st.selectbox("Crop", sorted(df["Crop"].dropna().unique()))
    season = st.selectbox("Season", sorted(df["Season"].dropna().unique()))
    area = st.number_input(
        "Area (hectares)",
        min_value=0.0,
        value=float(df["Area_hectares"].median()),
        step=0.1,
    )

  with col2:
    st.subheader("🌧️ Environmental Information")
    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        value=float(df["Rainfall_mm"].median()),
        step=1.0,
    )
    temperature = st.number_input(
        "Temperature (°C)",
        value=float(df["Temperature_C"].median()),
        step=0.1,
    )
    soil_moisture = st.number_input(
        "Soil Moisture (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(df["Soil_Moisture_%"].median()),
        step=0.1,
    )
    fertilizer = st.number_input(
        "Fertilizer (kg)",
        min_value=0.0,
        value=float(df["Fertilizer_kg"].median()),
        step=1.0,
    )
    pesticide = st.number_input(
        "Pesticide (kg)",
        min_value=0.0,
        value=float(df["Pesticide_kg"].median()),
        step=1.0,
    )

  st.markdown("---")
  predict_button = st.button("Predict Crop Yield", use_container_width=True)

  if predict_button:
    input_data = pd.DataFrame({
        "State": [state],
        "District": [district],
        "Crop": [crop],
        "Season": [season],
        "Area_hectares": [area],
        "Rainfall_mm": [rainfall],
        "Temperature_C": [temperature],
        "Soil_Moisture_%": [soil_moisture],
        "Fertilizer_kg": [fertilizer],
        "Pesticide_kg": [pesticide],
    })[features]

    try:
      prediction = model.predict(input_data)[0]
      st.markdown("---")
      st.subheader("📊 Prediction Results")
      result_col1, result_col2 = st.columns(2)

      with result_col1:
        st.metric("Estimated Crop Yield", f"{prediction:.2f} ton/hectare")

      avg_rainfall = df["Rainfall_mm"].mean()
      avg_temperature = df["Temperature_C"].mean()
      avg_soil = df["Soil_Moisture_%"].mean()

      risk_score = 0
      if abs(rainfall - avg_rainfall) > df["Rainfall_mm"].std():
        risk_score += 1
      if abs(temperature - avg_temperature) > df["Temperature_C"].std():
        risk_score += 1
      if abs(soil_moisture - avg_soil) > df["Soil_Moisture_%"].std():
        risk_score += 1

      risk = (
          "Low" if risk_score == 0 else ("Moderate" if risk_score == 1 else "High")
      )

      with result_col2:
        st.metric("Weather Risk Level", risk)

      st.success(
          f"Estimated yield for **{crop}** is **{prediction:.2f}"
          " ton/hectare**."
      )
    except Exception as e:
      st.error(f"Prediction error: {e}")

elif page == "Dataset Overview":
  st.title("📋 Dataset Overview")
  st.write(
      "Explore the historical agricultural data used by the crop yield"
      " prediction system."
  )
  st.markdown("---")

  c1, c2, c3, c4 = st.columns(4)
  with c1:
    st.metric("Records", len(df))
  with c2:
    st.metric("Columns", len(df.columns))
  with c3:
    st.metric("Districts", df["District"].nunique())
  with c4:
    st.metric("Crops", df["Crop"].nunique())

  st.markdown("---")
  st.dataframe(df, use_container_width=True)

elif page == "Insights & Graphs":
  st.title("📈 Insights & Graphs")
  st.write("Visual analysis of crop yield and important agricultural factors.")
  st.markdown("---")

  st.subheader("Average Yield by Crop")
  crop_yield = df.groupby("Crop")[target].mean().sort_values(ascending=False)
  fig1, ax1 = plt.subplots(figsize=(10, 5))
  crop_yield.plot(kind="bar", ax=ax1, color="#246B3D")
  ax1.set_xlabel("Crop")
  ax1.set_ylabel("Average Yield (ton/hectare)")
  ax1.set_title("Average Crop Yield")
  plt.xticks(rotation=45)
  plt.tight_layout()
  st.pyplot(fig1)

  st.subheader("Average Yield by Season")
  season_yield = df.groupby("Season")[target].mean().sort_values(ascending=False)
  fig2, ax2 = plt.subplots(figsize=(8, 5))
  season_yield.plot(kind="bar", ax=ax2, color="#174D2B")
  ax2.set_xlabel("Season")
  ax2.set_ylabel("Average Yield (ton/hectare)")
  ax2.set_title("Average Yield by Season")
  plt.tight_layout()
  st.pyplot(fig2)

  st.subheader("Rainfall vs Crop Yield")
  fig3, ax3 = plt.subplots(figsize=(9, 5))
  ax3.scatter(df["Rainfall_mm"], df[target], alpha=0.6, color="#245C3A")
  ax3.set_xlabel("Rainfall (mm)")
  ax3.set_ylabel("Yield (ton/hectare)")
  ax3.set_title("Rainfall vs Crop Yield")
  plt.tight_layout()
  st.pyplot(fig3)

  st.subheader("Temperature vs Crop Yield")
  fig4, ax4 = plt.subplots(figsize=(9, 5))
  ax4.scatter(df["Temperature_C"], df[target], alpha=0.6, color="#174D2B")
  ax4.set_xlabel("Temperature (°C)")
  ax4.set_ylabel("Yield (ton/hectare)")
  ax4.set_title("Temperature vs Crop Yield")
  plt.tight_layout()
  st.pyplot(fig4)

  st.subheader("Soil Moisture vs Crop Yield")
  fig5, ax5 = plt.subplots(figsize=(9, 5))
  ax5.scatter(df["Soil_Moisture_%"], df[target], alpha=0.6, color="#246B3D")
  ax5.set_xlabel("Soil Moisture (%)")
  ax5.set_ylabel("Yield (ton/hectare)")
  ax5.set_title("Soil Moisture vs Crop Yield")
  plt.tight_layout()
  st.pyplot(fig5)

  st.subheader("Average Yield by District")
  district_yield = (
      df.groupby("District")[target].mean().sort_values(ascending=False)
  )
  fig6, ax6 = plt.subplots(figsize=(11, 6))
  district_yield.plot(kind="bar", ax=ax6, color="#245C3A")
  ax6.set_xlabel("District")
  ax6.set_ylabel("Average Yield (ton/hectare)")
  ax6.set_title("District-wise Average Crop Yield")
  plt.xticks(rotation=75)
  plt.tight_layout()
  st.pyplot(fig6)

  st.markdown("---")
  st.subheader("Key Insights")
  st.write(
      f"• **{crop_yield.idxmax()}** has the highest average yield among the crops"
      " in the dataset."
  )
  st.write(
      f"• **{crop_yield.idxmin()}** has the lowest average yield among the crops"
      " in the dataset."
  )
  st.write(
      f"• **{season_yield.idxmax()}** has the highest average yield among the"
      " available seasons."
  )
  st.write(
      "• The scatter plots help visualize how rainfall, temperature, and soil"
      " moisture relate to crop yield."
  )