# 📊 Customer Segmentation GUI

An interactive web application for customer segmentation using **RFM Analysis** and **K-Means Clustering**, built with **Python** and **Streamlit**.

## 🚀 Project Overview

This project provides an interactive GUI that allows users to upload their own customer transaction dataset and perform customer segmentation without modifying the source code.

The application combines data exploration, RFM analysis, K-Means clustering, and interactive visualizations in one dashboard.

## ✨ Features

* 📂 Upload CSV or Excel datasets
* 🔍 Explore and preview uploaded data
* 📊 Data summary and statistics
* 🧹 Basic data inspection
* 📈 Multiple interactive visualizations
* 👥 RFM Analysis
* 🤖 K-Means Clustering
* 📉 Elbow Method
* 📐 Silhouette Score
* 🎯 PCA visualization
* 👤 Customer Explorer
* 📋 Cluster and segment summaries
* 💾 Export segmented customer data

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Plotly
* Streamlit
* OpenPyXL

## 📊 Methodology

### 1. RFM Analysis

Customers are analyzed based on:

* **Recency** – How recently the customer made a purchase
* **Frequency** – How frequently the customer purchased
* **Monetary** – How much the customer spent

### 2. K-Means Clustering

K-Means is applied to the RFM features to group customers into different segments based on their purchasing behavior.

### 3. Visualization

The dashboard provides several visualization options to help understand customer behavior and cluster characteristics.

## ▶️ How to Run

Clone the repository:

```bash
git clone YOUR_REPOSITORY_LINK
```

Navigate to the project folder:

```bash
cd customer-segmentation-gui
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 📂 Dataset

The application does not depend on a fixed dataset.

Users can upload their own CSV or Excel customer transaction dataset directly through the interface.

## 🎯 Project Goal

The goal of this project is to provide an easy-to-use interactive interface for exploring customer behavior and discovering meaningful customer segments using data analysis and machine learning.

## 👩‍💻 Author

**Rozalenda Taher**

Data Science Student | AI & Data Analysis
