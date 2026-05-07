# ==========================================================
# CUSTOMER SEGMENTATION – BUSINESS-READY STREAMLIT APP
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gower

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Studio",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Segmentation App")
st.caption("From raw customers → actionable marketing decisions")

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_excel("marketing_campaign.xlsx")

df_raw = load_data()

# ----------------------------------------------------------
# FEATURE ENGINEERING
# ----------------------------------------------------------
current_year = pd.Timestamp.now().year
df_raw["Age"] = current_year - df_raw["Year_Birth"]
df_raw["TotalKids"] = df_raw["Kidhome"] + df_raw["Teenhome"]

features = [
    "Age", "Income", "TotalKids", "Recency",
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts",
    "NumWebPurchases", "NumStorePurchases",
    "Education", "Marital_Status"
]

df = df_raw[features].dropna().reset_index(drop=True)

# ----------------------------------------------------------
# HIERARCHICAL CLUSTERING (GOWER)
# ----------------------------------------------------------
gower_dist = gower.gower_matrix(df)
Z = linkage(squareform(gower_dist), method="average")

N_CLUSTERS = 4
df["Cluster"] = fcluster(Z, t=N_CLUSTERS, criterion="maxclust")

# ----------------------------------------------------------
# BUSINESS CLUSTER DEFINITIONS
# ----------------------------------------------------------
cluster_profiles = {
    1: {
        "name": "Conservatives",
        "description": "Price-sensitive families with low engagement and limited discretionary spend.",
        "strategy": {
            "Channel": "Email / SMS",
            "Offer": "Essential discounts",
            "Tone": "Trust-focused, practical",
            "Frequency": "Low"
        }
    },
    2: {
        "name": "Responders",
        "description": "Digitally active customers who respond well to targeted campaigns.",
        "strategy": {
            "Channel": "Email + Web",
            "Offer": "Personalised recommendations",
            "Tone": "Relevant & informative",
            "Frequency": "Medium"
        }
    },
    3: {
        "name": "Affluents",
        "description": "High-income premium buyers, self-directed and non-discount driven.",
        "strategy": {
            "Channel": "Direct / Premium Web",
            "Offer": "Exclusive experiences",
            "Tone": "Premium & aspirational",
            "Frequency": "Low"
        }
    },
    4: {
        "name": "Loyalists",
        "description": "Previously engaged high-value customers with reactivation potential.",
        "strategy": {
            "Channel": "Multi-channel",
            "Offer": "Loyalty rewards",
            "Tone": "Appreciative & personal",
            "Frequency": "Medium-High"
        }
    }
}

# ----------------------------------------------------------
# SIDEBAR — NEW CUSTOMER INPUT
# ----------------------------------------------------------
st.sidebar.header("🧍 New Customer Details")

user_input = {
    "Age": st.sidebar.number_input("Age", 18, 100, 40),
    "Income": st.sidebar.number_input("Income", 0, 200000, 60000),
    "TotalKids": st.sidebar.number_input("Total Kids", 0, 5, 1),
    "Recency": st.sidebar.number_input("Days Since Last Purchase", 0, 365, 30),
    "MntWines": st.sidebar.number_input("Wine Spend", 0, 2000, 300),
    "MntFruits": st.sidebar.number_input("Fruit Spend", 0, 1000, 50),
    "MntMeatProducts": st.sidebar.number_input("Meat Spend", 0, 3000, 400),
    "MntFishProducts": st.sidebar.number_input("Fish Spend", 0, 1000, 80),
    "MntSweetProducts": st.sidebar.number_input("Sweet Spend", 0, 1000, 60),
    "NumWebPurchases": st.sidebar.number_input("Web Purchases", 0, 50, 6),
    "NumStorePurchases": st.sidebar.number_input("Store Purchases", 0, 50, 5),
    "Education": st.sidebar.selectbox("Education", df["Education"].unique()),
    "Marital_Status": st.sidebar.selectbox("Marital Status", df["Marital_Status"].unique())
}

new_customer = pd.DataFrame([user_input])

# ----------------------------------------------------------
# ASSIGN CLUSTER (NEAREST GOWER)
# ----------------------------------------------------------
combined = pd.concat([df.drop(columns=["Cluster"]), new_customer], ignore_index=True)
combined_gower = gower.gower_matrix(combined)

distances = combined_gower[-1, :-1]
nearest_idx = np.argmin(distances)
assigned_cluster = df.loc[nearest_idx, "Cluster"]
confidence = 1 - distances[nearest_idx]

# ----------------------------------------------------------
# MAIN RESULTS
# ----------------------------------------------------------
st.success(
    f"✅ Customer assigned to **Cluster {assigned_cluster} — "
    f"{cluster_profiles[assigned_cluster]['name']}**"
)

# ----------------------------------------------------------
# TABS
# ----------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 Cluster Insight",
    "🎯 Marketing Actions",
    "📊 Customer vs Cluster",
    "📈 Overall Clusters"
])

# ----------------------------------------------------------
# TAB 1 — CLUSTER INSIGHT
# ----------------------------------------------------------
with tab1:
    profile = cluster_profiles[assigned_cluster]

    st.subheader(profile["name"])
    st.write(profile["description"])

    st.metric("Similarity Confidence", f"{confidence*100:.1f}%")

# ----------------------------------------------------------
# TAB 2 — MARKETING ACTIONS
# ----------------------------------------------------------
with tab2:
    st.subheader("Recommended Marketing Strategy")

    strategy = profile["strategy"]

    for k, v in strategy.items():
        st.write(f"**{k}:** {v}")

    st.info("📌 Next Best Action: Launch a personalised campaign within 7 days.")

# ----------------------------------------------------------
# TAB 3 — CUSTOMER VS CLUSTER
# ----------------------------------------------------------
with tab3:
    st.subheader("Customer vs Cluster Average")

    numeric_cols = [
        "Age","Income","TotalKids","Recency",
        "MntWines","MntFruits","MntMeatProducts",
        "MntFishProducts","MntSweetProducts",
        "NumWebPurchases","NumStorePurchases"
    ]

    cluster_avg = df[df["Cluster"] == assigned_cluster][numeric_cols].mean()

    compare_df = pd.DataFrame({
        "Customer": new_customer[numeric_cols].iloc[0],
        "Cluster Avg": cluster_avg
    })

    st.dataframe(compare_df.round(2))

# ----------------------------------------------------------
# TAB 4 — CLUSTER DISTRIBUTION & DENDROGRAM
# ----------------------------------------------------------
with tab4:
    st.subheader("Cluster Distribution")

    cluster_counts = df["Cluster"].value_counts().sort_index()

    fig, ax = plt.subplots()
    ax.bar(cluster_counts.index, cluster_counts.values)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Customers")
    ax.set_title("Customer Distribution by Cluster")
    st.pyplot(fig)

    st.subheader("Hierarchical Clustering Dendrogram")

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    dendrogram(Z, truncate_mode="level", p=5, ax=ax2)
    ax2.set_ylabel("Gower Distance")
    st.pyplot(fig2)
