import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Koku Gözlemleri Kümeleme", layout="centered")

st.title("🌬️ Koku Gözlemleri Kümeleme Uygulaması")
st.markdown("""
Bu uygulama, yüklediğiniz Excel dosyasındaki **koku gözlemlerini** kullanarak 
**K-Means algoritması** ile kümelendirir ve harita üzerinde gösterir.
""")

# Dosya yükleyici
uploaded_file = st.file_uploader("📁 Excel dosyasını yükleyin (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Excel dosyasını oku
        df = pd.read_excel(uploaded_file, sheet_name=0, skiprows=1)

        # Gerekli sütunları seç ve sayısal değerlere çevir
        df = df[['Latitude', 'Longitude', 'Odour Intensity', 'Hedonic Tone']]
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()

        # Enlem ve boylam filtreleme
        df = df[(df['Latitude'] >= 40) & (df['Latitude'] <= 42) &
                (df['Longitude'] >= 26) & (df['Longitude'] <= 29)]

        st.success("✅ Veri başarıyla yüklendi ve filtrelendi.")
        st.write("📄 İlk 5 satır:", df.head())

        # Küme sayısı seçimi
        n_clusters = st.slider("🔢 Küme Sayısını Seçin", min_value=2, max_value=10, value=3)

        # Ölçeklendirme
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)

        # KMeans uygulaması
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Küme'] = kmeans.fit_predict(scaled_data)

        # Sabit renkler listesi
        renkler = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'brown', 'pink', 'gray', 'olive']

        # Harita çizimi
        st.subheader("🗺️ Kümeleme Haritası")
        fig, ax = plt.subplots(figsize=(10, 6))

        for cluster in df['Küme'].unique():
            cluster_data = df[df['Küme'] == cluster]
            ax.scatter(cluster_data['Longitude'], cluster_data['Latitude'],
                       label=f'Küme {cluster}', color=renkler[cluster], s=60)

        ax.set_xlabel("Boylam")
        ax.set_ylabel("Enlem")
        ax.set_title("Koku Gözlemleri Kümelendirme Haritası (KMeans)")
        ax.grid(True)
        ax.legend(title="Kümeler")

        st.pyplot(fig)

        # Küme özet tablosu
        st.subheader("📋 Kümeleme Sonuçları")
        st.dataframe(df)

        st.subheader("📊 Küme Özeti (Ortalama Değerler)")
        st.dataframe(df.groupby("Küme")[['Odour Intensity', 'Hedonic Tone']].mean().round(2))

    except Exception as e:
        st.error(f"❌ Bir hata oluştu: {e}")
else:
    st.info("📎 Lütfen bir Excel dosyası yükleyin.")
