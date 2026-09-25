import streamlit as st
import pandas as pd
import numpy as np
import math
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# =========================================================
# 1. Sayfa Konfigürasyonu ve CSS Tasarımı
# =========================================================
st.set_page_config(page_title="İstanbul Airbnb Fiyat Tahmini", layout="wide", page_icon="🏠")

st.markdown("""
    <style>
    .main-title { font-family: 'Poppins', sans-serif; color: #FF5A5F; font-weight: bold; text-align: center; margin-bottom: 30px; }
    .stButton>button { background-color: #FF5A5F !important; color: white !important; width: 100%; border-radius: 8px; font-weight: bold; height: 50px; font-size: 18px; border: none; }
    .stButton>button:hover { background-color: #E64A4E !important; box-shadow: 0 4px 15px rgba(255,90,95,0.4); }
    .custom-card { background-color: #fdfdfd; padding: 20px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.05); border-top: 4px solid #ff5a5f; margin-bottom: 20px; }
    .result-card { background-color: #f4fbf7; padding: 25px; border-radius: 12px; border: 1px solid #c2eecf; text-align: center; margin-top: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.02); }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# 2. Çoklu Dil (Localization) Sözlüğü
# =========================================================
lang = st.sidebar.radio("🌐 Language / Dil Seçimi", ["Türkçe", "English"], horizontal=True)

TEXT = {
    "Türkçe": {
        "title": "🏠 İstanbul Airbnb Gecelik Fiyat Tahmin Motoru",
        "profile_lbl": "👤 Profiliniz Nedir? / Kimsiniz?",
        "profile_guest": "✈️ Müşteriyim (Gezgin)",
        "profile_host": "💼 Ev Sahibi / Yatırımcıyım",
        "sidebar_header": "📋 İlan Özellikleri",
        "loc": "Konum (İlçe)",
        "room_type": "Oda / Ev Tipi",
        "min_nights": "Minimum Konaklama Gecesi",
        "month_lbl": "📅 Konaklama Planlanan Ay",
        "reviews": "Toplam Yorum Sürümü",
        "reviews_month": "Aylık Ortalama Yorum Sayısı",
        "host_count": "Ev Sahibinin Diğer İlan Sayısı",
        "loc_analysis": "🔍 Konum Analizi",
        "region_info": "📍 Bölge Bilgisi",
        "region_desc": "<b>{ilce}</b> ilçesi, İstanbul'un <b>{yaka} Yakası</b> sınırları içerisindedir.",
        "bosphorus": "🌊 İlçe Merkezinin Boğaz Hattına Uzaklığı",
        "poi_title": "🚗 Önemli Noktalara Ulaşım ve Maliyet Analizi",
        "ai_title": "💰 Yapay Zeka Fiyat Tahmini",
        "ai_desc": "Kriterlere göre makine öğrenmesi modelini çalıştırın:",
        "calc_btn": "🚀 Gecelik Fiyatı Hesapla",
        "result_title": "Önerilen Gecelik Kira Bedeli",
        "note": "⚠️ Tahmin, geçmiş veri trendlerinin makine öğrenmesi ile simüle edilmesiyle hesaplanmıştır.",
        "map_title": "📍 Seçilen İlçenin Harita Üzerindeki Konumu",
        "yaka_avrupa": "Avrupa",
        "yaka_anadolu": "Anadolu",
        "b2b_header": "📊 Yatırımcı & Ev Sahibi İçin Pazar Analizi (Market Intelligence)",
        "b2b_m0": "Toplam Aktif İlan Sayısı (Arz)",
        "b2b_m_share": "Kategori Pazar Payı",
        "b2b_m1": "Seçilen Kategori Ortalama Fiyatı",
        "b2b_m3": "Kategorideki Profesyonel Yönetim Oranı (Acente)",
        "b2b_bench": "🏆 Bölgedeki En Popüler 5 Rakip İlan",
        "b2c_header": "🎁 Gezginler İçin Fırsat Evler (Gizli Cevherler)",
        "b2c_info": "💡 Aşağıdaki 5 ev {ilce} ilçesindeki kategori ortalamasından daha ucuz olmasına rağmen daha fazla yorum almıştır.",
        "table_name": "İlan Adı", "table_type": "Oda Tipi", "table_price": "Fiyat (TL)",
        "table_reviews": "Toplam Yorum", "table_avail": "Müsaitlik (Gün)",
        "travel_alert": "⚠️ Zaman & Maliyet Uyarısı: {ilce} konumu turistik merkezlere biraz uzaktır. Günlük yaklaşık {süre} dakika gidiş-dönüş trafiği ve ortalama {taksi} TL taksi maliyetini göz önünde bulundurmalısınız.",
        "taxi_text": "{tl} TL",
        "budget_title": "🎒 Akıllı Bütçe & Seyahat Planlayıcı",
        "budget_lbl": "Toplam Bütçeniz",
        "budget_currency": "Para Birimi",
        "days_lbl": "Kalmak İstediğiniz Gece Sayısı",
        "plan_btn": "🗺️ Rotamı ve Bütçemi Optimize Et",
        "budget_report": "📋 Kişiselleştirilmiş Bütçe Raporu",
        "b_rep1": "Mevcut bütçenizle ({tl_bütçe:,.0f} TL) seçtiğiniz {ilce} ilçesinde en fazla {kalan} gece kalabilirsiniz. Hedefiniz: {hedef} gece.",
        "b_upsell": " Hedeflediğiniz {hedef} gece boyunca burada kalabilmek için bütçenizin üzerine ortalama {ek_tutar:,.1f} {birim} daha eklemelisiniz.",
        "b_rep2": "💡 Daha Uzun Kalabileceğiniz Alternatif Ekonomik İlçeler: ",
        "b_rep3": "🌟 Bütçenizin Yettiği Popüler Canlı İlçeler: ",
        "col_poi": "Önemli Nokta / Turistik Merkez", "col_dist": "Mesafe (km)", "col_time": "Süre (dk)",
        "col_taxi": "Tahmini Taksi Ücreti"
    },
    "English": {
        "title": "🏠 Istanbul Airbnb Nightly Price Predictor",
        "profile_lbl": "👤 Select Your Profile / Who are you?",
        "profile_guest": "✈️ I am a Guest (Traveler)",
        "profile_host": "💼 I am a Host / Investor",
        "sidebar_header": "📋 Listing Features",
        "loc": "Location (District)",
        "room_type": "Room / Property Type",
        "min_nights": "Minimum Nights",
        "month_lbl": "📅 Planned Month of Stay",
        "reviews": "Total Reviews",
        "reviews_month": "Average Reviews per Month",
        "host_count": "Host's Total Listings",
        "loc_analysis": "🔍 Location Analysis",
        "region_info": "📍 Region Info",
        "region_desc": "<b>{ilce}</b> district is located on the <b>{yaka} Side</b> of Istanbul.",
        "bosphorus": "🌊 Distance to Bosphorus Line",
        "poi_title": "🚗 Logistics & Commute Cost Analysis for Travelers",
        "ai_title": "💰 AI Price Prediction",
        "ai_desc": "Run the machine learning model based on criteria:",
        "calc_btn": "🚀 Calculate Nightly Price",
        "result_title": "Recommended Nightly Rent",
        "note": "⚠️ Prediction calculated by simulating past Airbnb data trends with machine learning.",
        "map_title": "📍 Location of the Selected District on the Map",
        "yaka_avrupa": "European",
        "yaka_anadolu": "Anatolian",
        "b2b_header": "📊 Market Analysis & Market Intelligence for Investors",
        "b2b_m0": "Total Active Listings (Supply)",
        "b2b_m_share": "Category Market Share",
        "b2b_m1": "Selected Category Average Price",
        "b2b_m3": "Professional Management Rate (Agency) in Category",
        "b2b_bench": "🏆 Top 5 Most Popular Competitor Listings",
        "b2c_header": "🎁 Hidden Gems for Travelers (Best Value Houses)",
        "b2c_info": "💡 The following 5 listings are cheaper than the category average price in {ilce}, yet they received more reviews.",
        "table_name": "Listing Name", "table_type": "Room Type", "table_price": "Price (TL)",
        "table_reviews": "Total Reviews", "table_avail": "Availability (Days)",
        "travel_alert": "⚠️ Time & Cost Alert: {ilce} is relatively far from tourist hotspots. You should consider a daily round-trip traffic of around {süre} minutes and an average taxi cost of {taksi} TL.",
        "taxi_text": "{tl} TL",
        "budget_title": "🎒 Smart Budget & Trip Planner",
        "budget_lbl": "Your Total Budget",
        "budget_currency": "Currency",
        "days_lbl": "Number of Nights You Wish to Stay",
        "plan_btn": "🗺️ Optimize My Trip & Budget",
        "budget_report": "📋 Personalized Budget Report",
        "b_rep1": "With your current budget (approx. {tl_bütçe:,.0f} TL), you can stay a maximum of {kalan} nights in {ilce}. Your target: {hedef} nights.",
        "b_upsell": " To complete your targeted {hedef} nights, you need to add an average of {ek_tutar:,.1f} {birim} to your budget.",
        "b_rep2": "💡 Alternative Economical Districts for Longer Stay: ",
        "b_rep3": "🌟 Lively & Popular Alternative Districts: ",
        "col_poi": "Tourist Attraction / Hotspot", "col_dist": "Distance (km)", "col_time": "Duration (min)",
        "col_taxi": "Est. Taxi Fare"
    }
}

t = TEXT[lang]

# --- PROFİL SEÇİCİ ---
secilen_profil = st.sidebar.radio(t['profile_lbl'], [t['profile_guest'], t['profile_host']])

# =========================================================
# 3. Global Sabitler ve Yardımcı Fonksiyonlar
# =========================================================
BOGAZ_NOKTALARI = [(41.005, 28.995), (41.045, 29.035), (41.085, 29.065), (41.130, 29.080)]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


@st.cache_data(show_spinner=False)
def karayolu_mesafesi_osrm(lat1, lon1, lat2, lon2):
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
        response = requests.get(url, timeout=2).json()
        if response.get("code") == "Ok":
            return response["routes"][0]["distance"] / 1000
    except Exception:
        pass
    return haversine_km(lat1, lon1, lat2, lon2)


@st.cache_data(ttl=3600)
def guncel_doviz_kurlarini_getir():
    try:
        url = "https://open.er-api.com/v6/latest/TRY"
        response = requests.get(url, timeout=2).json()
        if response.get("result") == "success":
            rates = response.get("rates", {})
            return {
                "TRY": 1.0, "USD": rates.get("USD", 1 / 32.5), "EUR": rates.get("EUR", 1 / 35.0),
                "GBP": rates.get("GBP", 1 / 41.5), "JPY": rates.get("JPY", 1 / 0.21)
            }
    except Exception:
        pass
    return {"TRY": 1.0, "USD": 1 / 32.5, "EUR": 1 / 35.0, "GBP": 1 / 41.5, "JPY": 1 / 0.21}


# =========================================================
# 4. Veri Yükleme ve Model Eğitimi
# =========================================================
@st.cache_data
def veriyi_ve_modeli_hazirla():
    df = pd.read_csv(r"listings.csv")
    df['neighbourhood'] = df['neighbourhood'].str.replace('ı', 'i').str.replace('ğ', 'g').str.replace('ü',
                                                                                                      'u').str.replace(
        'ş', 's').str.replace('ö', 'o').str.replace('Ç', 'C').str.replace('İ', 'I').str.strip()
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    df = df.dropna(subset=['price'])
    df = df[(df["price"] > 300) & (df["price"] < 8000)]

    yaka_mapping = {
        'Adalar': 'Anadolu', 'Atasehir': 'Anadolu', 'Beykoz': 'Anadolu', 'Cekmekoy': 'Anadolu',
        'Kadikoy': 'Anadolu', 'Kartal': 'Anadolu', 'Maltepe': 'Anadolu', 'Pendik': 'Anadolu',
        'Sancaktepe': 'Anadolu', 'Sultanbeyli': 'Anadolu', 'Sile': 'Anadolu', 'Tuzla': 'Anadolu',
        'Umraniye': 'Anadolu', 'Uskudar': 'Anadolu', 'Arnavutkoy': 'Avrupa', 'Avcilar': 'Avrupa',
        'Bagcilar': 'Avrupa', 'Bahcelievler': 'Avrupa', 'Bakirkoy': 'Avrupa', 'Basaksehir': 'Avrupa',
        'Bayrampasa': 'Avrupa', 'Besiktas': 'Avrupa', 'Beylikduzu': 'Avrupa', 'Beyoglu': 'Avrupa',
        'Buyukcekmece': 'Avrupa', 'Catalca': 'Avrupa', 'Esenler': 'Avrupa', 'Esenyurt': 'Avrupa',
        'Eyup': 'Avrupa', 'Fatih': 'Avrupa', 'Gaziosmanpasa': 'Avrupa', 'Gungoren': 'Avrupa',
        'Kagithane': 'Avrupa', 'Kucukcekmece': 'Avrupa', 'Sariyer': 'Avrupa', 'Silivri': 'Avrupa',
        'Sultangazi': 'Avrupa', 'Sisli': 'Avrupa', 'Zeytinburnu': 'Avrupa'
    }
    df['yaka'] = df['neighbourhood'].map(yaka_mapping).fillna('Avrupa')
    ilceler = sorted(list(df['neighbourhood'].unique()))
    oda_tipleri = sorted(list(df['room_type'].unique()))

    secilen_mesafeler = []
    for lat, lon in BOGAZ_NOKTALARI:
        mesafe = ((df['latitude'] - lat) ** 2 + (df['longitude'] - lon) ** 2) ** 0.5
        secilen_mesafeler.append(mesafe)
    df['uzaklik_bogaz'] = np.minimum.reduce(secilen_mesafeler)

    df_encoded = pd.get_dummies(df, columns=['room_type', 'neighbourhood', 'yaka'], drop_first=False)
    X = df_encoded.select_dtypes(include=['float64', 'int64', 'bool', 'uint8']).drop(
        columns=['price', 'id', 'host_id', 'number_of_reviews_ltm', 'last_review', 'name', 'host_name',
                 'neighbourhood_group', 'license', 'reviews_per_month'], errors='ignore')
    y = df_encoded['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42, n_estimators=60, n_jobs=-1)
    model.fit(X_train, y_train)

    return model, X.columns, ilceler, oda_tipleri, yaka_mapping, df


model, model_columns, ilceler, oda_tipleri, yaka_mapping, raw_df = veriyi_ve_modeli_hazirla()

# =========================================================
# 5. SOL PANEL GİRDİ ALANLARI
# =========================================================
st.sidebar.markdown("---")
st.sidebar.header(t['sidebar_header'])
secilen_ilce = st.sidebar.selectbox(t['loc'], ilceler, index=ilceler.index("Beyoglu") if "Beyoglu" in ilceler else 0)
secilen_oda = st.sidebar.selectbox(t['room_type'], oda_tipleri)

# --- MEVSİMSELLİK GİRDİ ALANI ---
aylar = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"] if lang == "Türkçe" else ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
secilen_ay = st.sidebar.selectbox(t['month_lbl'], aylar, index=6) # Temmuz varsayılan

# Katsayı Hesabı (Yaz: %25 Zam, Geçiş: Stabil, Kış/Erken Bahar: %15 İndirim)
mevsim_katsayisi = 1.25 if secilen_ay in aylar[5:8] else (1.0 if secilen_ay in [aylar[4], aylar[8], aylar[9]] else 0.85)

st.sidebar.markdown("---")
minimum_nights = st.sidebar.slider(t['min_nights'], 1, 30, 1)
availability_365 = int(raw_df['availability_365'].mean())

number_of_reviews = st.sidebar.number_input(t['reviews'], value=5, min_value=0, step=1)
reviews_per_month = st.sidebar.slider(t['reviews_month'], min_value=0.0, max_value=15.0, value=1.0, step=0.1)
host_listings_count = st.sidebar.number_input(t['host_count'], value=1, min_value=1, step=1)

# =========================================================
# 6. ARKA PLAN MATEMATİKSEL HESAPLAMALARI
# =========================================================
ilce_verileri = raw_df[raw_df['neighbourhood'] == secilen_ilce]
latitude = float(ilce_verileri['latitude'].mean())
longitude = float(ilce_verileri['longitude'].mean())

girdi_mesafeler = []
for lat, lon in BOGAZ_NOKTALARI:
    mesafe = ((latitude - lat) ** 2 + (longitude - lon) ** 2) ** 0.5
    girdi_mesafeler.append(mesafe)
uzaklik_bogaz = min(girdi_mesafeler)
belirlenen_yaka = yaka_mapping.get(secilen_ilce, 'Avrupa')

input_data = pd.DataFrame(0, index=[0], columns=model_columns)
input_data['latitude'] = latitude
input_data['longitude'] = longitude
input_data['minimum_nights'] = minimum_nights
input_data['number_of_reviews'] = number_of_reviews
input_data['calculated_host_listings_count'] = host_listings_count
input_data['availability_365'] = availability_365
input_data['uzaklik_bogaz'] = uzaklik_bogaz

if f"room_type_{secilen_oda}" in model_columns: input_data[f"room_type_{secilen_oda}"] = 1
if f"neighbourhood_{secilen_ilce}" in model_columns: input_data[f"neighbourhood_{secilen_ilce}"] = 1
if f"yaka_{belirlenen_yaka}" in model_columns: input_data[f"yaka_{belirlenen_yaka}"] = 1
input_data = input_data.astype(float)

# =========================================================
# 7. SAĞ PANEL ARAYÜZÜ (GÖRSELLEŞTİRME VE ANALİZ)
# =========================================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader(t['loc_analysis'])
    yaka_metni = t['yaka_avrupa'] if belirlenen_yaka == 'Avrupa' else t['yaka_anadolu']
    bolge_aciklamasi = f"{secilen_ilce} ilcesi, Istanbul'un {yaka_metni} Yakasi sinirlari icerisindedir."

    st.markdown(f"""
    <div class="custom-card">
         <h4>{t['region_info']}</h4>
         <p style='margin-bottom:10px;'>{bolge_aciklamasi}</p>
    </div>
    """, unsafe_allow_html=True)

    bogaz_mesafeleri_km = []
    for lat, lon in BOGAZ_NOKTALARI:
        mesafe_km = haversine_km(latitude, longitude, lat, lon)
        bogaz_mesafeleri_km.append(mesafe_km)
    gosterilecek_bogaz_km = min(bogaz_mesafeleri_km)
    st.metric(label=t['bosphorus'], value=f"{gosterilecek_bogaz_km:.2f} km")

    st.markdown("---")
    st.markdown(f"#### {t['poi_title']}")

    onemli_yerler = {
        "Ayasofya" if lang == "Türkçe" else "Hagia Sophia": (41.0086, 28.9802),
        "Topkapı Sarayı" if lang == "Türkçe" else "Topkapi Palace": (41.0115, 28.9834),
        "Yerebatan Sarnıcı" if lang == "Türkçe" else "Basilica Cistern": (41.0084, 28.9779),
        "Sultanahmet Camii" if lang == "Türkçe" else "Blue Mosque": (41.0054, 28.9768),
        "Kapalıçarşı" if lang == "Türkçe" else "Grand Bazaar": (41.0106, 28.9680),
        "Galata Kulesi" if lang == "Türkçe" else "Galata Tower": (41.0256, 28.9742),
        "Dolmabahçe Sarayı" if lang == "Türkçe" else "Dolmabahce Palace": (41.0390, 28.9983),
        "Taksim Meydanı" if lang == "Türkçe" else "Taksim Square": (41.0370, 28.9850)
    }

    poi_data = []
    toplam_osrm_km = 0
    for yer, (y_lat, y_lon) in onemli_yerler.items():
        mesafe = karayolu_mesafesi_osrm(latitude, longitude, y_lat, y_lon)
        toplam_osrm_km += mesafe
        tahmini_dakika = mesafe * 2

        hesaplanan_taksi = 65.40 + (mesafe * 43.56) + 30
        tahmini_taksi = max(210, hesaplanan_taksi)

        poi_data.append({
            t['col_poi']: yer,
            t['col_dist']: f"{mesafe:.1f} km",
            t['col_time']: f"{tahmini_dakika:.0f} dk",
            t['col_taxi']: f"{tahmini_taksi:.0f} TL"
        })

    st.dataframe(pd.DataFrame(poi_data), use_container_width=True, hide_index=True)

    ortalama_merkez_uzaklik = toplam_osrm_km / len(onemli_yerler)
    if ortalama_merkez_uzaklik > 15:
        toplam_gidis_donus_sure = ortalama_merkez_uzaklik * 4
        realistik_ortalama_taksi = max(210, 65.40 + (ortalama_merkez_uzaklik * 43.56) + 30)
        st.warning(t['travel_alert'].format(ilce=secilen_ilce, süre=f'{toplam_gidis_donus_sure:.0f}',
                                            taksi=f'{realistik_ortalama_taksi:.0f}'))

with col2:
    st.subheader(t['ai_title'])
    st.caption(t['ai_desc'])

    if st.button(t['calc_btn']):
        # --- TAHMİNE MEVSİMSELLİK KATSAYISI ETKİSİ ---
        ham_tahmin = model.predict(input_data)[0]
        tahmin_edilen_fiyat = ham_tahmin * mevsim_katsayisi

        kurlar = guncel_doviz_kurlarini_getir()
        fiyat_usd = tahmin_edilen_fiyat * kurlar["USD"]
        fiyat_eur = tahmin_edilen_fiyat * kurlar["EUR"]
        fiyat_gbp = tahmin_edilen_fiyat * kurlar["GBP"]
        fiyat_jpy = tahmin_edilen_fiyat * kurlar["JPY"]

        st.markdown(f"""
        <div class="result-card">
             <h3 style='color: #2e7d32; margin-bottom: 5px;'>{t['result_title']}</h3>
             <h1 style='color: #1b5e20; margin-top: 0px;'>{tahmin_edilen_fiyat:,.2f} TL</h1>
             <p style='color: #2e7d32; font-weight: bold; font-size: 16px; margin-top: 10px;'>
                💵 ${fiyat_usd:.1f} USD &nbsp;|&nbsp; 💶 {fiyat_eur:.1f} EUR &nbsp;|&nbsp; 💷 £{fiyat_gbp:.1f} GBP &nbsp;|&nbsp; 💴 ¥{fiyat_jpy:.0f} JPY
             </p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
        st.caption(t['note'])

    # =========================================================
    # 7.5 B2C AKILLI BÜTÇE & SEYAHAT PLANLAYICI
    # =========================================================
    if secilen_profil == t['profile_guest']:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader(t['budget_title'])

        b_col1, b_col2, b_col3 = st.columns([2, 1, 2])
        with b_col1:
            user_budget = st.number_input(t['budget_lbl'], value=5000, min_value=1, step=10)
        with b_col2:
            currency_choice = st.selectbox(t['budget_currency'], ["TRY", "USD", "EUR", "GBP", "JPY"])
        with b_col3:
            user_days = st.number_input(t['days_lbl'], value=3, min_value=1, step=1)

        if st.button(t['plan_btn']):
            kurlar_bütçe = guncel_doviz_kurlarini_getir()
            user_budget_tl = user_budget / kurlar_bütçe.get(currency_choice, 1.0)

            oda_filtresi = raw_df[raw_df['room_type'] == secilen_oda]
            ilce_maliyetleri = oda_filtresi.groupby('neighbourhood')['price'].mean().to_dict()
            ilce_yorumlari = oda_filtresi.groupby('neighbourhood')['number_of_reviews'].mean().to_dict()

            # --- MALİYET HESABINA MEVSİMSELLİK KATSAYISI ETKİSİ ---
            su_anki_ilce_fiyati = ilce_maliyetleri.get(secilen_ilce, raw_df['price'].mean()) * mevsim_katsayisi
            kalan_gece_tam = math.floor(user_budget_tl / su_anki_ilce_fiyati)

            st.markdown(f"#### {t['budget_report']}")

            if kalan_gece_tam < user_days:
                eksik_tutar_tl = (su_anki_ilce_fiyati * user_days) - user_budget_tl
                eksik_tutar_para = eksik_tutar_tl * kurlar_bütçe.get(currency_choice, 1.0)

                ana_metni = t['b_rep1'].format(ilce=secilen_ilce, tl_bütçe=user_budget_tl, kalan=kalan_gece_tam,
                                               hedef=user_days)
                ek_metni = t['b_upsell'].format(hedef=user_days, ek_tutar=eksik_tutar_para, birim=currency_choice)

                st.error(ana_metni + ek_metni)
            else:
                st.success(t['b_rep1'].format(ilce=secilen_ilce, tl_bütçe=user_budget_tl, kalan=kalan_gece_tam,
                                              hedef=user_days))

            # Alternatif ekonomik veya popüler konumlar listesi
            bütçe_limit_gecelik_tl = user_budget_tl / user_days
            ekonomik_ilceler = [k for k, v in sorted(ilce_maliyetleri.items(), key=lambda item: item[1]) if
                                (v * mevsim_katsayisi) < su_anki_ilce_fiyati][:3]
            if ekonomik_ilceler:
                st.markdown(t['b_rep2'] + ", ".join([f"{i}" for i in ekonomik_ilceler]))

            uygun_ilceler = {k: v * mevsim_katsayisi for k, v in ilce_maliyetleri.items() if (v * mevsim_katsayisi) <= bütçe_limit_gecelik_tl}
            populer_uygunlar = [k for k, v in
                                sorted(uygun_ilceler.items(), key=lambda item: ilce_yorumlari.get(item[0], 0),
                                       reverse=True) if k != secilen_ilce][:3]
            if populer_uygunlar:
                st.markdown(t['b_rep3'] + ", ".join([f"{i}" for i in populer_uygunlar]))

# =========================================================
# 8. [KOŞULLU GÖSTERİM] GEZGİN / MÜŞTERİ FIRSAT MOTORU
# =========================================================
if secilen_profil == t['profile_guest']:
    st.markdown("---")
    st.subheader(t['b2c_header'])

    if not ilce_verileri.empty:
        ilce_oda_verileri = ilce_verileri[ilce_verileri['room_type'] == secilen_oda]
        ilce_ort_fiyat = ilce_oda_verileri['price'].mean() if not ilce_oda_verileri.empty else ilce_verileri[
            'price'].mean()
        ilce_ort_yorum = ilce_oda_verileri['number_of_reviews'].mean() if not ilce_oda_verileri.empty else \
        ilce_verileri['number_of_reviews'].mean()

        gizli_cevherler = ilce_verileri[
            (ilce_verileri['price'] <= ilce_ort_fiyat) &
            (ilce_verileri['number_of_reviews'] >= ilce_ort_yorum) &
            (ilce_verileri['room_type'] == secilen_oda)
            ].sort_values(by="number_of_reviews", ascending=False).head(5)

        if not gizli_cevherler.empty:
            st.caption(t['b2c_info'].format(ilce=secilen_ilce))
            df_gems = gizli_cevherler[['name', 'room_type', 'price', 'number_of_reviews', 'availability_365']].rename(
                columns={'name': t['table_name'], 'room_type': t['table_type'], 'price': t['table_price'],
                         'number_of_reviews': t['table_reviews'], 'availability_365': t['table_avail']}
            )
            st.dataframe(df_gems, use_container_width=True, hide_index=True)
        else:
            st.caption("Seçilen oda tipi için bu ilçede belirgin bir gizli cevher ilanı eşleşmedi.")

# =========================================================
# 9. [KOŞULLU GÖSTERİM] YATIRIMCI & EV SAHİBİ PAZAR ANALİZİ
# =========================================================
if secilen_profil == t['profile_host']:
    st.markdown("---")
    st.subheader(t['b2b_header'])

    if not ilce_verileri.empty:
        ilce_oda_verileri = ilce_verileri[ilce_verileri['room_type'] == secilen_oda]
        toplam_aktif_ilan = len(ilce_oda_verileri) if not ilce_oda_verileri.empty else 0

        istanbul_genel_kategori_toplam = len(raw_df[raw_df['room_type'] == secilen_oda])
        pazar_payi = (
                    toplam_aktif_ilan / istanbul_genel_kategori_toplam * 100) if istanbul_genel_kategori_toplam > 0 else 0
        
        # --- PAZAR ANALİZİNE MEVSİMSELLİK ETKİSİ ---
        ilce_ort_fiyat = (ilce_oda_verileri['price'].mean() if not ilce_oda_verileri.empty else ilce_verileri['price'].mean()) * mevsim_katsayisi

        m_col0, m_col_share, m_col1, m_col3 = st.columns(4)
        with m_col0:
            st.metric(t['b2b_m0'], f"{toplam_aktif_ilan} İlan" if lang == "Türkçe" else f"{toplam_aktif_ilan} Listings")
        with m_col_share:
            st.metric(t['b2b_m_share'], f"%{pazar_payi:.2f}")
        with m_col1:
            st.metric(t['b2b_m1'], f"{ilce_ort_fiyat:,.0f} TL")
        with m_col3:
            acente_orani = (ilce_oda_verileri[
                                'calculated_host_listings_count'] > 3).mean() * 100 if not ilce_oda_verileri.empty else (
                                                                                                                                    ilce_verileri[
                                                                                                                                        'calculated_host_listings_count'] > 3).mean() * 100
            st.metric(t['b2b_m3'], f"%{acente_orani:.1f}")

        st.markdown(f"#### {t['b2b_bench']}")
        en_populer_5 = ilce_verileri[ilce_verileri['room_type'] == secilen_oda].sort_values(by="number_of_reviews",
                                                                                            ascending=False).head(5)

        if not en_populer_5.empty:
            df_bench = en_populer_5[['name', 'room_type', 'price', 'number_of_reviews', 'availability_365']].rename(
                columns={'name': t['table_name'], 'room_type': t['table_type'], 'price': t['table_price'],
                         'number_of_reviews': t['table_reviews'], 'availability_365': t['table_avail']}
            )
            st.dataframe(df_bench, use_container_width=True, hide_index=True)

# =========================================================
# 10. COĞRAFİ HARİTA GÖSTERİMİ
# =========================================================
st.markdown("---")
st.subheader(t['map_title'])
harita_verisi = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
st.map(harita_verisi)