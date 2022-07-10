# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)

# 1. İş Problemi (Business Problem)
# 2. Veriyi Anlama (Data Understanding)
# 3. Veri Hazırlama (Data Preparation)
# 4. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
# 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)

###############################################################
# 1. İş Problemi (Business Problem)
###############################################################
# Online ayakkabı mağazası olan FLO müşterilerini segmentlere ayırıp busegmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranışlardaki öbeklenmelere göre gruplar oluşturulacak.

# Veri Seti Hikayesi

# FLO Veri Seti Değişken Açıklamaları
# master_id: Eşsiz müşteri numarası
# order_channel: Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel: En son alışverişin yapıldığı kanal
# first_order_date: Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date: Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi



###############################################################
# 1: Veriyi Anlama ve Hazırlama (Data Understanding and Preparing)
###############################################################
import datetime as dt
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width',1000)

# "flo_data_20K.csv" verisetinin okunması. Dataframe’in kopyasının oluşturulması.
df_ = pd.read_csv("/Users/zinnetbahcetepe/Desktop/VeriBilimiOkulu/pythonProject/ödevler/Modül3/flo_data_20k.csv")
df =df_.copy()
df.head()

#  Verisetinde
# a. İlk 10 gözlem,
df.head(10)
# b. Değişken isimleri,
df.columns
# c. Betimsel istatistik,
df.describe().T
# d. Boş değer,
df.isnull().sum()
# e. Boyut,
df.shape
# f. Değişken tipleri, incelemesi yapınız.
df.info()


# Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir.
# Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz:
df.head()
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.info()
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()

# Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.
df.groupby("order_channel").agg({"master_id" : "count",
                                 "order_num_total" : "sum",
                                 "customer_value_total" : "sum"})
#en fazla AndroidApp'ten alışveriş yapılmış, (%50'ye yakını) %25 mobil,
#geri kaln deskop ve IOS'tan

# En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
df.sort_values("customer_value_total", ascending=False).head(10)

# En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
df.sort_values.("order_num_total", ascending=False).head(10)

# Veri ön hazırlık sürecini fonksiyonlaştırınız.
def pre_processing(dataframe):
    #Veri Setinin Okunması
    dataframe_ = pd.read_csv("/Users/zinnetbahcetepe/Desktop/VeriBilimiOkulu/pythonProject/ödevler/Modül3/flo_data_20k.csv")
    dataframe = dataframe_.copy()

    # Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_online"] + dataframe["customer_value_total_ever_offline"]

    # Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    # Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.
    dataframe_dagilim = dataframe.groupby("order_channel").agg({"master_id": "count",
                                     "order_num_total": "sum",
                                     "customer_value_total": "sum"})
    print(dataframe_dagilim)

    # En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
    dataframe_cvt = dataframe.sort_values("customer_value_total", ascending=False).head(10)
    print(dataframe_cvt)

    # En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
    dataframe_ont = dataframe.sort_values("order_num_total", ascending=False).head(10)
    print(dataframe_ont)

    return dataframe

pre_processing(df)

###############################################################
# RFM Metriklerinin Hesaplanması
###############################################################

# Recency, Frequency ve Monetary tanımlarını yapınız.
#Recency: müşterinin yeni olma durumu
#Frequency: müşterinin alışveriş sıklığı
#Monetary:müşterinin harcama miktarı

# Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.

df["last_order_date"].max()#2021-05-30 en son alışveriş tarihi
#Recency (müşterinin yeni olma durumunun) hesaplanabilmesi için analiz yaptığımız tarihi alışveriş tarihinden ayırmalıyız.
#bunun için en basit yol en son alışveriş tarihine +2 gün ekleyip analizi gerçekleştirdiğimi tariihmiş gibi belirlemek

#recency değeri için bugünün tarihini (2021, 06, 01) olarak kabul ediniz.
analysis_date = dt.datetime(2021, 6, 1)

# Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.
df.columns
rfm_= df.groupby('master_id').agg({
    'last_order_date' : lambda date: (analysis_date-date.max()).days,
    'order_num_total' : lambda num: num.nunique(),
    'customer_value_total' : lambda TotalPrice: TotalPrice.sum()})

rfm_.sort_values('master_id', ascending=False).head()
rfm_["customer_id"] = df["master_id"]

# Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz.
rfm_.columns = ["customer_id", "recency", "frequency", "monetary"]

#ikinci yol
rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
rfm["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]')
rfm["frequency"] = df["order_num_total"]
rfm["monetary"] = df["customer_value_total"]

rfm.sort_values('customer_id', ascending=False).head()

###############################################################
# RF Skorunun Hesaplanması
###############################################################


# Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
# Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.
rfm["RF_SCORE"] = (rfm["recency_score"].astype(str) +
                   rfm["frequency_score"].astype(str))

rfm["RF_SCORE"].head()
###############################################################
# RF Skorunun Segment Olarak Tanımlanması
###############################################################

# Oluşturulan RF skorları için segment tanımlamaları yapınız.
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

# Aşağıdaki seg_map yardımı ile skorları segmentlere çeviriniz.
rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

rfm.head()
###############################################################
# Görev 5: Aksiyon Zamanı !
###############################################################

# Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg({"mean", "count"})

#  RFM analizi yardımıyla aşağıda verilen 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv olarak kaydediniz.
### a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor.
# Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde.
# Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçmek isteniliyor.

target_segments_customer_ids = rfm[rfm["segment"].isin(["champions","loyal_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) &(df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
cust_ids.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)
cust_ids.shape

cust_ids.head()


# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden olan ama uzun süredir
# alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.
target_segments_customer_ids = rfm[rfm["segment"].isin(["cant_loose","hibernating","new_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & ((df["interested_in_categories_12"].str.contains("ERKEK"))|(df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]
cust_ids.to_csv("indirim_hedef_müşteri_ids.csv", index=False)

cust_ids.shape

###############################################################
# TÜM İŞLEMLERİN FONKSİYONLAŞTIRILMASI
###############################################################

def create_rfm(dataframe):
    # Veriyi Hazırlma
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)


    # RFM METRIKLERININ HESAPLANMASI
    dataframe["last_order_date"].max()  # 2021-05-30
    analysis_date = dt.datetime(2021, 6, 1)
    rfm = pd.DataFrame()
    rfm["customer_id"] = dataframe["master_id"]
    rfm["recency"] = (analysis_date - dataframe["last_order_date"]).astype('timedelta64[D]')
    rfm["frequency"] = dataframe["order_num_total"]
    rfm["monetary"] = dataframe["customer_value_total"]

    # RF ve RFM SKORLARININ HESAPLANMASI
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
    rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))


    # SEGMENTLERIN ISIMLENDIRILMESI
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }
    rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

    return rfm[["customer_id", "recency","frequency","monetary","RF_SCORE","RFM_SCORE","segment"]]

rfm_df = create_rfm(df)





