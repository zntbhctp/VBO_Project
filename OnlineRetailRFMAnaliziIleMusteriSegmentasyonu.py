###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

# 1. İş Problemi (Business Problem)
# 2. Veriyi Anlama (Data Understanding)
# 3. Veri Hazırlama (Data Preparation)
# 4. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
# 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)
# 7. Tüm Sürecin Fonksiyonlaştırılması

###############################################################
# 1. İş Problemi (Business Problem)
###############################################################

# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.

# Veri Seti Hikayesi
# https://archive.ics.uci.edu/ml/machine-learning-databases/00502/

# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.
#Bu Analizde 2010 - 2011 arası datalar RFM ile Analiz edilecektir

# Değişkenler

# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.

#GÖREV 1
###############################################################
# 2. Veriyi Anlama (Data Understanding)
###############################################################

import datetime as dt
from typing import Union, Any

import pandas as pd
#conda install openpyxl #preferences -> Python Interpreter -> + buton -> openpyxl -> install package
from numpy import ndarray
from pandas import Series, DataFrame
from pandas.core.generic import NDFrame

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None) #tüm satırları görmek demektir, çok veri olduğundan bu veri setinde istemiyoruz
pd.set_option('display.float_format', lambda x: '%.3f' % x) #float datatyp'larda virgülden sonra 3 basamak ayarı
pd.set_option('display.width', 500)
#Adım 1: Online Retail II excelindeki 2010-2011 verisini okuyunuz. Oluşturduğunuz dataframe’in kopyasını oluşturunuz.
df = pd.read_excel("/Users/zinnetbahcetepe/Desktop/VeriBilimiOkulu/pythonProject/ödevler/Modül3/online_retail_II.xlsx", sheet_name="Year 2010-2011")

#veri seti yedeklenir
df_ = df.copy()
#Adım 2: Veri setinin betimsel istatistiklerini inceleyiniz.
df.head()
df.shape
df.columns
#Adım 3: Veri setinde eksik gözlem var mı? Varsa hangi değişkende kaç tane eksik gözlem vardır?
df.isnull().sum()
#Description: 1454, Customer ID:135080 adet eksik gözlem vardı

#Adım 4: Eksik gözlemleri veri setinden çıkartınız. Çıkarma işleminde ‘inplace=True’ parametresini kullanınız.
df.dropna(inplace=True)
df.shape


#Adım 5: Eşsiz ürün sayısı kaçtır?
df["Description"].nunique()

#Adım 6: Hangi üründen kaçar tane vardır?
df["Description"].value_counts().head()

#Adım 7: En çok sipariş edilen 5 ürünü çoktan aza doğru sıralayınız
df.groupby("Description").agg({'Quantity': 'sum'}).sort_values('Quantity', ascending=False).head()

###############################################################
# 2. Veriyi Hazırlama (Data Preparation)
###############################################################
#Eksik veri silme bu başlık altında olmalı
#Adım 4: Eksik gözlemleri veri setinden çıkartınız. Çıkarma işleminde ‘inplace=True’ parametresini kullanınız.
df.dropna(inplace=True)

#Adım 8: Faturalardaki ‘C’ iptal edilen işlemleri göstermektedir. İptal edilen işlemleri veri setinden çıkartınız.
#tilda işareti kısayolu: option+N
df = df[~df["Invoice"].str.contains("C", na=False)]
#atamada yaparak iptal faturaları veri setinden çıkardık
df.head()
#Adım 9: Fatura başına elde edilen toplam kazancı ifade eden ‘TotalPrice’ adında bir değişken oluşturunuz
df["TotalPrice"] = df["Quantity"]*df["Price"]
#Fatura başına toplam fiyat
df.groupby("Invoice").agg({"TotalPrice": "sum"}).head()


#GOREV 2: RFM Metriklerinin Hesaplanması

#Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız.
#Recency: müşterinin yeni olma durumu
#Frequency: müşterinin alışveriş sıklığı
#Monetary:müşterinin harcama miktarı

df["InvoiceDate"].max()#2011-12-09 en son alışveriş tarihi
#Recency (müşterinin yeni olma durumunun) hesaplanabilmesi için analiz yaptığımız tarihi alışveriş tarihinden ayırmalıyız.
#yani müşteri bizim analiz yaptığımız tarihe yakınlığına göre yeni eski olmalı
#bunun için en basit yol en son alışveriş tarihine +2 gün ekleyip analiz tariihi olarak belirleme

#recency değeri için bugünün tarihini (2011, 12, 11) olarak kabul ediniz.
today_date = dt.datetime(2011, 12, 11)

#Adım 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini groupby, agg ve lambda ile hesaplayınız.
#Adım 3: Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.
rfm = df.groupby("Customer ID").agg({
    "InvoiceDate": lambda date: (today_date-date.max()).days,
    "Invoice": lambda num: num.nunique(),
    "TotalPrice": lambda TotalPrice: TotalPrice.sum()})

#Adım 4: Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz.
rfm.columns = ["Recency", "Frequency", "Monetary"]


#rfm dataframe’ini oluşturduktan sonra veri setini "monetary>0" olacak şekilde filtreleyiniz.
rfm: Union[Union[Series, None, ndarray, DataFrame, NDFrame], Any] = rfm[rfm["Monetary"] > 0]
rfm.head()

#GOREV 3:RFM Skorlarının Oluşturulması ve Tek bir Değişkene Çevrilmesi
#Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
#Adım 2: Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.
#recenncy de istediğimiz şey yeni müşteri olduğu için labels 5'ten 1'e doğru
rfm["recency_score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])

rfm["monetary_score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

rfm["frequency_score"] = pd.qcut(rfm["Frequency"], 5, labels=[1, 2, 3, 4, 5])
#burada hata aldık bazı değerler birden fala tekrar ettiğinden uniqe olmadığından qcutı kullanamadı
#rank fonksiyonunu kullanarak ilk gördüğün değeri ilk eşleşen labela ata demiş olacağız

rfm["frequency_score"] = pd.qcut(rfm["Frequency"].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])

#Adım 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.

#Adım 2: Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.

#Adım 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.
rfm["RF_SCORE"] = (rfm["recency_score"].astype(str) +
                   rfm["frequency_score"].astype(str))

rfm["RF_SCORE"].head()

#GOREV 4:RF Skorunun Segment Olarak Tanımlanması
#Adım 1: Oluşturulan RF skorları için segment tanımlamaları yapınız.
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
#champions müşteriler 439 kişi
rfm[rfm["RF_SCORE"]=='55']

#Adım 2: Aşağıdaki seg_map yardımı ile skorları segmentlere çeviriniz.
#seg_map te yaptığımız sözlük yapısını kullanarak segment isimlendirmesi yapma
rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)
rfm["segment"].head()

#Görev 5: Aksiyon Zamanı !
#Adım 1: Önemli gördüğünü 3 segmenti seçiniz.
# Bu üç segmenti hem aksiyon kararları açısından hemde segmentlerin yapısı açısından(ortalama RFM değerleri) yorumlayınız.


#segmente göre groupby yapıp ortalama ve count alalım
rfm[["segment", "Recency", "Frequency", "Monetary"]].groupby("segment").agg(["mean", "count"])
#at_Risk müşteri sayısı 593 buna odaklanılmalı, about_to_sleep  352 kişi belki onlar uyandırılabilir
#cant_loose 63 kişi kaybedilmemeli,need_attention 187 kişi
#new_customers 42 kişi ve sürpriz bir hamle bekliyorlar


#Adım 2: "Loyal Customers" sınıfına ait customer ID'leri seçerek excel çıktısını alınız.
rfm[rfm["segment"]=="loyal_customers"].head()
#müşteriId bilgisi için index

rfm[rfm["segment"]=="loyal_customers"].index

#dışarı aktarım için veriyi hazırlama

new_df = pd.DataFrame()

new_df["new_customer_id"] = rfm[rfm["segment"]=="loyal_customers"].index
#customerId'leri int a çevirme
new_df["new_customer_id"] = new_df["new_customer_id"].astype(int)
new_df["new_customer_id"].head()

#dışarı csv olarak aktarma
new_df.to_csv('new_customer_id.csv')

#tüm segment bilgisini çıktı alma
rfm.to_csv('rfm.csv')
