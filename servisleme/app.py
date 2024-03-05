#!/usr/bin/env python
# coding: utf-8

# In[1]:


#app.py
import streamlit as st
import numpy as np
import joblib
from predict_cost import predict


# In[2]:


st.markdown('SATILIK KONUT FİYAT TAHMİNİ')
st.write('---------------------')
ev_türü= st.number_input('0=Bina / 1=Daire / 2=Dağ Evi / 3=Köy Evi / 4=Müstakil Ev / 5=Villa / 6=Yazlık')
metrekare= st.number_input('0 dan büyük bir metrekare bilgisi giriniz',key="input_1")
oda_sayısı = st.number_input('ODA SAYISI (0=1 odalı / 1=2 odalı / 2=3 odalı / 3=4 odalı / 4=4 ve üzeri)',key="input_2")
bina_yaşı = st.number_input('BİNA YAŞI (0=0-5 yaş / 1=10-25 yaş / 2=5-10 yaş)',key="input_3")
dairenin_katı = st.number_input('DAİRE KATI (0 =-1.kat / 1 =-3.kat / 2 =1-10 katları / 3=11-20 katları  / 4=diğer',key="input_4")
takas = st.number_input('TAKAS (0=VAR /1=YOK)',key="input_5")
ısıtma = st.number_input('ISITMA TÜRÜ (0=diğer / 1=doğalgaz / 2=klimalı / 3=sobalı / 4=şömine)', key="input_6")
yapı_durumu = st.number_input('YAPI DURUMU(0=Sıfır / 1=İkinci El)', key="input_7") 
yapı_tipi = st.number_input('YAPI TİPİ (0=Betonarme / 1=Kagir / 2=Prefabrik / 3=Çelik)', key="input_8")
site_içerisinde = st.number_input('SİTE İÇERİSİNDE (0=evet / 1=hayır)',key="input_9")
eşya_durumu = st.number_input('EŞYA DURUMU (0=Boş / 1=Eşyalı)', key="input_10")
banyo_sayısı = st.number_input('BANYO SAYISI (0-5 arası değer giriniz)',key="input_11")
wc_sayısı = st.number_input('WC SAYISI (0-5 arası değer giriniz)', key="input_12")
#tahmin butonu

if st.button('Ev fiyatını tahmin et'):
        fiyat=predict(np.array([[ev_türü,metrekare,oda_sayısı,bina_yaşı,dairenin_katı,takas,ısıtma,yapı_durumu,yapı_tipi,site_içerisinde,eşya_durumu,banyo_sayısı,wc_sayısı]]))
        st.write(f"Tahmin Edilen Fiyat: {fiyat[0]:,.2f} TL")


# In[ ]:




