import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB
from PIL import Image
image = Image.open('gambar1.png')
st.image(image)
# --server.port 80

st.write("""
# Aplikasi Pemilihan Bibit Kelapa Sawit
""")


st.sidebar.header('Silahkan Masukkan Kondisi Lahan Kebun Anda')

def user_input_features():
    JenisTanah = st.sidebar.selectbox('Jenis Tanah',('Mineral','Pasir'))
    JenisDataran = st.sidebar.selectbox('Jenis Dataran',('Dataran Rendah','Rawa-rawa'))
    KondisiDataran = st.sidebar.selectbox('Kondisi Dataran',('Miring','Datar'))
    PHTanah = st.sidebar.slider('PH Tanah',1,14,0)
    KelasDrainase = st.sidebar.selectbox('Kelas Drainase',('Lancar','Sedang','Terhambat','Tidak Ada'))
    
    st.sidebar.header('Rencana Perawatan Tanaman Pada Umur 5-10 Tahun')

    Pruning = st.sidebar.slider('Pruning (Semester)',1,3,0)
    PengendalianLalang = st.sidebar.slider('Pengendalian Lalang (Semester)',1,4,0)
    GarukPiringan = st.sidebar.slider('Garuk Piringan(Semester)',1,4,0)
    PenyemprotanPiringan = st.sidebar.slider('Penyemprotan Piringan (Semester)',1,4,0)
    BTP = st.sidebar.slider('BTP (Semester)',1,4,0)
    BabatSemak = st.sidebar.slider('Babat Semak(Semester)',1,4,0)
    Urea = st.sidebar.slider('Urea (KG)/Batang',0.0,3.0,0.0)
    MOP = st.sidebar.slider('MOP (KG)/Batang',0.0,3.0,0.0)
    Dolomite = st.sidebar.slider('Dolomite (KG)/Batang',0.0,2.0,0.0)
    ZNSO4 = st.sidebar.slider('Zinc Sulphate (ZnSO4) (KG)/Batang',0.0,2.0,0.0)
    CUSO4 = st.sidebar.slider('Copper Sulphate (CuSO4) (KG)/Batang',0.0,2.0,0.0)
    RockPhosphate = st.sidebar.slider('Rock Phosphat (KG)/Batang',0.0,2.0,0.0)
    NPK = st.sidebar.slider('NPK (KG)/Batang',0.0,3.0,0.0)
    HGFP = st.sidebar.slider('HGFP (KG)/Batang',0.0,1.0,0.0)
    TSP = st.sidebar.slider('TSP (KG)/Batang',0.0,1.0,0.0)
    data = {
            'JenisTanah' : JenisTanah,
            'JenisDataran':JenisDataran,
            'KondisiDataran': KondisiDataran,
            'PHTanah' : PHTanah,
            'KelasDrainase' : KelasDrainase,
            'Pruning':Pruning,
            'PengendalianLalang': PengendalianLalang,
            'GarukPiringan' : GarukPiringan,
            'PenyemprotanPiringan' : PenyemprotanPiringan,
            'BTP' : BTP,
            'BabatSemak' : BabatSemak,
            'Urea' : Urea,
            'MOP' : MOP,
            'Dolomite' : Dolomite,
            'ZNSO4' : ZNSO4,
            'CUSO4' : CUSO4,
            'RockPhosphate' : RockPhosphate,
            'NPK' : NPK,
            'HGFP' : HGFP,
            'TSP' : TSP
            }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

st.subheader('Data Kondisi Lahan dan Perawatan yang telah diisi')
st.write(input_df)

input_df['JenisTanah'] = input_df['JenisTanah'].replace("Pasir", "1")
input_df['JenisTanah'] = input_df['JenisTanah'].replace("Mineral", "2")

input_df['JenisDataran'] = input_df['JenisDataran'].replace("Dataran Rendah", "1")
input_df['JenisDataran'] = input_df['JenisDataran'].replace("Rawa-rawa", "2")

input_df['KondisiDataran'] = input_df['KondisiDataran'].replace("Miring", "1")
input_df['KondisiDataran'] = input_df['KondisiDataran'].replace("Datar", "2")

input_df['KelasDrainase'] = input_df['KelasDrainase'].replace("Tidak Ada", "1")
input_df['KelasDrainase'] = input_df['KelasDrainase'].replace("Lancar", "2")
input_df['KelasDrainase'] = input_df['KelasDrainase'].replace("Sedang", "3")
input_df['KelasDrainase'] = input_df['KelasDrainase'].replace("Terhambat", "4")

# kombinasi user input dengan masukan dataset
df = pd.read_csv('dataset.csv')

#df.drop(['NamaBlok'],axis=1,inplace=True)

df['JenisTanah'] = df['JenisTanah'].replace("Pasir", "1")
df['JenisTanah'] = df['JenisTanah'].replace("Mineral", "2")

df['JenisTanah'] = df['JenisTanah'].replace(" Pasir ", "1")
df['JenisTanah'] = df['JenisTanah'].replace(" Mineral ", "2")

df['JenisDataran'] = df['JenisDataran'].replace("DataranRendah", "1")
df['JenisDataran'] = df['JenisDataran'].replace("Rawa-rawa", "2")

df['KondisiDataran'] = df['KondisiDataran'].replace("Miring", "1")
df['KondisiDataran'] = df['KondisiDataran'].replace("Datar", "2")

df['KelasDrainase'] = df['KelasDrainase'].replace("TidakAda", "1")
df['KelasDrainase'] = df['KelasDrainase'].replace("Lancar", "2")
df['KelasDrainase'] = df['KelasDrainase'].replace("Sedang", "3")
df['KelasDrainase'] = df['KelasDrainase'].replace("Terhambat", "4")

#st.write(df)

#features = df.columns[1:6]
#df = df.drop('Unnamed:  0', axis=1)
X = df.drop('JenisBibit', axis=1)
y = df['JenisBibit']

clf = GaussianNB()
clf.fit(X, y)

prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)

st.subheader('Prediksi Bibit :')
st.write(prediction)

