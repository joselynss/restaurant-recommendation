import streamlit as st
import pandas as pd
from datetime import datetime
import os
import pickle
import time
import numpy as np
from surprise import Dataset, Reader, SVDpp

df = pd.read_csv('final_processed_data.csv')
all_restaurants = sorted(df['Restaurant Name'].unique())

with open('svd.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
rating_data_df = data['rating_data_df']
resto_encoder = data['resto_encoder']

def recommend_from_favorite_restos(fav_resto_names, df, resto_encoder, model, k=10):
    try:
        fav_resto_ids = resto_encoder.transform(fav_resto_names)
    except:
        raise ValueError("Ada nama restoran yang tidak ditemukan di dataset.")

    new_user_id = df['User ID'].max() + 1

    new_df = pd.DataFrame({
        'User ID': [new_user_id] * len(fav_resto_ids),
        'Restaurant ID': fav_resto_ids,
        'User Rating': [5.0] * len(fav_resto_ids)
    })

    temp_df = pd.concat([df, new_df], ignore_index=True)

    reader = Reader(rating_scale=(1, 5))
    trainset = Dataset.load_from_df(temp_df[['User ID', 'Restaurant ID', 'User Rating']], reader).build_full_trainset()

    temp_model = SVDpp(n_factors=200, n_epochs=20, verbose=False)
    temp_model.fit(trainset)

    all_restaurant_ids = df['Restaurant ID'].unique()
    unrated_restaurant_ids = np.setdiff1d(all_restaurant_ids, fav_resto_ids)

    predictions = [
        (resto_id, temp_model.predict(new_user_id, resto_id).est)
        for resto_id in unrated_restaurant_ids
    ]
    predictions.sort(key=lambda x: x[1], reverse=True)

    top_k_predictions = predictions[:k]
    top_k_names = [(resto_encoder.inverse_transform([resto_id])[0], rating)
                   for resto_id, rating in top_k_predictions]

    return top_k_names

# save to sheets
import gspread
from oauth2client.service_account import ServiceAccountCredentials
def log_to_google_sheet(selected, recommendations, satisfaction):
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('/Users/joselyn/Downloads/thesis_code2/virtual-dynamo-463017-n3-139684f32bad.json', scope)
    client = gspread.authorize(creds)
    sheet = client.open("log_from_user").sheet1
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "|".join(selected),
        "|".join(recommendations),
        satisfaction
    ]
    sheet.append_row(row)


def main():
    st.title("🍽️ Restaurant Recommendation System in Jakarta 🍽️")

    # state
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'show_recommendations' not in st.session_state:
        st.session_state.show_recommendations = False
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False

    # Halaman Thank You
    if st.session_state.submitted:
        st.markdown('''
            :green[**Terima kasih atas penilaian yang diberikan 😊**]
            ''')
        if st.button("⬅️ Back to Main Page"):
            st.session_state.clear()
            st.rerun()
        return
    
    # Halaman Utama
    st.markdown(
        "<h5><strong>Pilih 3 - 5 restoran favorit kamu di Jakarta:</strong></h5>",
        unsafe_allow_html=True
    )
    selected = st.multiselect(
        label="",
        options=all_restaurants,
        placeholder="Ketik untuk mencari restoran"
    )

    if len(selected) == 0:
        pass
    elif len(selected) < 3:
        st.markdown(
            "<p style='color:red;'>Pilih minimal 3 restoran untuk mendapatkan rekomendasi.</p>",
            unsafe_allow_html=True
        )
    elif len(selected) > 5:
        st.markdown(
            "<p style='color:red;'>Maksimal 5 restoran yang bisa dipilih.</p>",
            unsafe_allow_html=True
        )

    if 3 <= len(selected) <= 5 and not st.session_state.show_recommendations:
        if st.button("🍔 Get Recommendation"):
            with st.spinner("Sedang memproses rekomendasi kamu..."):
                recommendations = recommend_from_favorite_restos(selected, rating_data_df, resto_encoder, model, k=10)
            st.session_state.recommendations = [resto for resto, rating in recommendations]
            st.session_state.show_recommendations = True
            st.rerun()

    if st.session_state.show_recommendations:
        st.success("Berikut 10 rekomendasi restoran untukmu:")
        for i, resto in enumerate(st.session_state.recommendations, 1):
            st.write(f"{i}. {resto}")

        st.markdown("---")

        satisfaction = st.slider(
            "Seberapa puas Anda dengan hasil rekomendasi yang diberikan?",
            min_value=0,
            max_value=10,
            step=1,
            key='satisfaction'
        )

        if satisfaction is not None:
            st.write("Kamu menilai ", satisfaction, "dari 10")

            if st.button("Submit"):
                # save to sheets
                log_to_google_sheet(selected, st.session_state.recommendations, satisfaction)
                st.session_state.submitted = True
                st.rerun()

if __name__ == '__main__':
    main()

# matplotlib==3.10.0
# numpy==1.26.4 #1.24.4
# pandas==2.2.2 #2.3.0
# scikit-learn==1.6.1
# seaborn==0.13.2
# streamlit==1.45.1
# surprise==1.1.4