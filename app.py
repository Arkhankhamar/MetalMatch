import streamlit as st
import pandas as pd
from joblib import load
#from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import base64

# Fonction pour obtenir la représentation base64 d'un fichier binaire
def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Fonction pour définir une image comme fond d'écran
def set_background(png_file):
    # Obtenir la représentation base64 de l'image
    bin_str = get_base64(png_file)
    # Utiliser la représentation base64 pour définir l'image comme fond d'écran
    page_bg_img = f"""
        <style>
            .stApp {{
                background-image: url("data:image/png;base64,{bin_str}");
                background-size: cover;
            }}
        </style>
    """
    # Appliquer le style avec l'image de fond à la page Streamlit
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Appeler la fonction pour définir l'image comme fond d'écran
set_background("background.png")




# Charger le DataFrame
chemin = Path(__file__).parent
fichier_data = chemin / "df_clean.csv"
df = pd.read_csv(fichier_data)

# Charger le modèle Nearest Neighbors
modelNN = load(chemin / "model.joblib")
# Créer une instance de TfidfVectorizer pour transformer l'entrée utilisateur en une représentation tf-idf
tfidf = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df['soup'])

# Interface utilisateur Streamlit
st.title("!Viens ici Jean-Hellfest inculte!")
st.header('Complète ta culture métal grâce à cet outil')

st.write('\n')
st.write('\n')

dico_bands = {name: index for name, index in zip(df['band_name'], df.index)}

with st.sidebar:
    st.image('logo_heavymetal.png')
    st.title('\m/ (>.<) \m/ . . \m/ (>.<) \m/')
    st.header("Tu as aimé un groupe lors d'un festoche mais tu ne connais pas d'autres groupe du même genre: cet outil est fait pour toi!!!")

col1, col2 = st.columns(2)

eureka=False

with col1:
    #user_input = st.text_input('Entrez un nom de groupe:')
    user_input = st.selectbox('Choisissez un groupe:', df['band_name'].sort_values(ascending=True))

    if st.button('Rechercher:'):
        # Trouver l'index du groupe saisi par l'utilisateur
        user_index = dico_bands[user_input]
        st.write('Tu as choisi:')

        choix = df.iloc[user_index, :-1]
        st.write(f"Nom : {choix[0]}")
        st.write(f"Style : {choix[5]}")
        st.write(f"Origine : {choix[3]}")
        st.write(f"Formé en : {choix[2]}")
        st.write('\n')  # Ajoute une ligne vide
        # Trouver les voisins les plus proches du nouvel exemple
        _, indices = modelNN.kneighbors(tfidf_matrix[user_index])
        eureka=True

with col2:
    # Afficher les recommandations (noms des groupes similaires)
    if eureka:
        st.header('Ecoute ces groupes:')
        for index in indices[0]:
            if index != user_index:  # Ignorer l'entrée utilisateur elle-même
                recommendation = df.loc[index, ['band_name', 'style', 'origin', 'formed']]#.values.tolist()
                st.write(f"Nom : {recommendation[0]}")
                st.write(f"Style : {recommendation[1]}")
                st.write(f"Origine : {recommendation[2]}")
                st.write(f"Formé en : {recommendation[3]}")
                st.write('\n')  # Ajoute une ligne vide entre chaque recommandation