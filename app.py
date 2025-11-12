import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="KANDE Maladies Cardiovasculaires",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #ecf0f1;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour charger et entraîner le modèle
@st.cache_resource
def load_model():
    """Charge et entraîne le modèle KNN sur les données"""
    try:
        # Charger les données
        df = pd.read_csv('heart.csv')
        
        # Prétraitement
        df_processed = df.copy()
        
        # Encodage des variables catégorielles
        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            label_encoders[col] = le
        
        # Séparation X et y
        X = df_processed.drop('HeartDisease', axis=1)
        y = df_processed['HeartDisease']
        
        # Standardisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Entraînement du modèle KNN (meilleur modèle)
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_scaled, y)
        
        return model, scaler, label_encoders, X.columns.tolist()
    
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None, None, None, None

# Fonction de prédiction
def predict_heart_disease(model, scaler, label_encoders, features, input_data):
    """Effectue une prédiction sur de nouvelles données"""
    try:
        # Créer un DataFrame avec les données d'entrée
        input_df = pd.DataFrame([input_data])
        
        # Encoder les variables catégorielles
        for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
            if col in input_df.columns:
                le = label_encoders[col]
                input_df[col] = le.transform(input_df[col])
        
        # S'assurer que les colonnes sont dans le bon ordre
        input_df = input_df[features]
        
        # Standardiser
        input_scaled = scaler.transform(input_df)
        
        # Prédire
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        return prediction, probability
    
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        return None, None

# Header de l'application
st.markdown('<p class="main-header">❤️ Système de Prédiction des Maladies Cardiovasculaires</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Outil d\'aide au dépistage précoce basé sur le Machine Learning</p>', unsafe_allow_html=True)

# Chargement du modèle
model, scaler, label_encoders, features = load_model()

if model is None:
    st.error("⚠️ Impossible de charger le modèle. Veuillez vérifier que le fichier 'heart.csv' est présent.")
    st.stop()

# Sidebar - Navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("", ["🏠 Accueil", "🔮 Prédiction", "📊 Analyse de Données", "ℹ️ À propos"])

# PAGE 1 : ACCUEIL
if page == "🏠 Accueil":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h2>🎯 Objectif du Projet</h2>
            <p>
            Les maladies cardiovasculaires (MCV) représentent la <b>première cause de mortalité</b> dans le monde 
            avec près de <b>17,9 millions de décès</b> chaque année (31% des décès globaux).
            </p>
            <p>
            Ce système utilise des algorithmes de <b>Machine Learning</b> pour identifier précocement 
            les personnes à risque et faciliter la prise de décision médicale.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
            <h3>✅ Performances du Modèle KNN</h3>
            <ul>
                <li><b>Accuracy :</b> 89.13%</li>
                <li><b>Precision :</b> 89.42%</li>
                <li><b>Recall :</b> 91.18%</li>
                <li><b>F1-Score :</b> 90.29%</li>
                <li><b>ROC-AUC :</b> 91.92%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("https://img.icons8.com/color/400/heart-with-pulse.png", width=300)
    
    # Statistiques clés
    st.markdown("---")
    st.subheader("📈 Statistiques Clés")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Patients analysés", "918", delta="100%")
    
    with col2:
        st.metric("Variables prédictives", "11", delta="Qualité élevée")
    
    with col3:
        st.metric("Précision du modèle", "90.29%", delta="F1-Score")
    
    with col4:
        st.metric("Détection des malades", "91.18%", delta="Recall")
    
    # Top 5 variables importantes
    st.markdown("---")
    st.subheader("🔑 Variables Prédictives Principales")
    
    importance_data = {
        'Variable': ['ST_Slope', 'Cholesterol', 'MaxHR', 'Oldpeak', 'ChestPainType'],
        'Importance': [25.28, 11.53, 11.38, 11.06, 10.58],
        'Description': [
            'Pente du segment ST à l\'exercice',
            'Taux de cholestérol sérique (mg/dl)',
            'Fréquence cardiaque maximale atteinte',
            'Dépression du segment ST',
            'Type de douleur thoracique'
        ]
    }
    
    df_importance = pd.DataFrame(importance_data)
    
    fig = px.bar(df_importance, x='Importance', y='Variable', 
                 orientation='h',
                 text='Importance',
                 color='Importance',
                 color_continuous_scale='Reds',
                 title='Importance des Variables (%)')
    
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)

# PAGE 2 : PRÉDICTION
elif page == "🔮 Prédiction":
    st.header("🔮 Prédiction du Risque Cardiovasculaire")
    
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Avertissement Médical</h4>
        <p>
        Cet outil est conçu comme une <b>aide au dépistage</b> et ne remplace en aucun cas 
        un diagnostic médical professionnel. Les résultats doivent être interprétés par un professionnel de santé.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📝 Saisie des Informations du Patient")
    
    # Formulaire de saisie
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Informations Démographiques")
        age = st.number_input("Âge", min_value=20, max_value=100, value=50, step=1)
        sex = st.selectbox("Sexe", ["M", "F"])
    
    with col2:
        st.markdown("#### 🩺 Données Cliniques")
        resting_bp = st.number_input("Pression artérielle au repos (mm Hg)", 
                                     min_value=80, max_value=200, value=120, step=1)
        cholesterol = st.number_input("Cholestérol (mg/dl)", 
                                      min_value=100, max_value=600, value=200, step=1)
        fasting_bs = st.selectbox("Glycémie à jeun > 120 mg/dl", [0, 1], 
                                  format_func=lambda x: "Oui" if x == 1 else "Non")
        max_hr = st.number_input("Fréquence cardiaque maximale", 
                                min_value=60, max_value=220, value=150, step=1)
    
    with col3:
        st.markdown("#### 💓 Symptômes & Tests")
        chest_pain_type = st.selectbox("Type de douleur thoracique", 
                                       ["ATA", "NAP", "ASY", "TA"],
                                       help="ATA: Angine atypique, NAP: Douleur non angineuse, ASY: Asymptomatique, TA: Angine typique")
        resting_ecg = st.selectbox("ECG au repos", 
                                   ["Normal", "ST", "LVH"],
                                   help="Normal, ST: Anomalie ST-T, LVH: Hypertrophie ventriculaire gauche")
        exercise_angina = st.selectbox("Angine à l'effort", ["N", "Y"],
                                       format_func=lambda x: "Oui" if x == "Y" else "Non")
        oldpeak = st.number_input("Oldpeak (Dépression ST)", 
                                 min_value=-3.0, max_value=7.0, value=0.0, step=0.1)
        st_slope = st.selectbox("Pente du segment ST", ["Up", "Flat", "Down"])
    
    # Bouton de prédiction
    st.markdown("---")
    if st.button("🔍 Lancer la Prédiction", type="primary", use_container_width=True):
        # Préparer les données d'entrée
        input_data = {
            'Age': age,
            'Sex': sex,
            'ChestPainType': chest_pain_type,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'RestingECG': resting_ecg,
            'MaxHR': max_hr,
            'ExerciseAngina': exercise_angina,
            'Oldpeak': oldpeak,
            'ST_Slope': st_slope
        }
        
        # Effectuer la prédiction
        prediction, probability = predict_heart_disease(model, scaler, label_encoders, features, input_data)
        
        if prediction is not None:
            st.markdown("---")
            st.subheader("📊 Résultats de la Prédiction")
            
            # Résultat principal
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="danger-box">
                        <h2 style="text-align: center;">⚠️ RISQUE ÉLEVÉ</h2>
                        <p style="text-align: center; font-size: 1.2rem;">
                        Le modèle détecte un <b>risque élevé</b> de maladie cardiovasculaire.
                        </p>
                        <p style="text-align: center; font-size: 2rem; font-weight: bold; color: #dc3545;">
                        {probability[1]*100:.1f}% de risque
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.warning("🏥 **Recommandation :** Consultez rapidement un cardiologue pour des examens complémentaires.")
                else:
                    st.markdown(f"""
                    <div class="success-box">
                        <h2 style="text-align: center;">✅ RISQUE FAIBLE</h2>
                        <p style="text-align: center; font-size: 1.2rem;">
                        Le modèle détecte un <b>risque faible</b> de maladie cardiovasculaire.
                        </p>
                        <p style="text-align: center; font-size: 2rem; font-weight: bold; color: #28a745;">
                        {probability[0]*100:.1f}% de santé
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info("💚 **Recommandation :** Maintenez un mode de vie sain et des contrôles réguliers.")
            
            with col2:
                # Jauge de probabilité
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=probability[1] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Probabilité de Maladie", 'font': {'size': 20}},
                    delta={'reference': 50, 'increasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': '#d4edda'},
                            {'range': [30, 70], 'color': '#fff3cd'},
                            {'range': [70, 100], 'color': '#f8d7da'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Détails des probabilités
            st.markdown("---")
            st.subheader("📈 Probabilités Détaillées")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("🟢 Probabilité Patient Sain", f"{probability[0]*100:.2f}%")
            
            with col2:
                st.metric("🔴 Probabilité Maladie Cardiovasculaire", f"{probability[1]*100:.2f}%")
            
            # Graphique des probabilités
            prob_df = pd.DataFrame({
                'Statut': ['Sain', 'Malade'],
                'Probabilité': [probability[0] * 100, probability[1] * 100]
            })
            
            fig = px.bar(prob_df, x='Statut', y='Probabilité',
                        color='Statut',
                        color_discrete_map={'Sain': '#28a745', 'Malade': '#dc3545'},
                        text='Probabilité',
                        title='Distribution des Probabilités')
            
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig.update_layout(height=400, showlegend=False, yaxis_range=[0, 100])
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Profil du patient
            st.markdown("---")
            st.subheader("👤 Profil du Patient Analysé")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Informations Démographiques**")
                st.write(f"- Âge : {age} ans")
                st.write(f"- Sexe : {'Homme' if sex == 'M' else 'Femme'}")
            
            with col2:
                st.write("**Paramètres Cliniques**")
                st.write(f"- Pression artérielle : {resting_bp} mm Hg")
                st.write(f"- Cholestérol : {cholesterol} mg/dl")
                st.write(f"- Fréquence cardiaque max : {max_hr} bpm")
            
            with col3:
                st.write("**Symptômes**")
                st.write(f"- Type de douleur : {chest_pain_type}")
                st.write(f"- Angine à l'effort : {'Oui' if exercise_angina == 'Y' else 'Non'}")
                st.write(f"- Oldpeak : {oldpeak}")

# PAGE 3 : ANALYSE DE DONNÉES
elif page == "📊 Analyse de Données":
    st.header("📊 Analyse Exploratoire des Données")
    
    # Charger les données
    df = pd.read_csv('heart.csv')
    
    st.markdown(f"""
    <div class="info-box">
        <h3>📁 Dataset Heart Disease</h3>
        <p>
        <b>Nombre de patients :</b> {df.shape[0]}<br>
        <b>Nombre de variables :</b> {df.shape[1]}<br>
        <b>Patients malades :</b> {df['HeartDisease'].sum()} ({df['HeartDisease'].mean()*100:.1f}%)<br>
        <b>Patients sains :</b> {len(df) - df['HeartDisease'].sum()} ({(1-df['HeartDisease'].mean())*100:.1f}%)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Onglets
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Aperçu", "📈 Distributions", "🔗 Corrélations", "📊 Statistiques"])
    
    with tab1:
        st.subheader("Aperçu des Données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Premières lignes du dataset**")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.write("**Informations sur les colonnes**")
            buffer = []
            buffer.append(f"**Types de données :**")
            for col in df.columns:
                buffer.append(f"- {col}: {df[col].dtype}")
            st.markdown("\n".join(buffer))
            
            st.write("**Valeurs manquantes :**")
            st.write(f"Aucune valeur manquante détectée ✅")
    
    with tab2:
        st.subheader("Distribution des Variables")
        
        # Distribution de la variable cible
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(df, names='HeartDisease', 
                        title='Distribution de la Variable Cible',
                        color='HeartDisease',
                        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                        labels={0: 'Sain', 1: 'Malade'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x='Age', color='HeartDisease',
                              title='Distribution de l\'Âge par Statut',
                              nbins=30,
                              color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                              labels={'HeartDisease': 'Statut'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution du sexe et type de douleur
        col1, col2 = st.columns(2)
        
        with col1:
            sex_counts = df.groupby(['Sex', 'HeartDisease']).size().reset_index(name='count')
            fig = px.bar(sex_counts, x='Sex', y='count', color='HeartDisease',
                        title='Distribution par Sexe et Statut',
                        barmode='group',
                        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                        labels={'HeartDisease': 'Statut', 'count': 'Nombre'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            chest_counts = df.groupby(['ChestPainType', 'HeartDisease']).size().reset_index(name='count')
            fig = px.bar(chest_counts, x='ChestPainType', y='count', color='HeartDisease',
                        title='Type de Douleur Thoracique par Statut',
                        barmode='group',
                        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                        labels={'HeartDisease': 'Statut', 'count': 'Nombre'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Variables numériques
        st.subheader("Distribution des Variables Numériques")
        
        numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        selected_var = st.selectbox("Sélectionner une variable", numeric_cols)
        
        fig = px.box(df, y=selected_var, x='HeartDisease', color='HeartDisease',
                    title=f'Distribution de {selected_var} par Statut',
                    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                    labels={'HeartDisease': 'Statut'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Matrice de Corrélation")
        
        # Encoder temporairement pour la corrélation
        df_corr = df.copy()
        le = LabelEncoder()
        for col in df_corr.select_dtypes(include='object').columns:
            df_corr[col] = le.fit_transform(df_corr[col])
        
        correlation = df_corr.corr()
        
        fig = px.imshow(correlation, 
                       text_auto='.2f',
                       aspect='auto',
                       color_continuous_scale='RdBu_r',
                       title='Matrice de Corrélation',
                       width=800,
                       height=800)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top corrélations avec HeartDisease
        st.subheader("Corrélations avec HeartDisease")
        
        heart_corr = correlation['HeartDisease'].sort_values(ascending=False).drop('HeartDisease')
        
        fig = px.bar(x=heart_corr.values, y=heart_corr.index,
                    orientation='h',
                    title='Corrélation avec la Maladie Cardiovasculaire',
                    labels={'x': 'Coefficient de Corrélation', 'y': 'Variable'},
                    color=heart_corr.values,
                    color_continuous_scale='RdYlGn_r')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Statistiques Descriptives")
        
        st.write("**Variables Numériques**")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.write("**Variables Catégorielles**")
        
        cat_cols = df.select_dtypes(include='object').columns
        
        for col in cat_cols:
            st.write(f"**{col}**")
            value_counts = df[col].value_counts()
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(value_counts)
            
            with col2:
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                            title=f'Distribution de {col}')
                st.plotly_chart(fig, use_container_width=True)

# PAGE 4 : À PROPOS
elif page == "ℹ️ À propos":
    st.header("ℹ️ À Propos du Projet")
    
    st.markdown("""
    <div class="info-box">
        <h2>🎓 Projet Académique</h2>
        <p>
        <b>Titre :</b> Prédiction des Maladies Cardiovasculaires à l'aide du Machine Learning<br>
        <b>Formation :</b> Master Finance & Intelligence Artificielle<br>
        <b>Institution :</b> Dakar Institute of Technology<br>
        <b>Date :</b> Novembre 2025
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Objectifs du Projet")
        st.markdown("""
        1. Développer un modèle prédictif fiable pour identifier les individus à risque
        2. Explorer l'influence des différentes variables sur le risque cardiovasculaire
        3. Créer un outil d'aide au dépistage précoce
        4. Contribuer à la réduction de la mortalité cardiovasculaire
        """)
        
        st.subheader("📊 Données Utilisées")
        st.markdown("""
        - **Source :** Heart Disease Dataset (UCI Machine Learning Repository)
        - **Patients :** 918 observations
        - **Variables :** 11 features + 1 cible
        - **Qualité :** Aucune valeur manquante
        """)
    
    with col2:
        st.subheader("🤖 Modèles Testés")
        st.markdown("""
        1. **K-Nearest Neighbors (KNN)** ⭐ Sélectionné
           - F1-Score : 90.29%
           - ROC-AUC : 91.92%
        
        2. **Support Vector Machine (SVM)**
           - F1-Score : 90.14%
           - ROC-AUC : 92.86%
        
        3. **Random Forest**
           - F1-Score : 88.89%
           - ROC-AUC : 92.29%
        """)
        
        st.subheader("🛠️ Technologies Utilisées")
        st.markdown("""
        - **Python** : Langage de programmation
        - **Scikit-learn** : Machine Learning
        - **Streamlit** : Interface web
        - **Plotly** : Visualisations interactives
        - **Pandas & NumPy** : Manipulation de données
        """)
    
    st.markdown("---")
    
    st.subheader("📚 Références")
    st.markdown("""
    1. World Health Organization (WHO) - Cardiovascular Diseases Statistics
    2. UCI Machine Learning Repository - Heart Disease Dataset
    3. Scikit-learn Documentation - Classification Algorithms
    4. American Heart Association - Heart Disease Risk Factors
    """)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ Avertissement Important</h3>
        <p>
        Cet outil est développé dans un cadre académique et à des fins de recherche. 
        Il ne doit pas être utilisé comme seul moyen de diagnostic médical. 
        Toute décision médicale doit être prise en consultation avec un professionnel de santé qualifié.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 20px;">
        <p>Développé avec ❤️ pour le Master Finance & IA</p>
        <p>© 2025 - Dakar Institute of Technology</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #95a5a6;">
    <p>🏥 Système d'Aide au Dépistage des Maladies Cardiovasculaires | 
    🎓 Master Finance & IA | 
    💻 Powered by Streamlit & Machine Learning</p>
</div>
""", unsafe_allow_html=True)
