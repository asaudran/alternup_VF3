#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import time


def display_title():
    # Utilisation de HTML et CSS pour personnaliser le style du titre
    st.markdown("""
        <style>
        .title {
            background-color: #AEC6CF;
            color: #ffffff;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            font-size: 2em;
            font-weight: bold;
        }
        </style>
        <div class="title">
            Plateforme interactive d'optimisation de consommation énergétique
        </div>
    """, unsafe_allow_html=True)

# Appeler la fonction pour afficher le titre personnalisé
display_title()


# Fonction pour générer des données simulées
def generate_data():
    rng = pd.date_range(start='2020-01-01', end='2023-12-31 23:00:00', freq='H')
    data = np.random.normal(loc=50, scale=10, size=len(rng))
    hour_mod = [1.5 if 7 <= hr < 19 else 0.7 for hr in rng.hour]
    month_mod = [1.2 if mth in [1, 2, 11, 12] else (1.5 if mth in [3, 4, 9, 10] else 0.8) for mth in rng.month]
    data *= hour_mod
    data *= month_mod
    i = 0
    while i < len(rng):
        if np.random.rand() < 0.05:
            duration = np.random.randint(3, 73)
            spike = np.random.normal(loc=100, scale=30)
            data[i:min(i + duration, len(rng))] += spike
            i += duration
        else:
            i += 1
    data = np.abs(data)
  #  return pd.DataFrame({'Timestamp': rng, 'Consumption': data})
# Ajout de la colonne 'Site' avec des données correspondant aux différents sites
    sites = np.random.choice(["Site A", "Site B", "Site C", "Site D", "Site E"], len(rng))
    return pd.DataFrame({'Timestamp': rng, 'Consumption': data, 'Site': sites})

test_data = generate_data()
print(test_data.head())
print(test_data.columns)

def generate_synthese_data():
    np.random.seed(0)
    # Utilisation de la fréquence 'A' (Annuelle) au lieu de 'D' (Journalière)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='A')
    data_synthese = {
        "Timestamp": dates,
        "Site": np.random.choice(["Site A", "Site B", "Site C", "Site D", "Site E"], len(dates)),
        "Consommation Annuelle (GWh)": np.random.uniform(5, 20, len(dates)).round(2),
        "Catégorie": np.random.choice(["Hyper Electro-Intensif", "Electro-Intensif", "Autres"], len(dates)),
        "Taux d'Abattement (%)": np.random.choice(range(50, 101, 10), len(dates))
    }
    return pd.DataFrame(data_synthese)


def generate_future_data(start_date, end_date):
    rng = pd.date_range(start=start_date, end=end_date, freq='H')
    data = np.random.normal(loc=50, scale=10, size=len(rng))
    hour_mod = [1.5 if 7 <= hr < 19 else 0.7 for hr in rng.hour]
    month_mod = [1.2 if mth in [1, 2, 11, 12] else (1.5 if mth in [3, 4, 9, 10] else 0.8) for mth in rng.month]
    data *= hour_mod
    data *= month_mod
    i = 0
    while i < len(rng):
        if np.random.rand() < 0.05:
            duration = np.random.randint(3, 73)
            spike = np.random.normal(loc=100, scale=30)
            data[i:min(i + duration, len(rng))] += spike
            i += duration
        else:
            i += 1
    data = np.abs(data)
    sites = np.random.choice(["Site A", "Site B", "Site C", "Site D", "Site E"], len(rng))
    return pd.DataFrame({'Timestamp': rng, 'Consumption': data, 'Site': sites})


# Initialiser l'application Streamlit
def main():
    #st.title('Dashboard de consommation énergétique')
    
    # Utilise .get() pour récupérer 'data' ou initialiser avec generate_data() si 'data' n'est pas encore dans session_state
    df = st.session_state.get('data', generate_data())
    if 'data' not in st.session_state:
        st.session_state['data'] = df  # Assurez-vous que la session_state est mise à jour
        print("Data initialized in session state.")
    
    print("Current DataFrame in session state:", df.head())
    
    # Générer les données synthétiques pour la synthèse une seule fois
    synthese_df = generate_synthese_data()
    
    st.sidebar.header('CleverWatt')
    site = st.sidebar.radio('Choisir un site', ['Site A', 'Site B', 'Site C', 'Site D', 'Site E'])
    consumption_unit = st.sidebar.radio("Unité de consommation", ("MWh/h", "MWh/jour"))
    tab1, tab2, tab3, tab4 = st.tabs(["Consommation", "Synthèse", "Prédiction", "Optimisation"])
    selected_year = st.sidebar.selectbox('Choisir une année', [2020, 2021, 2022, 2023])
    with tab1:
        display_consumption_data(df, site, selected_year, consumption_unit)
    with tab2:
        display_synthese_data(synthese_df, site, selected_year)
    with tab3:
        display_prediction_options(df)
    with tab4:
        display_optimization_options()

        
def display_consumption_data(df, site, year, unit):
    st.header(f"Consommation pour {site}")
    # Ajout d'une introduction sur la génération des données
    st.write("C'est une version de démo. L'hébergement en ligne a limité les fonctionnalités. Les données de consommation sont donc générées de manière simulée pour le site sélectionné. Dans le futur, nous prévoyons d'intégrer les données réelles des entreprises partenaires en utilisant les API de RTE pour obtenir des données de consommation en direct.")

    # Filtrer les données pour le site et l'année spécifiés
    filtered_data = df[(df['Site'] == site) & (df['Timestamp'].dt.year == year)]

    # Calculer la consommation totale pour l'affichage
    total_consumption = filtered_data['Consumption'].sum() / 1000
    st.write(f"Consommation totale pour {site} en {year}: {total_consumption} MWh")

    if unit == "MWh/jour":
        # Aggrégation quotidienne
        consumption_stats = filtered_data.set_index('Timestamp').resample('D')['Consumption'].sum() / 1000
    else:
        # Aggrégation horaire, utiliser 'h' pour heure conformément aux dernières normes de pandas
        consumption_stats = filtered_data.groupby(filtered_data['Timestamp'].dt.floor('h'))['Consumption'].mean() / 1000

    # Créer le graphique de consommation
    fig = px.area(consumption_stats, x=consumption_stats.index, y='Consumption',
                  labels={'x': 'Date', 'Consumption': f'Consommation ({unit})'},
                  title=f'Consommation pour l\'année {year} en {unit}',
                  template='plotly_white', color_discrete_sequence=['darkblue'])
    fig.update_traces(fill='tonexty', fillcolor='rgba(135, 206, 250, 0.5)')
    st.plotly_chart(fig, use_container_width=True)

    # Afficher des statistiques supplémentaires si nécessaire
    display_consumption_stats(filtered_data, [2020, 2021, 2022, 2023])

    # Pour les autres sites, indiquer que la visualisation est en développement
    if site != "Site A":
        st.write("Visualisation des données de consommation pour d'autres sites à développer.")


def display_synthese_data(df, site, year):
    st.header("Synthèse pour " + site)
    st.write("""
    Le tableau ci-dessous présente des données fictives d'abattement potentiel. L'objectif de cet onglet est d'avoir un aperçu des droits de réduction du site industriel selon sa consommation.
    """)
    
    # Filtration des données pour le site et l'année sélectionnés
    synthese_data = df[df['Timestamp'].dt.year == year]
    synthese_data = synthese_data[synthese_data['Site'] == site]
    
    if synthese_data.empty:
        st.write("Aucune donnée disponible pour " + site + " en " + str(year))
    else:
        st.dataframe(synthese_data.style.applymap(
            lambda x: f"background-color: {'#ccffcc' if isinstance(x, int) and x >= 50 else ''}", 
            subset=['Taux d\'Abattement (%)']))

        st.write("""
        Il est également possible d"intégrer un suivi des jalons pour l'obtention des réductions :
        1. Dépôt des dossiers de demande.
        2. Création du plan d'efficacité énergétique.
        3. Acceptation du dossier.
        """)

        # Tableau de suivi des jalons
        st.write("Tableau de suivi des actions:")
        milestones = ["Demande", "Dossier", "Retour"]
        completed = st.multiselect("Cochez les étapes complétées :", milestones)

        # Mise à jour du tableau de suivi
        for milestone in milestones:
            if milestone in completed:
                st.write(milestone, ":", "✓", "**Complété**")
            else:
                st.write(milestone, ":", "❌", "**En attente**")


def display_prediction_options(df):
    st.header("Prédiction de la consommation")
    st.write("""
    La prédiction est faite à partir des données passées et des informations utilisateurs. Pour le moment, les calculs ne sont pas disponibles.
    """)
    # Section de maintenance future
    st.subheader('Planification des maintenances futures', anchor=None)
    maintenance_start = st.date_input("Début de la maintenance", key="maintenance_start")
    maintenance_end = st.date_input("Fin de la maintenance", key="maintenance_end")
    maintenance_amount = st.number_input("Montant de réduction de la consommation (en MWh)", 0, 10000, 500, key="maintenance_amount")
    st.info(f"Les prédictions prendront en compte une réduction de {maintenance_amount} MWh pendant la période du {maintenance_start} au {maintenance_end}. Pour le moment, l'application en ligne ne fait pas les calculs.")

    # Section du lancement d'une nouvelle unité
    st.subheader('Lancement d\'une nouvelle unité de production', anchor=None)
    launch_start = st.date_input("Date de lancement", key="launch_start")
    launch_end = st.date_input("Fin de la période de montée en puissance", key="launch_end")
    launch_amount = st.number_input("Augmentation de la consommation estimée (en MWh)", 0, 10000, 500, key="launch_amount")
    st.info(f"Les prédictions prendront en compte une augmentation de {launch_amount} MWh pendant la période du {launch_start} au {launch_end}. Pour le moment, l'application en ligne ne fait pas les calculs.")

    # Couleurs
    st.markdown("""
        <style>
        .stTextInput, .stDateInput {
            background-color: rgb(202, 240, 248);
        }
        </style>
        """, unsafe_allow_html=True)

    future_year = st.selectbox("Choisir l'année de prévision", list(range(2024, 2029)))
    unit = st.radio("Unité de consommation", ("MWh/h", "MWh/jour"), key="prediction_unit")

    if st.button('Prédire la consommation'):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.1)  # Simuler le traitement
            progress_bar.progress(i + 1)
        future_data = generate_future_data(f'{future_year}-01-01', f'{future_year}-12-31 23:00:00')
        progress_bar.empty()

        if future_data.empty:
            st.write("Aucune donnée de prédiction générée pour l'année sélectionnée.")
            return

        if unit == "MWh/jour":
            daily_data = future_data.copy()
            daily_data['Day'] = daily_data['Timestamp'].dt.date
            daily_data = daily_data.groupby('Day').sum().reset_index()
            daily_data.rename(columns={'Day': 'Timestamp', 'Consumption': 'Daily Consumption'}, inplace=True)
            fig = px.line(daily_data, x='Timestamp', y='Daily Consumption', title=f'Prédiction de la consommation pour {future_year} en {unit}')
        else:
            fig = px.line(future_data, x='Timestamp', y='Consumption', title=f'Prédiction de la consommation pour {future_year} en {unit}')

        if fig.data:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Aucune donnée de prévision disponible pour afficher le graphique.")


def calculate_potential_savings(rebate_percentage):
    # Cette fonction doit être implémentée avec la logique de calcul réelle.
    # L'exemple suivant utilise une valeur fictive basée sur le pourcentage d'abattement.
    mock_savings = 10000 * (rebate_percentage / 100.0)
    return round(mock_savings, 2)

def display_optimization_options():
    st.header("Conseils pour optimiser la consommation")

    st.markdown("""
    L'objectif de cet outil est de maximiser les abattements sur les coûts électriques pour les sites industriels en optimisant la consommation d'énergie grâce à une analyse prédictive. Les recommandations suivantes sont basées sur des données fictives et des calculs d'optimisation non implémentés pour le moment.
    """)

    st.subheader("Simulation d'optimisation")
    st.write("""
    Les données sont générées aléatoirement pour le moment. L'objectif de l'application est de proposer des conseils pour maximiser le taux d'abattement
    """)
    
    rebate_percentage = st.slider("Taux d’abattement envisagé (%)", 0, 100, 20)
    potential_savings = calculate_potential_savings(rebate_percentage)
    st.markdown(f"**Gain financier potentiel:** {potential_savings}€ d'économies possibles avec un abattement de {rebate_percentage}%.")

    st.markdown("""
    **Conseils pratiques:**
    * Envisagez de décaler les maintenances ou de les programmer durant les heures creuses.
    * Évaluez l'opportunité de retarder le lancement de nouvelles unités pour coïncider avec les périodes tarifaires les plus avantageuses.
    * Planifiez la consommation pour maintenir les critères d'électro-intensité et bénéficier de réductions significatives sur vos factures.
    """)

    st.subheader("Rapports personnalisés")
    if st.button('Générer rapport'):
        st.success("Une création personnalisée sera disponible à la prochaine version.")

    st.subheader("Définition des alertes de consommation")
    st.markdown("""
    Vous pouvez définir des alertes pour être averti lorsque votre consommation atteint un certain seuil. Ces alertes peuvent vous aider à prendre des mesures immédiates pour réduire votre consommation et maintenir l'efficacité énergétique.
    """)
    alert_threshold = st.number_input("Définir le seuil d'alerte de consommation (kWh)", min_value=0)
    if st.button('Définir alerte'):
        st.success(f"Alerte définie pour une consommation de {alert_threshold} kWh.")


def display_consumption_stats(df, years):
    st.subheader("Statistiques de consommation")
    # Ajout d'une introduction sur les statistiques de consommation
    st.write("Les statistiques de consommation ci-dessous sont générées de manière aléatoire pour l'instant. Dans les futures versions, ces statistiques seront basées sur des calculs réels.")
    for year in years:
        year_data = df[df['Timestamp'].dt.year == year]
        stats_df = pd.DataFrame({
            'Energie Soutirée': [year_data['Consumption'].sum() / 1000],
            'Valeur max. Puissance moyenne glissante sur 24h': [year_data['Consumption'].rolling(24).mean().max() / 1000],
            'Durée d\'utilisation (heures)': [year_data.shape[0]],
            'EHCH': [year_data['Consumption'].max() / 1000],
            'EHCE': [year_data['Consumption'].min() / 1000],
            'Taux utilisation heures creuses': [year_data['Consumption'].quantile(0.25) / 1000]
        })
        st.write(f"Statistiques pour l'année {year}")
        st.write(stats_df)        
        


if __name__ == "__main__":
    main()


# In[ ]:




