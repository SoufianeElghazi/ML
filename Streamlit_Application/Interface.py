from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import requests
from streamlit_lottie import st_lottie
import plotly.express as px
import joblib
from ENOVA_BOT import *
import streamlit.components.v1 as components
from dataclasses import dataclass
from typing import Literal

#---------------------------------------------------------------------------------------------------#
# ----------------------------------Page Configuration:---------------------------------------------#
#---------------------------------------------------------------------------------------------------#
# Set the custom theme using set_page_config
st.set_page_config(
    page_title="ENOVA RT diagnosis",
    page_icon=":star:",
    layout="wide",
    initial_sidebar_state="expanded",
)

#---------------------------------------------------------------------------------------------------#
# ----------------------------------Functions to use:-----------------------------------------------#
#---------------------------------------------------------------------------------------------------#
# Load the model (this will be executed only once)
@st.cache_data(hash_funcs={pickle._Pickler: lambda x: None})
def load_model():
    # Load the model
    model_random_forest = joblib.load('Random_Forest.joblib')
    return model_random_forest
model_random_forest=load_model()

@st.cache_data(hash_funcs={pickle._Pickler: lambda x: None})
def read_data():
    data =pd.read_excel("Dataset.xlsx")
    return data
data=read_data()

@st.cache_data(hash_funcs={pickle._Pickler: lambda x: None})
def preprocess_data(data):
    data.drop(['penicilline.1','cannabisme','iode','OrganeLabel'], axis=1, inplace=True)
    # Remplacez les valeurs manquantes par la moyenne pour la variable numérique age
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    # Supprimer les lignes avec des valeurs manquantes dans la colonne 'Type histologique'
    data.dropna(subset=['Type histologique'], inplace=True)
    data.loc[(data['sexe'] == 'M') & (data['menopause'].isnull()), 'menopause'] = 'Masculin'
    data.loc[(data['sexe'] == 'M') & (data['p'].isnull()), 'p'] = 0
    data.loc[(data['sexe'] == 'M') & (data['g'].isnull()), 'g'] = 0
    data['penicilline'].fillna("NON", inplace=True)
    data.loc[(data['Age'] >=65) & (data['menopause']=='NON'), 'menopause'] = 'OUI'
    categorical_columns = ['menopause', 'antecedentsfamiliauxcancer', 'tabac', 'alcool']
    for col in categorical_columns:
        if col == 'menopause':
            # Remplacer les valeurs vides par "OUI" si l'âge est supérieur à 55, sinon par "NON"
            data[col].fillna(data.apply(lambda row: 'OUI' if row['Age'] > 55 else 'NON', axis=1), inplace=True)
        else:
            mode_value = data[col].mode().iloc[0]  # Obtenir le mode
            data[col].fillna(mode_value, inplace=True)  # Remplacer par le mode
    # Liste des index des lignes à supprimer
    index_to_drop = [3468, 2693, 4446, 4216]
    # Supprimer les lignes avec les index spécifiés
    data.drop(index_to_drop, inplace=True)
    # Traitement des valeurs manquantes pour les colonnes numériques 'g' et 'p'
    data['g'].fillna(int(data['g'].mean()), inplace=True)  # Remplacer par la moyenne
    data['p'].fillna(int(data['p'].mean()), inplace=True)  # Remplacer par la moyenne
    return data
data=preprocess_data(data)

@st.cache_data(hash_funcs={pickle._Pickler: lambda x: None})
def prepared_data(data):
    # Appliquer la normalisation min-max aux colonnes numériques
    numerical_columns = ['Age', 'p', 'g']
    scaler = MinMaxScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    # Créer un objet LabelEncoder pour chaque colonne catégorielle
    label_encoders = {}
    for col in ['region','sexe','Type histologique', 'menopause', 'antecedentsfamiliauxcancer', 'penicilline', 'tabac', 'alcool']:
        label_encoder = LabelEncoder()
        data[col] = label_encoder.fit_transform(data[col])
        label_encoders[col] = label_encoder

    X = data.drop(['organecode'], axis=1)
    y = data['organecode']
    # Utiliser RandomOverSampler pour réaliser le sur-échantillonnage
    oversampler = RandomOverSampler(random_state=42)
    X_oversampler, y_oversampler = oversampler.fit_resample(X, y)

    # Réassembler les données sous forme de DataFrame pandas
    data_over = pd.concat([pd.DataFrame(X_oversampler, columns=X.columns), pd.Series(y_oversampler, name='organecode')], axis=1)
    X_over = data_over.drop('organecode', axis=1)
    y_over = data_over['organecode']
    #ceci est pour tous les colonnes 
    X_all_train, X_all_test, Y_all_train, Y_all_test = train_test_split(X_over, y_over, test_size=0.3, random_state=42)
    return X_all_train, X_all_test, Y_all_train, Y_all_test,label_encoders
X_all_train, X_all_test, Y_all_train, Y_all_test,label_encoders=prepared_data(data)
predictions_random_forest = model_random_forest.predict(X_all_test)

# ---- LOAD ASSETS ----
@st.cache_data(hash_funcs={pickle._Pickler: lambda x: None})
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_hospital = load_lottieurl("https://lottie.host/9c32ab86-e31d-47c3-b346-52a4e96d4791/1WvCqbnsRF.json")
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
lottie_prediction=load_lottieurl("https://lottie.host/a6130783-278a-47fb-b142-54a4ed7883c2/uTsVY92tex.json")
lottie_predect=load_lottieurl("https://lottie.host/58619992-5534-434d-98d7-72f4d3c41a44/CU5y7VF0xB.json")
lottie_analysis=load_lottieurl("https://lottie.host/a3971e0f-4347-4fe8-bcbc-93670edbe48a/uB9wFZioHw.json")
lottie_analyitics=load_lottieurl("https://lottie.host/7906aa39-73db-44d3-ae2e-a20b5f9c593b/uYiOm3MgZA.json")

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style/style.css")
# Load Animation
animation_symbol = "❄"
st.markdown(
    f"""
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    """,
    unsafe_allow_html=True,
)
#---------------------------------------------------------------------------------------------------#
# -------------------------------------------Header:------------------------------------------------#
#---------------------------------------------------------------------------------------------------#
        # 2. horizontal menu with custom style
selected = option_menu(
    menu_title=None,  # required
    options=["Home", "Analysis", "Prédictions","ENOVA BOT"],  
    icons=["house", "bar-chart", "book","robot"],  
    menu_icon="cast",  # optional
    default_index=0,  # optional
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#e5e90a"},
        "icon": {"color": "#ff7c01", "font-size": "25px"},
        "nav-link": {
            "font-size": "25px",
            "text-align": "left",
            "margin": "0px",
            "--hover-color": "#39265e",
            "color": "black",  # Add black text color
        },
        "nav-link-selected": {"background-color": "#7862a0"},
    },
        )


#---------------------------------------------------------------------------------------------------#
# -------------------------------------------ENOVA BOT:--------------------------------------------------#
#---------------------------------------------------------------------------------------------------#
if selected == "ENOVA BOT":
    st.header("Chat with ENOVA BOT ")
    st.markdown("""---""")
    with st.sidebar:
        langue='en'
        langues = ['fr', 'en']
        langue = st.selectbox("Langue?", langues, index=1)
    @dataclass
    class Message:
        """Class for keeping track of a chat message."""
        origin: Literal["human", "ai"]
        message: str

    def initialize_session_state():
        if "history" not in st.session_state:
            st.session_state.history = []
        if "token_count" not in st.session_state:
            st.session_state.token_count = 0
        if "model" not in st.session_state:
            # Chargez le modèle, les intents, les mots, et les classes à partir de ENOVA_BOT.py
            model, intents, words, classes = load_data(langue)
            st.session_state.model = model
            st.session_state.intents = intents
            st.session_state.words = words
            st.session_state.classes = classes
    def on_click_callback():
        human_prompt = st.session_state.human_prompt
        # Utilisez la fonction chatbot_response pour obtenir la réponse du chatbot
        ai_response = chatbot_response(human_prompt, langue)

        # Ajoutez le message de l'utilisateur à l'historique s'il n'est pas vide
        if human_prompt.strip():
            st.session_state.history.append(Message("human", human_prompt))
            st.session_state.history.append(Message("ai", ai_response))
        # Réinitialisez le champ de texte
        st.session_state.human_prompt = ""
    initialize_session_state()
    st.title("Hello I am ENOVA BOT 🤖")
    chat_placeholder = st.container()
    prompt_placeholder = st.form("chat-form")
    #credit_card_placeholder = st.empty()

    with chat_placeholder:
        for chat in st.session_state.history:
            emoji = "🤖" if chat.origin == 'ai' else "🧑"
            div = f"""
    <div class="chat-row {'' if chat.origin == 'ai' else 'row-reverse'}">
        <span class="chat-emoji">{emoji}</span>
        <div class="chat-bubble
        {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
            &#8203;{chat.message}
        </div>
    </div>
            """
            st.markdown(div, unsafe_allow_html=True)
        
        for _ in range(3):
            st.markdown("")

    with prompt_placeholder:
        st.markdown("**ENOVA BOT**")
        cols = st.columns((6, 1))
        chat_input = cols[0].text_input(
            "Chat",
            label_visibility="collapsed",
            key="human_prompt",
        value=st.session_state.get("human_prompt", ""),
    )

        submit_button = cols[1].form_submit_button(
            "Submit", 
            type="primary",
            on_click=on_click_callback,
        )

    components.html("""
    <script>
    const streamlitDoc = window.parent.document;

    const buttons = Array.from(
        streamlitDoc.querySelectorAll('.stButton > button')
    );
    const submitButton = buttons.find(
        el => el.innerText === 'Submit'
    );

    streamlitDoc.addEventListener('keydown', function(e) {
        switch (e.key) {
            case 'Enter':
                submitButton.click();
                break;
        }
    });
    </script>
    """, 
        height=0,
        width=0,
    )
    st.markdown("""---""")
    st.write('<hr style="border: 1px solid black; margin-top: 20px; margin-bottom: 20px;">', unsafe_allow_html=True)
    st_lottie(lottie_hospital, height=300, key="Hospital")
       #---------------------------------------------------------------------------------------------------#
else:

    st.header("Predict medical diagnoses using data from the ENOVA hospital information system.")
    st.markdown("""---""")
    image = Image.open("images/ENOVA&RT.jpg")
    st.image(image, caption='Enova RT', use_column_width=True, width=50)

    #---------------------------------------------------------------------------------------------------#
    # -------------------------------------------Home:--------------------------------------------------#
    #---------------------------------------------------------------------------------------------------#
    if selected == "Home":

        #img_contact_form = Image.open("images/yt_contact_form.png")
        #img_lottie_animation = Image.open("images/yt_lottie_animation.png")

        left_column, right_column = st.columns((2,1))
        with left_column:
            # Create a two-column layout
            col1, col2 = st.columns(2)

            # Auteur in col1
            with col1:
                st.write('<div style="display: flex; flex-direction: column; align-items: center;">', unsafe_allow_html=True)
                st.write('## Auteur')
                st.write('Créer Par :')
                image_author = Image.open('images/Soufiane.jpeg')
                st.image(image_author, width=100)
                st.write('Elghazi Soufiane')
                st.write('</div>', unsafe_allow_html=True)

            # Encadrant in col2
            with col2:
                st.write('<div style="display: flex; flex-direction: column; align-items: center;">', unsafe_allow_html=True)
                st.write('## Encadrant')
                st.write('Encadré Par :')
                image_enc = Image.open('images/Encadrant.jpg')
                st.image(image_enc, width=100)
                st.write('Abou Bakr Najdi')
                st.write('</div>', unsafe_allow_html=True)
        with right_column:
            st_lottie(lottie_hospital, height=300, key="Hospital")

        # Add a line to mark the end of the header and the beginning of the page's body
        st.write('<hr style="border: 1px solid black; margin-top: 20px; margin-bottom: 20px;">', unsafe_allow_html=True)
        st.title(":house: Dataset :")
        # Création d'une zone de défilement
        with st.container():
            # Affichage du DataFrame
            st.dataframe(data, width=700, height=400)
        st.markdown("""---""")
        st.sidebar.subheader("Hi, This is ENOVA&RT  :wave:")
        st.sidebar.title("ENova &RT :hospital:")
        st.sidebar.write(
        """<span style='color: black; font-weight: bold;'>ENOVA R&T is the leading provider of IT solutions specializing in hospital information systems 
        (HIS) in Morocco. Its software solutions are designed to optimize hospital procedures, improve patient journeys and 
        optimize the resources of hospital structures. With almost 20 years of experience, ENOVA R&T offers a wide range of
        services, including digital transformation, training and ongoing support, and data management. With a strong 
        commitment to customer satisfaction and innovation, ENOVA R&T is the ideal partner for hospitals looking to 
        optimize their management operations and improve patient outcomes. The company ensures that the hospital 
        structures using its solutions can get the most out of their technology investment.</span>""",
        unsafe_allow_html=True
    )
        st.sidebar.write("[Learn More >](https://web.facebook.com/enova.ma/?locale=fr_FR&_rdc=1&_rdr)")
        st.markdown("""---""") 
        st_lottie(lottie_coding, height=150, key="coding")
    
    #---------------------------------------------------------------------------------------------------#
    # -----------------------------------------Analysis:------------------------------------------------#
    #---------------------------------------------------------------------------------------------------#
    if selected=='Analysis':
        with st.sidebar:       
            st_lottie(lottie_analyitics, height=100, key="analytics")
        st.sidebar.markdown("""---""")
        st.sidebar.header("Please Filter Here:")
        region = st.sidebar.multiselect(
            "Select the City:",
            options=data["region"].unique(),
            default=[]  # Initialize with an empty list
        )
        organecode = st.sidebar.multiselect(
            "Select the organe code:",
            options=data["organecode"].unique(),
            default=[] # Initialize with an empty list
        )
        sexe = st.sidebar.multiselect(
            "Select the Gender:",
            options=data["sexe"].unique(),
            default=[]  # Initialize with an empty list
        )
        tabac = st.sidebar.multiselect(
            "Select the tabac situation:",
            options=data["tabac"].unique(),
            default=[]  # Initialize with an empty list
        )
        alcool = st.sidebar.multiselect(
            "Select the alcool situation:",
            options=data["alcool"].unique(),
            default=[]  # Initialize with an empty list
        )
        menopause = st.sidebar.multiselect(
            "Select the menopause situation:",
            options=data["menopause"].unique(),
            default=[]  # Initialize with an empty list
        )
        antecedentsfamiliauxcancer = st.sidebar.multiselect(
            "Select the antecedent familiauxcancer situation:",
            options=data["antecedentsfamiliauxcancer"].unique(),
            default=[]  # Initialize with an empty list
        )
        
        df_selection = data.query(
        "region == @region & organecode ==@organecode & sexe == @sexe & tabac ==@tabac & alcool ==@alcool & menopause ==@menopause & antecedentsfamiliauxcancer ==@antecedentsfamiliauxcancer")

            # ---- MAINPAGE ----
        st.title(":bar_chart: Diagnosis Dashboard")
        st.markdown("##")

        # TOP KPI's
        sexe_counts = data['sexe'].value_counts()
        nombre_femmes = sexe_counts.get('F', 0)
        nombre_hommes = sexe_counts.get('M', 0)
        nombre_total_individus = len(data)
        pourcentage_hommes = (nombre_hommes / nombre_total_individus) * 100
        pourcentage_femmes = (nombre_femmes / nombre_total_individus) * 100

        Hommes = f":boy: {pourcentage_hommes:.2f}%"
        Femmes = f":girl: {pourcentage_femmes:.2f}%"
        total_patients = int(data.shape[0])
    
        left_column, middle_column, right_column = st.columns(3)
        with left_column:
            st.subheader("Total Patient:")
            st.subheader(f" {total_patients:,}")
        with middle_column:
            st.subheader("Male:")
            st.subheader(f"{Hommes}")
        with right_column:
            st.subheader("Female:")
            st.subheader(f" {Femmes}")
        st.markdown("""---""")

        #-------------------------------------- bar chart------------------------------------------#

        # Calculate the counts of each unique "organecode"
        organecode_counts = data['organecode'].value_counts()
        fig_organecode_counts = px.bar(
            x=organecode_counts.index,
            y=organecode_counts.values,
            title="<b>Occurrences of Each Organecode</b>",
            color_discrete_sequence=["yellow"] * len(organecode_counts),
            template="plotly_white",
        )
        # Customize the layout
        fig_organecode_counts.update_layout(
            plot_bgcolor="#a8470a",  # Set the plot background color to transparent
            xaxis_title="Organecode",
            yaxis_title="Occurrences",
            xaxis_tickangle=-45,  # Rotate the x-axis labels for better readability
            font=dict(size=14),   # Set the font size for the entire chart
            autosize=True,        # Automatically adjust the chart size to the container
            width=1000, 
        )
        # Display the interactive chart
        st.plotly_chart(fig_organecode_counts, use_container_width=True)
        st.markdown("""---""")

        #--------------------------------------- Pie chart-------------------------------------------#
        left_column, middle_column, right_column = st.columns(3)

        with left_column:
            st.subheader("Alcool:")
            alcool_counts = data["alcool"].value_counts()
            fig_pie_chart1 = px.pie(alcool_counts, names=alcool_counts.index, values=alcool_counts.values)
            fig_pie_chart1.update_traces(textposition='inside', textinfo='percent+label', hole=0.4)
            fig_pie_chart1.update_layout(width=200, height=200, margin=dict(l=0, r=0, b=0, t=30))  # Adjust width and height
            fig_pie_chart1.update_layout(showlegend=False)  # Hide the legend to save space
            fig_pie_chart1.update_layout(font=dict(size=10))  # Customize font size
            st.plotly_chart(fig_pie_chart1)

        with middle_column:
            st.subheader("Tabac:")
            tabac_counts = data["tabac"].value_counts()
            fig_pie_chart2 = px.pie(tabac_counts, names=tabac_counts.index, values=tabac_counts.values)
            fig_pie_chart2.update_traces(textposition='inside', textinfo='percent+label', hole=0.4)
            fig_pie_chart2.update_layout(width=200, height=200, margin=dict(l=0, r=0, b=0, t=30))  # Adjust width and height
            fig_pie_chart2.update_layout(showlegend=False)  # Hide the legend to save space
            fig_pie_chart2.update_layout(font=dict(size=10))  # Customize font size
            st.plotly_chart(fig_pie_chart2)

        with right_column:
            st.subheader("Menopause:")
            menopause_counts = data["menopause"].value_counts()
            fig_pie_chart3 = px.pie(menopause_counts, names=menopause_counts.index, values=menopause_counts.values)
            fig_pie_chart3.update_traces(textposition='inside', textinfo='percent+label', hole=0.4)
            fig_pie_chart3.update_layout(width=200, height=200, margin=dict(l=0, r=0, b=0, t=30))  # Adjust width and height
            fig_pie_chart3.update_layout(showlegend=False)  # Hide the legend to save space
            fig_pie_chart3.update_layout(font=dict(size=10))  # Customize font size
            st.plotly_chart(fig_pie_chart3)
        st.markdown("""---""")

        st_lottie(lottie_analysis, height=150, key="analysis")

    #---------------------------------------------------------------------------------------------------#
    # ---------------------------------------Prédictions:-----------------------------------------------#
    #---------------------------------------------------------------------------------------------------#
    if selected=='Prédictions':

        with st.expander("Predicted Results"):
            # Check if the "View" button is clicked and the "Hide" button is not clicked
            # Predict the organecode for the test set
            predicted_organecodes = model_random_forest.predict(X_all_test)
            # Create a new DataFrame with the predicted organecode values
            results_df = pd.DataFrame({
                'Actual Organecode': Y_all_test,
                'Predicted Organecode': predicted_organecodes
            })
                # Display the table
            st.dataframe(results_df,width=300)   
        st.markdown("""---""")
        st.write('<hr style="border: 1px solid black; margin-top: 20px; margin-bottom: 20px;">', unsafe_allow_html=True)

        st.title(":book: Prédiction:")
        # Define a DataFrame to store the data
        all_data = pd.DataFrame()
        # Create a form to group input fields together
        with st.sidebar:
            with st.form("input_form"):       
                menopause_values = ['NON', 'Masculin', 'OUI', 'INCONNU']
                menopause_input = st.selectbox("Ménopause?", menopause_values, index=1)
            
                region_values = ['AUTRE','Béni Mellal-Khénifra','Casablanca-Settat','Drâa-Tafilalet','Eddakhla-Oued Eddahab','Fès-Meknès','Guelmim-Oued Noun','Laayoune-Sakia El Hamra',
                'Marrakech-Safi','Oriental','Rabat','Rabat-Salé-Kénitra','Souss-Massa','Tanger-Tetouan-Al Hoceima']
                region_input = st.selectbox("Choisissez une région", region_values, index=1)
            
                sexe_values = ['F', 'M']
                sexe_input = st.selectbox("Votre genre:", sexe_values, index=1)
            
                tabac_values = ['ACTIF','INCONNU','NON', 'SEVRAGE_EN_COURS', 'SEVRE']
                tabac_input = st.selectbox("Vous fumer?:", tabac_values, index=1)
                
                alcool_values = ['ACTIF','INCONNU','NON', 'SEVRAGE_EN_COURS', 'SEVRE']
                alcool_input = st.selectbox("Vous buvez?:", alcool_values, index=1)
                
                penicilline_values = ['NON', 'OUI']
                penicilline_input = st.selectbox("penicilline?:", penicilline_values, index=1)
                
                antecedentsfamiliauxcancer_values = ['INCONNU', 'NON', 'OUI']
                antecedentsfamiliauxcancer_input = st.selectbox("antecedents familiaux cancer?:", antecedentsfamiliauxcancer_values, index=1)
                
                Type_histologique_values = ['8000/1', '8000/3', '8003/3', '8005/0', '8012/3', '8013/3', '8020/3', '8041/3', '8046/3', '8051/3', '8052/3', '8070/2', '8070/3', '8071/2', '8071/3', '8076/3', '8077/0', '8077/2', '8081/2', '8082/3', '8090/3', '8140/2', '8140/3', '8147/3', '8160/3', '8170/3', '8200/3', '8201/3', '8211/3', '8221/1', '8240/3', '8244/3', '8246/3', '8260/3', '8263/3', '8312/3', '8316/3', '8317/3', '8318/3', '8319/3', '8330/1', '8330/3', '8370/3', '8380/1', '8380/3', '8384/4', '8430/3', '8442/1', '8460/3', '8470/2', '8470/3', '8472/1', '8474/3', '8480/3', '8490/3', '8500/2', '8500/3', '8502/3', '8504/3', '8509/3', '8510/3', '8520/3', '8525/3', '8560/3', '8575/3', '8580/3', '8584/1', '8586/3', '8620/3', '8700/3', '8720/3', '8800/3', '8801/3', '8803/3', '8804/3', '8805/3', '8806/3', '8810/3', '8811/3', '8832/3', '8840/3', '8850/0', '8850/3', '8890/3', '8900/3', '8931/3', '8933/3', '8936/3', '8941/3', '8980/3', '8982/3', '9020/3', '9040/3', '9044/3', '9060/3', '9061/3', '9063/3', '9070/3', '9071/3', '9080/0', '9080/3', '9085/3', '9100/0', '9100/3', '9120/3', '9140/3', '9180/3', '9231/3', '9260/3', '9380/3', '9382/3', '9391/3', '9400/3', '9440/3', '9450/3', '9470/3', '9500/3', '9508/3', '9530/3', '9540/3', '9591/3', '9597/3', '9650/3', '9680/3', '9699/3', '9702/3', '9712/3', '9719/3', '9731/3', '9823/3', '9930/3', '9948/3']
                Type_histologique_input = st.selectbox("Type histologique?:", Type_histologique_values, index=1)

                age_input = st.number_input("Entrez votre âge", value=30, min_value=1, max_value=150)
                g_input = st.number_input("Entrez le nombre de Grossesses", value=0, min_value=0, max_value=39)
                p_input = st.number_input("Entrez la nombre de naissances", value=0, min_value=0, max_value=39)
                # Add the "Confirm" button to the form
                submitted = st.form_submit_button("Confirm")

        # Check if the form has been submitted
        if submitted:
            # Place the code that should run after the "Confirm" button is clicked here
            # Mapping for 'menopause'
            menopause_mapping = {'NON': 2, 'Masculin': 1, 'OUI': 3, 'INCONNU': 0}
            # Mapping for 'region'
            region_mapping = {value: index for index, value in enumerate(region_values)}
            # Mapping for 'sexe'
            sexe_mapping = {'F': 0, 'M': 1}
            # Mapping for 'tabac'
            tabac_mapping = {value: index for index, value in enumerate(tabac_values)}
            # Mapping for 'alcool'
            alcool_mapping = {value: index for index, value in enumerate(alcool_values)}
            # Mapping for 'penicilline'
            penicilline_mapping = {'NON': 0, 'OUI': 1}
            # Mapping for 'antecedentsfamiliauxcancer'
            antecedents_mapping = {value: index for index, value in enumerate(antecedentsfamiliauxcancer_values)}
            # Mapping for 'Type histologique'
            Type_histologique_mapping = {Type_histologique_values[i]: i for i in range(len(Type_histologique_values))}
            # Créer une DataFrame avec les valeurs saisies
            new_data = pd.DataFrame({
                "region": [region_input],
                "Age": [age_input],
                "sexe": [sexe_input],
                "Type histologique":[Type_histologique_input],
                "g": [g_input],
                "p": [p_input],
                "menopause": [menopause_input],
                "antecedentsfamiliauxcancer":[antecedentsfamiliauxcancer_input],
                "penicilline":[penicilline_input],
                "tabac": [tabac_input],
                "alcool": [alcool_input]
            
            })

            # After the user enters the data in the Streamlit app
            # Map the user-entered values to their encoded values
            new_data['menopause'] = new_data['menopause'].map(menopause_mapping)
            new_data['region'] = new_data['region'].map(region_mapping)
            new_data['sexe'] = new_data['sexe'].map(sexe_mapping)
            new_data['tabac'] = new_data['tabac'].map(tabac_mapping)
            new_data['alcool'] = new_data['alcool'].map(alcool_mapping)
            new_data['penicilline'] = new_data['penicilline'].map(penicilline_mapping)
            new_data['antecedentsfamiliauxcancer'] = new_data['antecedentsfamiliauxcancer'].map(antecedents_mapping)
            new_data['Type histologique'] = new_data['Type histologique'].map(Type_histologique_mapping)

            # Utiliser le modèle pour prédire la colonne "organecode" pour les nouvelles données
            predicted_organecode = model_random_forest.predict(new_data)
            # Afficher la prédiction
            st.markdown("Your Diagnosis shows that u may have: <span style='color: red; font-weight: bold;'>{}</span>".format(predicted_organecode), unsafe_allow_html=True)

            # Reverse the encoding for categorical columns in new_data
            for col in ['region', 'sexe', 'Type histologique', 'menopause', 'antecedentsfamiliauxcancer', 'penicilline', 'tabac', 'alcool']:
                label_encoder = label_encoders[col]
                new_data[col] = label_encoder.inverse_transform(new_data[col])

            # Add the predicted organecode as a new column to new_data
            new_data['predicted_organecode'] = predicted_organecode
            # Append the new_data row to all_data DataFrame
            all_data=pd.concat([all_data, new_data], axis=0, ignore_index=True)
            # Création d'une zone de défilement
            with st.container():
                # Affichage du DataFrame
                st.dataframe(all_data)
            st_lottie(lottie_prediction, height=150, key="predictions")

        st.markdown("""---""")
        st.write('<hr style="border: 1px solid black; margin-top: 20px; margin-bottom: 20px;">', unsafe_allow_html=True)
        st_lottie(lottie_predect, height=300, key="predict")

#---------------------------------------------------------------------------------------------------#
# -----------------------------------------Footer:--------------------------------------------------#
#---------------------------------------------------------------------------------------------------#
# Footer content

st.sidebar.markdown("---")
# ---- CONTACT ----
with st.sidebar.container():
    st.sidebar.write("---")
    st.sidebar.header("Get In Touch With Me!")
    st.sidebar.write("##")
    contact_form = """
    <form action="https://formsubmit.co/soufiane.el-ghazi@esi.ac.ma" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.sidebar.columns(2)
    with left_column:
        st.sidebar.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.sidebar.empty()
st.sidebar.write("Contact me: soufiane.el-ghazi@esi.ac.ma")
#---------------------------------------------------------------------------------------------------#
# ------------------------------------------END:----------------------------------------------------#
#---------------------------------------------------------------------------------------------------#