import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import json
import random
import joblib
model_random_forest = joblib.load('Random_Forest.joblib')
langue="en"
# Load the pre-trained chatbot model and data
def load_data(language):
    if language=="fr": 
        model_fr =joblib.load('ENOVA_BOT_fr.joblib')   
        intents_fr = json.loads(open('intents-fr.json', encoding='utf-8').read())
        with open('mots.pkl', 'rb') as f:
            mots = pickle.load(f)    
        with open('classes-fr.pkl', 'rb') as f:
            classes_fr = pickle.load(f)
        return model_fr, intents_fr, mots, classes_fr
    if language=="en":
        model_ang =joblib.load('ENOVA_BOT_ang.joblib') 
        intents_ang = json.loads(open('intents-ang.json', encoding='utf-8').read())
        with open('classes-ang.pkl', 'rb') as f:
            classes_ang = pickle.load(f)
        with open('words.pkl', 'rb') as f:
            words = pickle.load(f)
        return model_ang, intents_ang, words, classes_ang

model, intents, words, classes = load_data(langue)

# Text preprocessing functions
def clean_up_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, langue):
    model, intents, words, classes = load_data(langue)  # Charger les données en fonction de la langue sélectionnée
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Intent prediction function
def predict_intent(sentence, model,langue):
    model, intents, words, classes = load_data(langue)  # Charger les données en fonction de la langue sélectionnée
    p = bow(sentence, words,langue)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    if not return_list:
        return_list.append({"intent": "noanswer", "probability": str(1.0)})
    return return_list

def predict_organecode(region_input, age_input, sexe_input, g_input, p_input,
                      menopause_input, antecedentsfamiliauxcancer_input, penicilline_input, tabac_input, alcool_input):
    # categorical values
    region_values = ['AUTRE','Béni Mellal-Khénifra','Casablanca-Settat','Drâa-Tafilalet','Eddakhla-Oued Eddahab','Fès-Meknès','Guelmim-Oued Noun','Laayoune-Sakia El Hamra',
            'Marrakech-Safi','Oriental','Rabat','Rabat-Salé-Kénitra','Souss-Massa','Tanger-Tetouan-Al Hoceima']
    tabac_values = ['ACTIF','INCONNU','NON', 'SEVRAGE_EN_COURS', 'SEVRE']
    alcool_values = ['ACTIF','INCONNU','NON', 'SEVRAGE_EN_COURS', 'SEVRE']
    antecedentsfamiliauxcancer_values = ['INCONNU', 'NON', 'OUI']
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
    
    # Créer une DataFrame avec les valeurs saisies
    new_data = pd.DataFrame({
        "region": [region_input],
        "Age": [age_input],
        "sexe": [sexe_input],
        "g": [g_input],
        "p": [p_input],
        "menopause": [menopause_input],
        "antecedentsfamiliauxcancer":[antecedentsfamiliauxcancer_input],
        "penicilline":[penicilline_input],
        "tabac": [tabac_input],
        "alcool": [alcool_input]
    })

    # Map the user-entered values to their encoded values
    new_data['menopause'] = new_data['menopause'].map(menopause_mapping)
    new_data['region'] = new_data['region'].map(region_mapping)
    new_data['sexe'] = new_data['sexe'].map(sexe_mapping)
    new_data['tabac'] = new_data['tabac'].map(tabac_mapping)
    new_data['alcool'] = new_data['alcool'].map(alcool_mapping)
    new_data['penicilline'] = new_data['penicilline'].map(penicilline_mapping)
    new_data['antecedentsfamiliauxcancer'] = new_data['antecedentsfamiliauxcancer'].map(antecedents_mapping)

    # Utiliser le modèle pour prédire la colonne "organecode" pour les nouvelles données
    predicted_organecode = model_random_forest.predict(new_data) 
    return predicted_organecode

def get_response(intents, intent_name,langue):
    model, intents, words, classes = load_data(langue)  # Charger les données en fonction de la langue sélectionnée
    data_collector={}
    response = ""  # Initialiser la variable response en dehors de la condition
    if intent_name == "cancer_diagnosis":
        # Collect required information
        if "data" not in data_collector:
            data_collector["data"] = {}
        for attribute in ["region", "Age", "sexe", "g", "p", "menopause", "antecedentsfamiliauxcancer", "penicilline", "tabac", "alcool"]:
            if attribute not in data_collector["data"]:
                user_response = input(f"Chatbot: Please provide {attribute}: ")
                data_collector["data"][attribute] = user_response
        # Call predict_organecode() with the collected data
        predicted_organecode_result = predict_organecode(
            data_collector["data"]["region"],
            data_collector["data"]["Age"],
            data_collector["data"]["sexe"],
            data_collector["data"]["g"],
            data_collector["data"]["p"],
            data_collector["data"]["menopause"],
            data_collector["data"]["antecedentsfamiliauxcancer"],
            data_collector["data"]["penicilline"],
            data_collector["data"]["tabac"],
            data_collector["data"]["alcool"]
        )

        # Include the prediction in the response
        response = f"ENOVA BOT: Votre prédiction pour le diagnostic du cancer - organecode est {predicted_organecode_result}"
    else:
        for intent in intents['intents']:
            if intent['tag'] == intent_name:
                responses = intent['responses']
                response = random.choice(responses)
                break
    return response


# Chatbot response function
def chatbot_response(msg, langue):
    model, intents, words, classes = load_data(langue)  # Charger les données en fonction de la langue sélectionnée
    intents_list = predict_intent(msg, model,langue)
    top_intent = intents_list[0]['intent']
    response = get_response(intents, top_intent,langue)
    return response

