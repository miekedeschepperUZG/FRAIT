import pandas as pd
import numpy as np
import ast
from utils.output import execute_query_to_panda, show_panda_nicely
from collections import Counter
import json
import re

def frait_choosen_prompt(ds_vragenlijst, ds_res):
    """
    Toevoegen van de "gekozen" prompt op basis van het antwoord van de user.
    Dit zowel in Nl als Eng voorzien.
    """


    ### koppel de 2 data.frames aan elkaar

    ## Transponeren van de data met email adres als header van de kolommen
    ds_res.set_index('E-mail', inplace=True)
    ds_res_transpose = ds_res.T

    ds_res_transpose.reset_index(inplace=True)
    ds_res_transpose.rename(columns={'index': 'question'}, inplace=True)

    ds_res_transpose['number'] = range(-3,112)


    ## Koppelen van beide op basis van number - LEFT JOIN op de resultaten
    ds = ds_res_transpose.merge(ds_vragenlijst, on='number', how='left')

    ## longformat maken
    df_long = ds.melt(id_vars=['section', 'question', 'number', 'question_type', 'question_label',	'anwers_options', 	'jump_to_question',	'prompt_nl', 	'prompt_eng'],
                      var_name='email',
                      value_name='antwoord')


    ## zoek het antwoord per vraag, pas deze index toe op kolom prompt_eng ,en stockeer dit in nieuwe kolom met naam emailadres-antwoord
    # Apply function to create a new column
    df_long['choosen_prompt_eng'] = df_long.apply(lambda row: get_prompt(row['antwoord'], row['anwers_options'], row['prompt_eng'],  row['number'], "eng"), axis=1)
    df_long['choosen_prompt_nl'] = df_long.apply(lambda row: get_prompt(row['antwoord'], row['anwers_options'], row['prompt_nl'], row['number'], "nl"), axis=1)
    df_long = df_long.explode(['choosen_prompt_eng', 'choosen_prompt_nl'])

    ### sectie toevoegen voor de lege vragen
    df_long.loc[(df_long['section'].isna()) & (df_long['number'] > 110), 'section'] = 'FINAL'
    df_long.loc[(df_long['section'].isna()) & (df_long['number'] < 1), 'section'] = 'BASIS'

    ### Evaluatievragen toevoegen
    df_long = df_long.apply(add_evaluatie_columns, axis=1) #, taal='Nl')
    df_long = df_long.apply(add_evaluatie_columns, axis=1) #, taal='Eng')

    ### specifieke volgorde voor de leesbaarheid van de output
    result = df_long[['email', 'number', 'section','question_type', 'question',  'question_label', 'anwers_options', 'jump_to_question', 'prompt_eng', 'prompt_nl',
    'antwoord', 'choosen_prompt_eng', 'choosen_prompt_nl',
    'evaluatie_vorm_vraag', 'evaluatie_vorm_antwoorden', 'evaluatie_inhoud_vraag', 'evaluatie_inhoud_antwoorden']]

    return(result)






def get_prompt(antwoord, answer_options, prompt_eng, number, taal):
    """
    Prompt vinden die hoort bij het antwoord
    """

    if number == 4:
        ## volgorde specifiek voor vraag 4
        ### vraag 4 = apart te behandelen
        answer_options_list = ast.literal_eval(answer_options)
        prompt_eng_list = ast.literal_eval(prompt_eng)
        if not isinstance(antwoord, str) or antwoord.strip() == "":
            antwoord = ";".join(answer_options_list)
        else:
            ## in vragenlijst manuele aanpassing...
            antwoord = antwoord.replace("Algemeen (bijv. ligduur, opnamedatum, ...)", "Algemeen")

        selected_answers = antwoord.split(";")

        matching_prompts = [prompt_eng_list[answer_options_list.index(ans)]
                            for ans in selected_answers if ans in answer_options_list]
        if taal == "eng":
            prompt_v4 = "I want the section in this order: " + " - ".join(matching_prompts)
        else:
            prompt_v4 = "Ik wil de secties in deze volgorde: " + " - ".join(matching_prompts)
        return prompt_v4


    elif number <4 or number == 111:
        return None
    elif number in (6,7,14,35,39,40,42,45,83,94,95,100,108):
        ## specieke meerkeuze vragen
        # Convert answer_options and prompt_eng from string to list
        try:
            # Convert answer_options and prompt_eng from string to list
            answer_options_list = ast.literal_eval(answer_options)
            prompt_eng_list = ast.literal_eval(prompt_eng)

            ## Opsplitsen in meerdere lijnen
            selected_answers = antwoord.split(";")
            matching_prompts = []
            for ans in selected_answers:
                if ans in answer_options_list:
                    matching_prompts.append(prompt_eng_list[answer_options_list.index(ans)])

            return matching_prompts

        except (ValueError, SyntaxError, IndexError):
            return None
    else:
        ## single select vragen
        try:
            # Find the index of antwoord in answer_options
            if number == 110:
                antwoord = antwoord.replace('"' , '\'')

            answer_options_list = [normalize_text(x) for x in ast.literal_eval(answer_options)]
            prompt_eng_list = ast.literal_eval(prompt_eng)
            antwoord = normalize_text(antwoord)

            index = answer_options_list.index(antwoord)

            # Return the corresponding prompt_eng
            return prompt_eng_list[index]

        except (ValueError, SyntaxError, IndexError):
            return None






def normalize_text(text):
    """
    Normalize text (remove extra spaces and non-breaking spaces)
    """
    if isinstance(text, str):
        return text.replace("\xa0", " ").strip()  # Replace non-breaking spaces with regular spaces
    return text  # Return as is if not a string


def frait_individual_prompt(df):
    """
    Opmakenvan de individuele prmopt op basis van de resultaten.
    Pre- en post tekst hieronder in code voorzien.
    resultaten gecombineerd d.m.v. ;

    #https://browser.icpc-3.info/
    """

    df = df.dropna(subset=["choosen_prompt_eng"])

    ### DEEL 1: individuele ENG prompt
    pre_text_eng = """This text is a medical discharge letter (the report of a hospital admission). I would like a summary in Dutch.
    The intention is that I, as a healthcare provider, can use this letter to properly follow up the patient.
    The summary must be a maximum of one page long and clearly structured.
    Use formal language based on the expertise of a doctor.
    Start with the patient's name, date of birth and the name of the hospital.
    All dates must be in the format day/month/year.
    If a requested part is not in the letter, bundle the missing parts at the bottom per section in one sentence and use 'Not mentioned' as an indication.
    Each section must be clearly separate and in capital letters."""


    post_text_eng = """
We would like to provide some additional explanation:
The name of the sections should be translated like: General = Algemeen,  Medical History = Voorgeschiedenis, Medication = Medicatie, Examinations = Onderzoeken, Admission history = Opnameverloop, Follow-up = Follow-up, Decision = Besluit.
If new diagnoses are requested in the history, this means that you add the new diagnoses that were made during admission to the history.
By interventions we mean operations.
The admission process describes the path that the patient takes during the admission, the evolution of the condition and the steps that are taken during the admission in chronological order. Here we also expect to report the administration of blood transfusions, radiotherapy or chemotherapy, the performance of diagnostic examinations such as biopsies, endoscopies, the placement of feeding tubes, etc.
In Medical imaging / Technical examinations we mean by the protocol all actions that are performed during this examination and all findings that are made in this.
If changes in medication are requested, we mean changes in dose, frequency and form and whether medication has been stopped completely or new medication has been started.
To do this, make a comparison between the medication on admission on the one hand and the medication on discharge and the medication at the therapy proposal on the other.
Medication that has remained the same is not mentioned in the case of changed medication. If the question is about the risk of clotting, mention blood clots/thrombosis and embolisms and the use of blood-thinning medication. Also mention here if you find anything about cerebral infarction/CVA or heart attack/AMIAMI. Clotting disorders should also be mentioned here.
Communication is about discussing the condition and the further policy with the patient and his family.
If a section should not be mentioned, leave it out of the summary completely.
The information in the summary should come from the original letter. An exception to this is if ICD codes or ICPC codes are mentioned.
When asking for ICD codes, the ICD-10 code should be mentioned next to the diagnosis, as can be found at https://icd10be.health.belgium.be/.
When asking for ICPC codes, you can find them at https://browser.icpc-3.info/.
If a patient has come from another hospital, mention this under general.
"""

    result_eng = df.groupby("email", as_index=False)["choosen_prompt_eng"].agg(" ; ".join)
    result_eng['choosen_prompt_eng'] = pre_text_eng + result_eng['choosen_prompt_eng'] + post_text_eng

    result_eng['choosen_prompt_eng'] = result_eng['choosen_prompt_eng'].apply(
        lambda x: re.sub(r'\s*;+\s*', ';', x))
    result_eng['choosen_prompt_eng'] = result_eng['choosen_prompt_eng'].apply(
        lambda x: re.sub(r';+', ';', x))

    ### DEEL 2: individuele NL prompt
    pre_text_nl = """
Deze tekst is een medische ontslagbrief (het verslag van een ziekenhuisopname). Ik wil graag een samenvatting in het Nederlands. 
De bedoeling is dat ik als zorgverlener met deze brief de patiënt goed kan opvolgen. 
De samenvatting moet maximaal één pagina lang zijn en overzichtelijk opgebouwd. 
Gebruik formele taal vanuit de expertise van een arts. 
Begin met de naam van de patiënt, de geboortedatum en de naam van het ziekenhuis. 
Alle datums moeten in het formaat dag/maand/jaar staan.
Als een gevraagd onderdeel niet in de brief staat, bundel dan de ontbrekende onderaan per sectie in één zin en gebruik 'Niet vermeld' als aanduiding. 
Elke sectie dient duidelijk apart te staan en in hoofdletters. 
"""

    post_text_nl = """
We geven graag nog wat extra uitleg: 
Als bij voorgeschiedenis naar nieuwe diagnoses wordt gevraagd, wordt bedoeld dat je de nieuwe diagnoses die tijdens de opname werden gesteld toevoegt aan de voorgeschiedenis. 
Met ingrepen bedoelen we operaties.
Het opnameverloop beschrijft het traject dat de patiënt aflegt binnen de opname, de evolutie van de aandoening en de stappen die binnen de opname worden gezet in chronologische volgorde. Hier verwachten we ook melding van het toedienen van bloedtransfusies, radiotherapie of chemotherapie, het uitvoeren van diagnostische onderzoeken als biopsieën, endoscopieën, het plaatsen van voedingssondes, etc.  
Bij Medische beeldvorming / Technische onderzoeken bedoelen we met het protocol alle handelingen die bij dit onderzoek worden gezet en alle vaststellingen die hierbij worden gedaan. 
Als er gevraagd wordt naar wijzigingen in de medicatie bedoelen we veranderingen in dosis, frequentie en vorm en of medicatie volledig gestopt is of nieuwe medicatie werd gestart. 
Maak hiervoor een vergelijking tussen enerzijds de medicatie bij opname en anderzijds de medicatie bij ontslag en de medicatie bij therapievoorstel. 
Medicatie die hetzelfde gebleven is, wordt niet vermeld bij gewijzigde medicatie. 
Als gevraagd wordt naar het stollingsrisico, vermeld hier dan bloedklonters/trombosen en embolieën en het gebruik van bloedverdunnende medicatie. Vermeld hier ook of je iets terugvindt over herseninfarct/CVA of hartinfarct/AMIAMI. Ook stollingsstoornissen moeten hier vermeld worden. 
Communicatie gaat over het bespreken van de aandoening en het verdere beleid met de patiënt en zijn familie. 
Als een sectie niet vermeld moet worden, laat deze dan volledig weg uit de samenvatting. 
De informatie in de samenvatting moet uit de oorspronkelijke brief komen. Een uitzondering hierop is als er melding wordt gemaakt van ICD-codes of ICPC-codes. 
Bij de vraag naar ICD-codes moet de ICD-10-code naast de gestelde diagnose vermeld worden, zoals te vinden op https://icd10be.health.belgium.be/.
Bij de vraag naar ICPC-codes kan je deze vinden op https://browser.icpc-3.info/. 
Als een patiënt overgekomen is vanuit een ander ziekenhuis, vermeld dit dan ook onder algemeen.     
    """

    result_nl = df.groupby("email", as_index=False)["choosen_prompt_nl"].agg(" ; ".join)
    result_nl['choosen_prompt_nl'] = pre_text_nl + result_nl['choosen_prompt_nl'] + post_text_nl
    ## remove dubbele ;
    result_nl['choosen_prompt_nl'] = result_nl['choosen_prompt_nl'].apply(lambda x: re.sub(r'\s*;+\s*', ';', x))
    result_nl['choosen_prompt_nl'] = result_nl['choosen_prompt_nl'].apply(lambda x: re.sub(r';+', '; ', x))


    ### DEEL 3: samenvoegen van beide
    result = pd.merge(result_eng, result_nl, on='email', how='outer')


    return result





def add_extra_row(df, naam, max_id ):
    """
    Toevoegen van extra rijen (bijv GENERIEK) aan de originele lijst

    """
    # Create new row with the required values
    new_row = {
        'E-mail': naam,
        'ID': max_id + 1,
        'Begintijd': '',
        'Tijd van voltooien': '',
        'Naam': naam,
        'Geef je volledige voor- en familienaam': naam,
        'Geef aan welk type zorgverlener je bent': '',
        'Geef aan hoeveel jaar ervaring je als praktiserend zorgverlener hebt': ''
    }

    for i in range(8,len(df.columns)):
        col_name = df.columns[i]

        df[col_name] = df[col_name].astype(str)  # Convert all values to string

        if (i-4) in (6,7,14,35,39,40,42,45,83,94,95,100,108):
            # Ensure values are treated as strings (to handle potential mixed types)
            df[col_name] = df[col_name].astype(str)
            # Count the number of non-empty records
            total_records = len(df)
            # Handle NaN values explicitly
            nan_count = (df[col_name] == "nan").sum()
            all_values = []
            for val in df[col_name]:
                if val != "nan":  # Ignore NaN values in splitting
                    all_values.extend(val.split(";"))

            # Count occurrences of each unique value
            value_counts = Counter(all_values)

            # Voorkomen in minsten 50% van de records -- soms net te weinig aangeoast naar 1/3de
            threshold = total_records * (1/3)
            frequent_values = [key for key, count in value_counts.items() if count >= threshold]

            # If NaN is the most frequent, return 'nan', else return the frequent values
            if nan_count > max(value_counts.values(), default=0):
                most_frequent = "nan"
            else:
                #most_frequent = frequent_values
                most_frequent = "; ".join(frequent_values)

        else:
            most_frequent = df[col_name].mode(dropna=False).iloc[0]  # Get the most frequent value, including NaNs
            if df[col_name].isna().sum() > df[col_name].value_counts(dropna=True).max():
                most_frequent = ""  # If NaNs occur most, set result to empty string

        new_row[col_name] = most_frequent

    return new_row




def generieke_rijen_toevoegen(df):
    """
    Toevoegen van het profiel GENERIEK,GENERIEK_ZIEKENHUISARTS_SPECIALIST en GENERIEK_HUISARTS aan de output van de vragenlijst
    """


    ### generieke lijn toevoegen voor alle deelnemers
    max_id = df['ID'].max()
    new_row_generiek = add_extra_row(df=df, naam = 'GENERIEK', max_id = max_id)
    df_new = pd.concat([df, pd.DataFrame([new_row_generiek])])

    ### doe hetzelfde maar specifiek enkel voor de ziekenhuisartsen
    df_zhas = df[df['Geef aan welk type zorgverlener je bent'] == 'ziekenhuisarts/specialist']
    new_row_zha = add_extra_row(df=df_zhas, naam='GENERIEK_ZIEKENHUISARTS_SPECIALIST', max_id=max_id+1)
    df_new = pd.concat([df_new, pd.DataFrame([new_row_zha])])

    ### doe hetzelfde maar specifiek enkel voor de huisartsen
    df_ha = df[df['Geef aan welk type zorgverlener je bent'] == 'huisarts']
    new_row_ha = add_extra_row (df=df_ha, naam = 'GENERIEK_HUISARTS', max_id = max_id+2)
    df_new = pd.concat([df_new, pd.DataFrame([new_row_ha])])

    return df_new


def frait_add_hardcoded_pre_prompt_evaluation(basis_result, extra_pre_prompt):
    """
    Toevoegen van de extra hard codes pre-prompt evaluatie vragen

    """

    ## Expand extra_pre_prompt voor elke unieke user
    emails = basis_result['email'].unique()
    small_expanded = pd.concat([extra_pre_prompt.assign(email=email) for email in emails], ignore_index=True)

    # Voeg the new rows to the basis dataframe
    basis_result = pd.concat([basis_result, small_expanded], ignore_index=True)


    return basis_result



def add_evaluatie_columns(row):
    """
    Evaluatie kolommen toevoegen voor iedere rij van de resultaten
    """

    # Algemene evaluatie voor vraag 111
    if row['number'] == 111:
        row[f'evaluatie_vorm_vraag'] = "Algemeen: geef een algemene score geven voor de volledige samenvatting voor wat betreft de VORM?"
        row[f'evaluatie_vorm_antwoorden'] = ["Zeer slecht", "Slecht", "Goed", "Zeer goed"]
        row[f'evaluatie_inhoud_vraag'] = "Algemeen: geef een algemene score geven voor de volledige samenvatting voor wat betreft de INHOUD?"
        row[f'evaluatie_inhoud_antwoorden'] = ["Zeer slecht", "Slecht", "Goed", "Zeer goed"]

    # Evaluatie als er een bestaande prompt is
    elif row['number'] > 3 and row['choosen_prompt_nl'] and row['choosen_prompt_nl'] != " " and row['choosen_prompt_nl'] != "  ":
        row[f'evaluatie_vorm_antwoorden'] = ["Ja", "Nee"]
        row[f'evaluatie_inhoud_antwoorden'] = [
            "Ja",
            "Nee - Onvolledig",
            "Nee - Teveel / Irrelevant (info wel in brief)",
            "Nee - Foute info / Hallucinatie (info niet in brief)"
        ]
        if row['question_type'] == 'item':
            row[f'evaluatie_vorm_vraag'] = f"Staat het gevraagde item in de samenvatting?"
            row[f'evaluatie_inhoud_vraag'] = f"Is de inhoud van dit item correct? Zo neen, duid aan wat er foutief is."
        elif row['question_type'] == 'sectie':
            row[f'evaluatie_vorm_vraag'] = f"Staat de gevraagde sectie in de samenvatting?"
            row[f'evaluatie_inhoud_vraag'] = f"Heb je alle inhoudelijke fouten kunnen aangeven bij deze sectie? Zo neen, duid aanvullend aan wat nog foutief is."
            # f"Is de volledige inhoud van deze sectie correct. Zo neen, duid aanvullend aan wat er foutief is."
        elif row['question_type'] == 'opmaak':
            row[f'evaluatie_vorm_vraag'] = f"Is de gevraagde opmaak toegepast?"
            row[f'evaluatie_inhoud_vraag'] = np.nan
            row[f'evaluatie_inhoud_antwoorden'] = np.nan
        elif row['question_type'] == 'volgorde':
            row[f'evaluatie_vorm_vraag'] = f"Is de gevraagde volgorde correct toegepast in de samenvatting?"
            row[f'evaluatie_inhoud_vraag'] = np.nan
            row[f'evaluatie_inhoud_antwoorden'] = np.nan
        elif row['question_type'] == 'ontbrekend':
            row[f'evaluatie_vorm_vraag'] = f"Staan de ontbrekende items opgelijst in de samenvating?"
            row[f'evaluatie_inhoud_vraag'] = np.nan
            row[f'evaluatie_inhoud_vraag'] = f"Zijn de ontbrekende items correct? Zo neen, duid aan wat er foutief is."
        else:
            row[f'evaluatie_vorm_vraag'] = np.nan
            row[f'evaluatie_inhoud_vraag'] = np.nan
            print(f"question_type ongekend voor deze vraag: {row['number']}")
    else:
        row[f'evaluatie_vorm_vraag'] = np.nan
        row[f'evaluatie_vorm_antwoorden'] = np.nan
        row[f'evaluatie_vorm_antwoorden'] = np.nan
        row[f'evaluatie_inhoud_antwoorden'] = np.nan

    return row


def create_metadata_evaluatie_antwoorden(output_file = None):
    """
    Metadata bij de evaluatie vragen / antwoorden voor de JSON DOCAT structuur input en output
    """

    EV1 = {"possible_answers": {
        "EV1_Y": {
            "label": "Ja"
        },
        "EV1_N": {
            "label": "Nee"
        }
    },
        "ui_type": "radiobutton"}

    EV2_N_A = {
        "label": "Onvolledig",
        "fixed_origin": "original letter"}
    EV2_N_B = {
        "label": "Teveel / Irrelevant (info wel in brief)",
        "fixed_origin": "generated summary"
    }
    EV2_N_C = {
        "label": "Foute info / Hallucinatie (info niet in brief)",
        "fixed_origin": "generated summary"
    }

    EV2 = {"possible_answers": {
        "EV2_Y": {
            "label": "Ja"
        },
        "EV2_N": {
            "label": "Nee",
            "cascade_answers": {
                "EV2_N_A": EV2_N_A,
                "EV2_N_B": EV2_N_B,
                "EV2_N_C": EV2_N_C
            }
        }
    },
        "ui_type": "radiobutton+cascasde"}

    EV3 = {"possible_answers": {
        "EV3_1": {
            "label": "Zeer slecht"
        },
        "EV3_2": {
            "label": "Slecht"
        },
        "EV3_3": {
            "label": "Goed"
        },
        "EV3_4": {
            "label": "Zeer goed"
        }
    },
        "ui_type": "radiobutton"}

    meta_data_dict = {
        "metadata_questions": {
            "EV1": EV1,
            "EV2": EV2,
            "EV3": EV3,
        }
    }


    meta_data_json = json.dumps(meta_data_dict, indent=1, ensure_ascii=False)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as output_file:
            output_file.write(meta_data_json)

    return meta_data_json


if __name__ == '__main__':
    ## inlezen van de csv file
    filenaam_vragenlijst = 'input/FRAIT_updated_questions_output.csv'
    ds_vragenlijst = pd.read_csv(filenaam_vragenlijst,  sep=";", encoding='latin-1')

    filenaam_vragen_resultaten = 'input/FRAIT_resultaten_vragenlijst.csv'
    ds_res = pd.read_csv(filenaam_vragen_resultaten,  sep=";", encoding='latin-1')

    ## TOEVOEGEN van de GENERIEKE LIJNEN aan de resultaten
    print("Voeg de extra GENERIEKE lijnen toe")
    ds_res_extra = generieke_rijen_toevoegen(df=ds_res)
    ds_res_extra.to_csv("output/FRAIT_resultaten_vragenlijst_MET_GENERIEKE.csv", sep=";", encoding = 'latin1')

    ## BEREKEN de basis prompts
    print("MAAK de basis prompts per vraag")
    basis_result = frait_choosen_prompt(ds_vragenlijst, ds_res_extra)
    basis_result.to_csv("output/FRAIT_basis_prompt_resultaten_zonder_preprompt.csv", sep=";", encoding = 'latin1')

    ### pre-rompt toevoegen
    filenaam_extra_pre_prompt = "input/FRAIT_input_pre_prompt_hardcoded.csv"
    extra_pre_prompt = pd.read_csv(filenaam_extra_pre_prompt, sep=";", encoding='latin-1')
    basis_result_with_extra = frait_add_hardcoded_pre_prompt_evaluation(basis_result,extra_pre_prompt )
    basis_result_with_extra.to_csv("output/FRAIT_basis_prompt_resultaten.csv", sep=";",
                        encoding='latin1')

    ## MAAK DE INDIVIDUELE PROMPTS
    print("MAAK de individuele prompts")
    indiviual_prompt_result = frait_individual_prompt(df=basis_result_with_extra)
    indiviual_prompt_result.to_csv("output/FRAIT_individuele_prompt_resultaten.csv", sep=";", encoding = 'latin1')
    #indiv_prompt = frait_individual_prompt(basis_result)

