import pandas as pd
#from googletrans import Translator
from deep_translator import GoogleTranslator, ChatGptTranslator
import numpy as np

class Question:
    def __init__(self, label, number, text, answer_options, next_questions, prompt_nl, prompt_eng, section, question_type):
        """
        Class to create q "question"
        # Example usage:
        question_vb = Question(
            number=1,
            text="Do you want to see section history?",
            answer_options=["Yes", "No"],
            next_questions=[2, 5]
        )

        # Accessing data
        print(question1.text)  # "Do you want to see section history?"
        print(question1.answer_options)  # ["Yes", "No"]

        """
        self.label = label
        self.number = number
        self.text = text
        self.answer_options = answer_options
        self.next_questions = next_questions
        self.prompt_nl = prompt_nl
        self.prompt_eng = prompt_eng
        self.section = section
        self.question_type = question_type

    def to_string(self):
        """Returns a string representation of the question."""
        answer_options_str = ", ".join(self.answer_options)
        next_questions_str = ", ".join(map(str, self.next_questions))
        prompt_nl_str = ", ".join(map(str, self.prompt_nl))
        prompt_eng_str = ", ".join(map(str, self.prompt_eng))
        return (
            f"Question Label: {self.label}\n"
            f"Question {self.number}: {self.text}\n"
            f"Answer Options: {answer_options_str}\n"
            f"Next Questions: {next_questions_str}\n"
            f"Prompt NL: {prompt_nl_str}\n"
            f"Prompt ENG: {prompt_eng_str}\n"
            f"Sectie: {self.section}\n" 
            f"Type Vraag: {self.question_type}"
        )

    def to_dict(self):
        result = {'number': self.number,
                 'question_label': self.text,
                  'section': self.section,
                 'anwers_options': self.answer_options,
                 'jump_to_question': self.next_questions,
                 'prompt_nl': self.prompt_nl,
                 'prompt_eng': self.prompt_eng,
                  'question_type': self.question_type}

        return result


def tag_vraag_naam(tag):
    """
    Based on a given tag, return question_label for Question
    """
    tag_values = {
        "##WAAR##": "Waar wil je dit zien?",
        "##OPMAAK##": "Welke opmaak moet hiervoor gebruikt worden?",
        "##LIJST##": "Welke oplijsting wil je zien?",
        "##LIJSTDETAIL##": "Hoe wil je de oplijsting zien?",
        "##TIJD##": "Welke volgorde wil je zien?",
        "##DETAILS MODUS##": "Hoe wil je de details zien?",
        "##PROTOCOL##": "Wil je de details van het protocol zien? (Indien Ja, vragen we nog hoe)",
        "##SoortReactie##": "Wil je het soort reactie zien?",
        "##SECTIES##": "Welke secties wil je zien in het besluit?",
        "##ALGEMEEN##_Onderdeel": "Welke onderdelen wil je bij Algemeen zien?",
        "##ALGEMEEN##_Opmaak": "Voor welke onderdelen wil je een specifieke opmaak zien (gelieve enkel een selectie te maken van de onderdelen die je hierboven ook aangeduid hebt)?",
        "##Voorgeschiedenis##": "Welke specifieke onderdelen wil je zien?",
        "##Allergie##": "Welke allergieën wil je zien?",
        "##WelkeMedicatie##_Onderdeel": "Welke medicatie wil je zien?",
        "##WelkeMedicatie##_Opmaak": "Voor welke medicatie wil je een specifieke opmaak zien (gelieve enkel een selectie te maken van de medicatie die je hierboven ook aangeduid hebt)?",
        "##GegevensMedicatie##": "Welke gegevens wil je bij de medicatie zien?",
        "##Communicatie##": "Welke onderdelen wil je zien?",
        #"##NieuweDiagnosen##": "Welke specifieke nieuwe diagnoses wens je met aparte opmaak te zien?",
        "##SpecifiekeLIJST##": "Hoe wil je dit voorgesteld zien?",
        "##SpecifiekeMedicatie##": "Welke specifieke medicatie wens je met aparte opmaak te zien?",
        "##SpecifiekeOperaties##": "Welke specifieke operaties wens je met aparte opmaak te zien?",
        "##FOLLOWUP##_Onderdeel": "Welke gegevens wil je bij Follow-Up zien?",
        "##FOLLOWUP##_Opmaak": "Voor welke onderdelen wil je een specifieke opmaak zien (gelieve enkel een selectie te maken van de onderdelen die je hierboven ook aangeduid hebt)?"
    }
    waarde = tag_values[tag]


    return waarde
def tag_vragen(q_nr, kolomnaam, label, sectie_vraag_df, details_df, section, bv,  index=0):
    """
    Create Question for the tags
    """


    full_tag = sectie_vraag_df[kolomnaam][index]
    sep_tag = full_tag.split("_")
    small_tag = sep_tag[0]

    antwoordopties = list(details_df[details_df['Tag'] == small_tag].Optie)
    if full_tag == small_tag:
        prompt_nl_opties = list(details_df[details_df['Tag'] == small_tag].Prompt_Nl)
        prompt_eng_opties = list(details_df[details_df['Tag'] == sectie_vraag_df[kolomnaam][index]].Prompt_Eng)
    else:
        if sep_tag[1] == "Onderdeel":
            basis = "Toon de "
        elif sep_tag[1] == "Opmaak":
            basis = "Highlight de "
        else:
            basis = "Toon de "

        prompt_nl_opties = [basis + optie for optie in antwoordopties]
        prompt_eng_opties = [translate_prompt_nl_to_eng(prompt) for prompt in prompt_nl_opties]

    if full_tag in ("##WAAR##",  "##OPMAAK##",  "##LIJST##",  "##DETAILS MODUS##"):
        question_type = "opmaak"
        prompt_nl_opties = [bv +": " + p for p in prompt_nl_opties]
        prompt_eng_opties = [bv +": " + p for p in prompt_eng_opties]
    elif full_tag in  ("##TIJD##", "##SpecifiekeLIJST##"):
        question_type = "volgorde"
    else:
        question_type = "item"


    question = Question(
        label=label,
        number=q_nr,
        text=tag_vraag_naam(full_tag),
        answer_options=antwoordopties,
        next_questions=[q_nr + 1] * len(antwoordopties),
        prompt_nl = prompt_nl_opties,
        prompt_eng = prompt_eng_opties,
        section=section,
        question_type = question_type

    )
    return question



def tag_levels(sectie_vraag_df, q_nr, questions, label, section, bv, niveau_columns=["Niveau4", "Niveau5", "Niveau6", "Niveau7"], index=0):
    """
    Check if columns in sectie_vraag_df are NaN, and update the questions dictionary.

    Parameters:
        sectie_vraag_df (DataFrame): The DataFrame containing the levels (Niveau columns).
        q_nr (int): The question number.
        questions (dict): Dictionary to store questions.
        label (str): The label for the question.
        niveau_columns (list): List of column names to check (e.g., ["Niveau4", "Niveau5", "Niveau6"]).
    """
    for col in niveau_columns:
        if not pd.isna(sectie_vraag_df[col][index]):  # Check if the value is NaN
            q_nr = q_nr + 1
            label = f"{label}_{sectie_vraag_df[col][index]}"
            question = tag_vragen(q_nr=q_nr, kolomnaam=col, label = label, sectie_vraag_df = sectie_vraag_df, details_df=details_df, index=index, section=section, bv=bv)
            questions.append(question)
    return q_nr



def print_questions(questions):
    """Print questions"""
    for question in questions:
        print(question.to_string())
        print("-" * 40)  # Separator for readability


def save_questions_to_file(questions, filename, to_csv=False):
    """ Save questions
    Standard txt file like output in print_questions.
    optional .csv with tabular output (to_csv=True)
     """
    if(to_csv):
        ## questions to pandas
        result = []
        for q in questions:
            res=q.to_dict()
            result.append(res)
        df = pd.DataFrame(result)
        df.to_csv(filename, sep=";",index=False , encoding='latin1')

    else:
        with open(filename, "w") as file:
            for question in questions:
                file.write(question.to_string() + "\n")
                file.write("-" * 40 + "\n")  # Separator for readability




def create_label_to_number_map(questions):
    """Map labels to their corresponding question numbers"""
    label_to_number = {}
    for question in questions:
        label_to_number[question.label] = question.number
    return label_to_number


def update_next_questions(questions):
    """replace the "JUMP_TO_XXX" labels with respective number"""
    new_questions = questions
    label_to_number = create_label_to_number_map(questions)
    for question in new_questions:
        updated_next_questions = []
        for next_question in question.next_questions:
            if isinstance(next_question, str) and next_question.startswith("JUMP_TO_"):
                ## label na JUMP_TO_
                jump_label = next_question.replace("JUMP_TO_", "")
                ## find corresponding label
                if jump_label in label_to_number:
                    updated_next_questions.append(label_to_number[jump_label])
                ## If FINAL_Q, the END
                elif jump_label == "END":
                    updated_next_questions.append(jump_label)
                else:
                    raise ValueError(f"Label {jump_label} not found in questions list.")
            else:
                updated_next_questions.append(next_question)
        ## Update the next_questions list for the current question
        question.next_questions = updated_next_questions
    return new_questions



def translate_prompt_nl_to_eng(text):
    """ Autoamtic translation of the prompt from dutch (nl) to english (en) """
    text_eng = GoogleTranslator(source='nl', target='en').translate(text=text)

    return text_eng


def create_questions(inputvragenlijst_df, details_df):
    """
    Create list of questionsn
    INPUT csv file : from  Niveau4 expecting only ##TAG##  questions
    """

    ### Adding some extra base questions to the questionnaire
    ## q_nr = question counter
    q_nr = 1
    naam_question = Question(
        label="Naam",
        number=q_nr,
        text="Geef volledige voornaam +  naam in",
        answer_options="",
        next_questions=[q_nr + 1],
        prompt_nl="",
        prompt_eng="",
        section= "BASIS",
        question_type = "algemeen"
    )

    questions = [naam_question]

    q_nr = q_nr + 1
    artstype_question = Question(
        label="Type zorgerverlener",
        number=q_nr,
        text="Geef aan welk type zorgverlener je bent",
        answer_options=["ziekenhuisarts/specialist", "huisarts", "andere zorgverlener"],
        next_questions=[q_nr + 1,q_nr + 1, q_nr + 1],
        prompt_nl="",
        prompt_eng="",
        section= "BASIS",
        question_type="algemeen"
    )

    questions.append(artstype_question)
    q_nr = q_nr + 1
    ervaring_question = Question(
        label="ervaring zorgverlener",
        number=q_nr,
        text="Geef aan hoeveel jaar ervaring je als praktiserend zorgverlener hebt",
        answer_options=["0-2 jaar, 2-5 jaar, 5-10 jaar, >10 jaar"],
        next_questions=[q_nr + 1,q_nr + 1, q_nr + 1, q_nr + 1],
        prompt_nl="",
        prompt_eng="",
        section= "BASIS",
        question_type="algemeen"
    )

    questions.append(ervaring_question)


    #### BASIS VRAAG:
    basis_vraag = "Kun je de volgorde aangeven voor de secties die je in je ontslagbrief wil? Indien je bepaalde secties niet wenst te zien, kan je dit later in de vragenlijst nog aangeven."
    secties = [option for option in  inputvragenlijst_df.Sectie.unique() if pd.notna(option)]
    #secties = [option for option in list(details_df[details_df['Tag'] == "##SECTIES##"].Optie) if pd.notna(option)]


    q_nr = q_nr + 1

    basisprompt_nl = ['Algemeen', 'Voorgeschiedenis', 'Medicatie', 'Onderzoeken', 'Opnameverloop', 'Follow-up', 'Besluit']
    basisprompt_eng = ['General', 'Medical History', 'Medication', 'Examinations', 'Admission history', 'Follow-up', 'Decision']
    basis_question = Question(
        label = "Basisvraag",
        number=q_nr,
        text=basis_vraag,
        answer_options=secties,
        next_questions= [q_nr+1]*len(secties), ## secties aantal keer herhalen
        prompt_nl = basisprompt_nl,
        prompt_eng =basisprompt_eng,
        section = "Algemeen",
        question_type="volgorde"
    )



    questions.append(basis_question)

    ### VRAGEN SECTIE UIT EXCEL
    sectienr = 0
    for section in secties:
        sectienr = sectienr+1
        q_nr = q_nr + 1
        #print(f"section: {section}")
        vraag = f"Wil je sectie '{section}' in je samenvatting zien?"
        antwoord = ["Ja", "Nee"]
        label = f"Sectie_{sectienr}"


        jump_tolabel = f"Sectie_{sectienr+1}" if sectienr < len(secties) else "FINAL_Q"

        q_prompt_nl_ja = f"Toon de sectie {section} in de samenvatting"
        q_prompt_nl_nee = f"Toon de sectie {section} niet in de samenvatting"
        q_prompt_eng_ja = translate_prompt_nl_to_eng(q_prompt_nl_ja)
        q_prompt_eng_nee = translate_prompt_nl_to_eng(q_prompt_nl_nee)

        ## als basisvraag ook al met ## begint

        question = Question(
            label = label,
            number=q_nr,
            text=f"Wil je sectie {section} in je samenvatting zien?",
            answer_options=["Ja", "Nee"],
            next_questions=[q_nr+1,f"JUMP_TO_{jump_tolabel}"]  , ## hoe er hier voor zorgen dat 'nextsection_number' aangepast wordt naar vraag van volgende sectie
            prompt_nl=[q_prompt_nl_ja, q_prompt_nl_nee],
            prompt_eng=[q_prompt_eng_ja, q_prompt_eng_nee],
            section=section,
            question_type="sectie"
        )
        questions.append(question)

        ### voor ieder volgend niveau uitvoeren
        sectie_df = inputvragenlijst_df[inputvragenlijst_df['Sectie'] == section]
        basisvragen = [option for option in  sectie_df.Basisvraag.unique() if pd.notna(option)]
        labelnr = 0
        basisnr = 0
        for bv in basisvragen:
            basisnr = basisnr + 1
            labelnr = labelnr+1

            label = f"sectie_{sectienr}_basisvraag_{labelnr}"
            q_nr = q_nr + 1

            if (bv.startswith("##", 0, 2)):
                sectie_vraag_df = sectie_df[sectie_df['Basisvraag'] == bv].reset_index(drop=True)
                question = tag_vragen(q_nr=q_nr, kolomnaam="Basisvraag", label=label, sectie_vraag_df=sectie_vraag_df, details_df=details_df, section=section, bv=section)
                questions.append(question)

                if (sectie_vraag_df.Niveau3[0].startswith("##", 0, 2)):
                    q_nr = q_nr + 1
                    label = f"{label}_{sectie_vraag_df['Niveau3'][0]}"
                    question = tag_vragen(q_nr=q_nr, kolomnaam="Niveau3", label=label, sectie_vraag_df=sectie_vraag_df, details_df=details_df, section=section, bv=section)
                    questions.append(question)
                    q_nr = tag_levels(sectie_vraag_df, q_nr, questions, label, section=section, bv=section,
                                      niveau_columns=["Niveau4", "Niveau5", "Niveau6", "Niveau7"])
                else:  ## komt normaal niet voor
                    q_nr = q_nr + 1
                    q_nr = vanaf_niv3(questions=questions, q_nr=q_nr, sectie_vraag_df=sectie_vraag_df, label=label,
                                      basisnr=basisnr, sectienr=sectienr, bv=section,
                                      basisvragen=basisvragen,
                                      section=section, index=0)
            else:

                label = f"sectie_{sectienr}_basisvraag_{labelnr}"
                if labelnr < len(basisvragen):
                    jump_tolabel = f"sectie_{sectienr}_basisvraag_{labelnr + 1}"
                elif sectienr < len(secties):
                    jump_tolabel = f"Sectie_{sectienr + 1}"
                else:
                    jump_tolabel = "FINAL_Q"

                if bv in ('ICD codes', 'ICPC codes'):
                    q_prompt_nl = f"Haal de {bv} op voor de diagnosen in sectie {section}"
                else:
                    q_prompt_nl = f"Toon in sectie {section} de {bv}"
                q_prompt_eng = translate_prompt_nl_to_eng(q_prompt_nl)

                question = Question(
                    label=label,
                    number=q_nr,
                    text=f"Wil je {bv} in de sectie {section} zien?",
                    answer_options=["Ja", "Nee"],
                    next_questions=[q_nr + 1, f"JUMP_TO_{jump_tolabel}"],

                    prompt_nl=[q_prompt_nl, ""],
                    prompt_eng=[q_prompt_eng, ""],
                    section=section,
                    question_type = "item"

                )
                questions.append(question)
                sectie_vraag_df = sectie_df[sectie_df['Basisvraag'] == bv].reset_index(drop=True)
                ## check multiple options
                if(len(sectie_vraag_df)> 1):
                    label = f"sectie_{sectienr}_basisvraag_{basisnr}"
                    for i in range(len(sectie_vraag_df)):
                        q_nr = q_nr + 1
                        label_niv = f"{label}_{sectie_vraag_df['Niveau3'][i]}"
                        if (sectie_vraag_df.Niveau3[i].startswith("##",0,2)):
                            question = tag_vragen(q_nr=q_nr, kolomnaam="Niveau3", label=label, sectie_vraag_df = sectie_vraag_df, details_df=details_df, section=section, bv=bv)
                            questions.append(question)
                            q_nr = tag_levels(sectie_vraag_df, q_nr, questions, label_niv, index=i, section=section, bv=bv,
                                       niveau_columns=["Niveau4", "Niveau5", "Niveau6", "Niveau7"])
                        else:
                            q_nr = vanaf_niv3(questions=questions, q_nr=q_nr, sectie_vraag_df=sectie_vraag_df, label=label, basisnr=basisnr,
                                              sectienr=sectienr, bv=bv, basisvragen=basisvragen, section=section,
                                              index=i)

                else:
                    ## check if niveau3 starts with ##
                    if isinstance(sectie_vraag_df.Niveau3[0], str) :
                        if (sectie_vraag_df.Niveau3[0].startswith("##", 0, 2)):
                            q_nr = q_nr + 1
                            label = f"{label}_{sectie_vraag_df['Niveau3'][0]}"
                            question = tag_vragen(q_nr=q_nr, kolomnaam="Niveau3", label=label, bv=bv,
                                                  sectie_vraag_df=sectie_vraag_df, details_df=details_df, section=section)
                            questions.append(question)
                            q_nr = tag_levels(sectie_vraag_df, q_nr, questions, label, section=section, bv=bv,
                                              niveau_columns=["Niveau4", "Niveau5", "Niveau6", "Niveau7"])
                        else:
                            q_nr = q_nr + 1
                            q_nr = vanaf_niv3(questions=questions, q_nr=q_nr, sectie_vraag_df=sectie_vraag_df,
                                              label=label,
                                              basisnr=basisnr, sectienr=sectienr, bv=bv, basisvragen=basisvragen,
                                              section=section, index=0)

    final_q = """Over de volledige vragenlijst:
Wil je bij antwoord JA bij de Ja/Nee vragen, dat er aangegeven wordt ONTBREKEND, indien geen gegevens te vinden zijn
(bijv. Allergie: ONTBREKEND)"""
    final_prompt_nl = "Indien een gevraagd onderdeel niet in de brief staat, bundel dan de ontbrekende per sectie, bijv. Allergie: ONTBREKEND"
    final_prompt_eng = "If a certain item is not available in the letter, bundle this by section, e.g. Allergie: MISSING"
    q_nr = q_nr + 1
    final_question = Question(
        label="FINAL_Q",
        number=q_nr,
        text=final_q,
        answer_options=["Ja: Toon 'ONTBREKEND'", "Nee: Laat het item volledig weg uit jouw op maat gemaakte samenvatting"],
        next_questions=["END","END"],
        prompt_nl= [final_prompt_nl, final_prompt_nl] ,
        prompt_eng= [final_prompt_eng, final_prompt_eng],
        section="FINAL",
        question_type = "ontbrekend"
    )

    questions.append(final_question)



    return questions


def vanaf_niv3(questions, q_nr, sectie_vraag_df, label, basisnr,sectienr, bv, basisvragen, section, index=0):
    """
    From Niveau3: add questions and subsequent questions
    """
    label_niv = f"{label}_{sectie_vraag_df['Niveau3'][index]}"

    vraag = f"Wil je {sectie_vraag_df.Niveau3[index]} bij de {bv} zien?"

    antwoordopties = ["Ja", "Nee"]
    if index < len(sectie_vraag_df) - 1:
        jump_tolabel = f"{label}_{sectie_vraag_df['Niveau3'][index +1]}"
    elif basisnr < len(basisvragen):
        jump_tolabel = f"sectie_{sectienr}_basisvraag_{basisnr + 1}"
    else:
        jump_tolabel = f"Sectie_{sectienr + 1}"

    if bv in ('ICD codes', 'ICPC codes'):
        q_prompt_nl = f"Haal de {bv} op voor de diagnosen in sectie {section}"
    else:
        q_prompt_nl = f"Toon in sectie {section} de {bv}"
    q_prompt_eng = translate_prompt_nl_to_eng(q_prompt_nl)

    question = Question(
        label=label_niv,
        number=q_nr,
        text=vraag,
        answer_options=antwoordopties,
        next_questions=[q_nr + 1, f"JUMP_TO_{jump_tolabel}"],
        prompt_nl=[q_prompt_nl, ""],
        prompt_eng=[q_prompt_eng, ""],
        section=section,
        question_type = "item"
    )
    label_niv = f"{label}_{sectie_vraag_df['Niveau3'][index]}"
    questions.append(question)
    ## als niveau4 bestaat: doe dan de detailvraag
    q_nr = tag_levels(sectie_vraag_df, q_nr, questions, label_niv, index=index, section=section,
                      niveau_columns=["Niveau4", "Niveau5", "Niveau6", "Niveau7"], bv=bv)


    return q_nr



if __name__ == '__main__':
    ### Read 2 csv files
    inputvragenlijst_df = pd.read_csv('input/vragenlijst_input_longformat_v2.csv', sep=";",
                                      encoding='latin-1')
    details_df = pd.read_csv('input/vragenlijst_input_longformat_details_v2.csv', sep=";", encoding='latin-1')

    ### CREATE lijst van questions gebaseerd op de excel file met specifiek formaat
    questions = create_questions(inputvragenlijst_df=inputvragenlijst_df, details_df=details_df)


    ####### PRINT RESULTAAT +  SAVE TO FILE
    print_questions(questions)
    #save_questions_to_file(questions, "FRAIT_questions_output.txt")
    save_questions_to_file(questions, "input/FRAIT_questions_output.csv", to_csv=True)


    ############ UPDATE JUMP_TO_ labels +  print & save

    updated_questions =  update_next_questions(questions)
    print_questions(updated_questions)
    #save_questions_to_file(questions, "FRAIT_updated_questions_output.txt")
    save_questions_to_file(questions, "input/FRAIT_updated_questions_output.csv", to_csv=True)