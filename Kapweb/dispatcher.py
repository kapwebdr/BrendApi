from os import path
import json
import re
import spacy
import random
import json

model_path = "brenda-spacy-model"

def load_phrases(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data
    
current_file = path.realpath(__file__)

def DispatchSpacy(prompt):
    nlp = spacy.load(path.join(path.dirname(current_file), "..", model_path))
    doc = nlp(prompt)
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    return [(ent.text, ent.label_) for ent in doc.ents]

def Dispatch(prompt, lang="fr"):
    file_path = path.join(path.dirname(current_file), "..", 'Lang', lang + ".json")
    data = load_phrases(file_path)
    prompt = prompt.strip()
    for agent in data:
        for phrase in data[agent]:
            try:
                pattern = phrase.lower()
                pattern = re.sub(r'\[([^]]+)\]', r'(\1)?', pattern, flags=re.I)
                pattern = re.sub(r'\(([^)]+)\)', r'(?:\1)', pattern, flags=re.I)
                pattern = re.sub(r'\{([^}]+)\*\}', r'(?P<\1>.+)', pattern, flags=re.I)
                pattern = re.sub(r'\{([^}]+)\}', r'(?P<\1>[^\\s]+)', pattern, flags=re.I)

                match = re.match(pattern, prompt, flags=re.I)
                
                if match:
                    values = match.groupdict()
                    values['agent'] = agent
                    return values
            except re.error as e:
                print(f"Regex error: {e} in pattern: {pattern}")
    return None


