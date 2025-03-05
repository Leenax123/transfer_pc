import json
from groq import Groq
import json
import re
import requests


MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = "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"

class Agent:
    def __init__(self):
        self.api_url = "http://localhost:4000"  # URL de l'API FastAPI
        self.tools = {
            "add_documents": self.add_documents,
            "search_documents": self.search_documents,
        }
        self.groq_client = Groq(api_key=GROQ_API_KEY) # Initialisation du client Groq avec la clé API

    # ====================================================================================================================================================================
    # appel du llm
    # ====================================================================================================================================================================

    def retrieve_tool(self, prompt : str):
        """Appel du modèle de langage pour obtenir la réponse sur le tool à utiliser."""
        print("Envoi du prompt au modèle : ", prompt)

        try:
            response = self.groq_client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}]
            )  # Appel à l'API Groq pour obtenir une complétion du modèle

            # Extraction de la réponse générée par le modèle
            if response and response.choices:  # Si la réponse est valide
                tool_response = response.choices[0].message.content.strip()  # Extraire et nettoyer la réponse
                # print(f"Réponse brute du modèle: {tool_response}") #DEBUG
                # print(f"Type de la réponse: {type(tool_response)}")

                # Si la réponse est déjà une chaîne JSON, on la parse en dictionnaire
                try:
                    response_data = json.loads(tool_response)  # Parsing de la chaîne JSON en dictionnaire
                    #transforme chaîne de caractères au format json -> dictionnaire Python
                    print("Réponse analysée : ", response_data)

                    action = response_data.get("action") #récupère l'action dans un dict
                    if action:
                        print(f"Action détectée : {action}")
                        return response_data  # Retourne les données extraites
                    else:
                        print("Action manquante dans la réponse.")
                        return None

                except json.JSONDecodeError as e:
                    print(f"Erreur de parsing JSON : {e}")
                    return None
            else:
                print("Réponse vide du modèle.")
                return None

        except Exception as e:
            print(f"Erreur lors de l'appel à Groq : {e}")
            return None

   

    # ====================================================================================================================================================================
    # FONCTIONS
    # ====================================================================================================================================================================
    def add_documents(self, sentences):
        """Appeler l'API pour ajouter un document"""
        data = {"sentences": sentences}
        headers = {"Content-Type": "application/json"} # Headers pour indiquer que les données sont en JSON
        url = f"{self.api_url}/add-document"  # URL complète pour l'API

        response = requests.post(url, headers=headers, json=data)
        print("Réponse complète de l'API:", response)
        return response

    def search_documents(self, query):
        """Appeler l'API pour rechercher un document"""
        url = f"{self.api_url}/search"  # URL complète pour l'API
        headers = {"Content-Type": "application/json"} # Headers pour indiquer que les données sont en JSON
        data={"query": query}
        response = requests.get(url, headers=headers, params=data)

        return response.json()

    # ====================================================================================================================================================================
    # Agent workflow
    # ====================================================================================================================================================================
    def initiate_workflow(self, user_query: str = "I want to add these sentences to the database : 'The weather is nice today', 'I enjoy reading science fiction books'. What is the closest sentence that you have to 'I love reading fantasy novels'?"):
        """gère le workflow de l'agent, ajoute un document puis recherche

        :param str sender: The person sending the message
        :param str recipient: The recipient of the message"""
            #a: str = ""
        # "The user wants to add new sentences to the database. " \
        # "The sentences are ready to be inserted in the collection."

        # définition du prompt
        prompt = f""" 
        You have two tools: add_documents and search_documents.
        Analyze this user query: "{user_query}".
        - If the user wants to add sentences, extract them in a JSON format: {{"action": "add", "sentences": ["sentence1", "sentence2"]}}.
        - If the user wants to search, return: {{"action": "search", "query": "search phrase"}}.
        - If both, return: {{"action": "both", "sentences": ["sentence1"], "query": "search phrase"}}.
        Only return JSON format, nothing else.
        """
    
    
        llm_response = self.retrieve_tool(prompt)
        
        if not llm_response:
            print("Erreur : le modèle n'a pas fourni de réponse valide.")
            return
        # Extraire l'action et les données
        try:
            response_data = llm_response  # Convertir la réponse JSON en dictionnaire
            action = response_data.get("action")
        except Exception as e:
            print(f"Erreur de parsing de la réponse du LLM : {e}")
            return

        if action == "add_documents":
            print("Ajout des documents...")
            add_response = self.add_documents(response_data["sentences"])
            print("Réponse ajout des documents:", add_response)

        elif action == "search_documents":
            print("Recherche du document...")
            search_response = self.search_documents(response_data["query"])
            print("Réponse recherche:", search_response)

        elif action == "both":
            print("Ajout des documents...")
            add_response = self.add_documents(response_data["sentences"])
            print("Réponse ajout des documents:", add_response)

            print("Recherche du document...")
            search_response = self.search_documents(response_data["query"])
            print("Réponse recherche:", search_response)

        else:
            print(f"Action '{action}' non reconnue.")


if __name__ == "__main__":
    print("Lancement de l'agent...")
    agent = Agent()
    
    # Exemple de requête utilisateur qui mélange ajout et recherche
    #user_query = "I want to add these sentences to the database: 'My name is Oussama', 'I love playing Valorant'. What is the closest sentence that you have to 'I love playing Overwatch too sometimes'?"
    
    agent.initiate_workflow()

