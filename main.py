from fastapi import FastAPI, Query
import uvicorn
from pydantic import BaseModel
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    DataType,
    CollectionSchema,
    MilvusClient,
    utility,
    db,
)

from sentence_transformers import SentenceTransformer, util
from typing import List
import numpy as np


class sentence_input(BaseModel):
    sentences: List[str]  # modèle d'entrée avec une liste de phrases


model_name = "sentence-transformers/all-MiniLM-L6-v2"  # choix du modèle
model = SentenceTransformer(model_name)  # SentenceTransformer : modèle de transformation de texte vers des vecteurs numériques

# ====================================================================================================================================================================
# Connection au serveur Milvus local
# ====================================================================================================================================================================
connections.connect(
    alias="default",  # Nom de connexion par défaut
    host="127.0.0.1",  # Note: il faut mettre ici le nom du container quand on travaille dans des containers! 
    port="19530",  # Port par défaut de Milvus
)
print(connections.list_connections)

# ====================================================================================================================================================================
# Création de la collection
# ====================================================================================================================================================================

collection_name = "new_collection"
client = MilvusClient(uri="http://localhost:19530")#idem ici on met le nom du container
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # définition de l'id
    FieldSchema(
        name="vector", dtype=DataType.FLOAT_VECTOR, dim=384
    ),  # définition des vecteurs (taille 384 pour all-MiniLM-L6-v2)
    FieldSchema(
        name="text", dtype=DataType.VARCHAR, max_length=512
    ),  # champ pour les textes (taille max 512)s
]
schema = CollectionSchema(
        fields)  

if not client.has_collection(
    collection_name=collection_name
):  # crée une collection uniquement s'il n'en existe pas au préalable
    # crée un schéma de collection avec les champs définis au-dessus
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        enable_dynamic_field=True,  # active les champs dynamiques
    )

# ====================================================================================================================================================================
# Création de l'index
# ====================================================================================================================================================================

index_params = (
    {
        "field_name": "vector",  # index inséré dans le champ des vecteurs
        "metric_type": "COSINE",  # type de similarité
        "index_type": "IVF_FLAT",  # type d'index à utiliser
        "params": {"nlist": 128},
    },
)

client.create_index(  # création de l'index avec les paramètres définis
    collection_name=collection_name,
    index_params=index_params,
    sync=False,  # ne pas attendre que l'index soit crée avant de continuer
)

client.load_collection(collection_name)  # charge la collection dans la mémoire Milvus

# ====================================================================================================================================================================
# Défibution de l'API
# ====================================================================================================================================================================
app = FastAPI()  # crée une instance de FastAPI


@app.post("/add-document")  # définitioon du endpoint de l'API
def add_documents(
    input: sentence_input,) :  #->dict pour renvoyer un dictionnaire en sortie
                             # fonction qui reçoit un input avec le modèle de texte défini dans la classe
    """fonction qui ajoute un documents"""
    if not input:
        return {"error": "La liste des phrases est vide"}  # erreur si entrée vide

    list_doc = (
        input.sentences
    )  # récupère le texte entré par l'utilisateur dans liste_doc

    sentence_vectors = model.encode(
        list_doc
    )  # encode les phrases en vecteurs avec le modèle pré-entrainé

    data = [
        {
            # "id": i,  # id  unique pour chaque phrase
            "vector": sentence_vectors[i],  # vecteur associé à chaque phrase
            "text": list_doc[i],  # la phrase en personne
        }
        for i in range(len(sentence_vectors))
    ]
    try:
        res = client.insert(
            collection_name=collection_name, data=data)  # ajout les data définies précédemment (insert() flushe automatiquement)

        if res is not None:  # si les données insérées avec succès
            return {
                "success": True,
                "message": "Données insérées",
            }

        else:
             return {
                 "success": False,
                 "message": "Aucune donnée insérée else",
            }  # si aucune donnée n'est insérée

    except Exception as e:  # echec en cas d'erreur durant l'insertion
        print(e)
        return {"success": False, "message": "Aucune donnée insérée except"}



@app.get("/search")
def search(query: str = Query(...)):  # focntion qui reçoit une query en entrée

    if (
        collection_name not in utility.list_collections()
    ):  # vérifie si la collection existe et retourne une erreur sinon
        return {"error": "Collection not found"}
    collection = Collection(
        collection_name
    )  # je crée l'objet Collection pour appeler d'autre, on pourrait faire client.load mais plus facile à manipuler par la suite
    collection.load()  # charge la collection

    # encode la requête en vecteurs
    query_vectors = model.encode([query])

    # rechercher les 2 phrases les plus similaires
    results = collection.search(
        data=query_vectors,
        anns_field="vector",  # le champ dans lequel chercher
        param={
            "metric_type": "COSINE",
            "params": {"nlist": 128},
        },  # divise les données en 128 partitions, on peut choisir un nlist plus petit
        # car 128 : des milions de vecteurs
        limit=2,  # nombre de résultats maxx
        output_fields=[
            "text"
        ],  # spécifie que l'on veut la phrase en elle-même en sortie
    )

    # Extraire les résultats
    best_matches = [
        {"text": hit.entity.get("text"), "score": hit.distance} for hit in results[0]
    ]
    # results=liste des recherches pour chaque vecteur de la collection, result[0] = recherche de la première phrase de la requete
    # hit = objet qui contient des infos sur la correspondance (distance entre les vecteurs)
    # hit.entity.get("text")=> accède à l'entité du hit et extrait la valeur du champ "text"
    # .get("text") = extrait le texte stocké dans "text"
    # hit.distance = donne la distance (ici cosine) entre les vecteurs

    return {"query": query, "best_matches": best_matches}


# Démarrer le serveur API
if __name__ == "__main__":      #le serveur ne démarre que lorsqu'on éxécute ce fichier
    uvicorn.run(app, host="127.0.0.1", port=4000) #pour un container docker toujours : 0.0.0.0 
                                                #le serveur écoute en localhost :127.0.0.1
                                                # le serveur utilise le port 4000 
