# Classification avec BERT 

Ce projet implémente un modèle de classification de textes juridiques ("legal_text_classification.csv") utilisant BERT. Il comprend des scripts pour l'entraînement du modèle, une interface de démonstration avec Gradio, et une API pour effectuer des prédictions via FastAPI.

## Contenu du dépôt

- `bert_classification.py` : Script pour l'entraînement du modèle avec PyTorch.
- `demo.py` : Script pour l'interface de démonstration utilisant Gradio.
- `api.py` : Script pour créer une API avec FastAPI.
- `requirements.txt` : Liste des dépendances requises pour exécuter les scripts.

## 1) Entraîner le modèle avec PyTorch

Le fichier `bert_classification.py` contient l'implémentation complète de l'entraînement d'un modèle BERT pour la classification de textes juridiques.

### Étapes principales

- **Préparation des données** : Chargement et transformation des données depuis un fichier CSV en un dataset PyTorch. Chaque entrée comprend le texte d'un cas juridique (`case_text`) et son résultat (`case_outcome`). La variable d'intérêt "case_outcome" a 10 modalités et donc nous avons 10 classes pour notre classification. Les textes sont tokenisés et les labels encodés.

- **Définition du modèle** : Utilisation de `BertModel` de la bibliothèque `transformers` (modèle BERT pré-entraîné (`bert-base-uncased`) comme base) et ajout d'une couche de classification linéaire audessus du modèle BERT pour prédire les résultats des cas.

```bash
python
class CustomBert(nn.Module):
    def _init_(self, model_name_or_path="bert-base-uncased", n_classes=10):
        super(CustomBert, self)._init_()
        self.bert_pretained = BertModel.from_pretrained(model_name_or_path)
        self.classifier = nn.Linear(self.bert_pretained.config.hidden_size, n_classes)
```
**Hyperparamètres** :  
Les hyperparamètres suivants ont été utilisés pour l'entraînement du modèle :
- Taux d'apprentissage (`LR`) : `2e-5`
- Taille des batches (`BATCH_SIZE`) : `12`
- Nombre d'époques (`N_EPOCHS`) : `2`

- **Entraînement du modèle** : Boucle d'entraînement qui calcule la perte, effectue la rétropropagation et met à jour les poids du modèle. Le fichier legal-text-classification.csv a été divisé en train et test datas en utilisant train_test_split de sklearn.model_selection.

- **Évaluation du modèle** : Calcul de la précision et de la perte sur un dataset de test. La fonction de perte utilisée est la `CrossEntropyLoss`.

- **Sauvegarde du modèle** : Sauvegarde du modèle entraîné pour une utilisation future.

### Exécution du script d'entraînement

```bash
python bert_classification.py
```

## 2) Interface de démonstration avec Gradio

La démo interactive est configurée à l'aide de Gradio. Le fichier `demo.py` implémente une interface utilisateur simple pour interagir avec le modèle BERT via Gradio.

### Étapes principales

- **Chargement du modèle** : Chargement du modèle entraîné à partir du fichier sauvegardé.

- **Fonction de classification** : Fonction `classifier_fn` qui prend un texte en entrée, le tokenise, passe les tokens à travers le modèle, et renvoie la classe prédite.

- **Interface Gradio** : Interface Gradio web est lancée permettant à l'utilisateur de saisir un texte et de recevoir une prédiction de la classification en temps réel.

### Exécution de l'interface Gradio

```bash
python demo.py
```

## 3) API pour les tests avec FastAPI

L'API REST est configurée à l'aide de FastAPI. Le fichier `api.py` met en place une API avec FastAPI pour permettre des prédictions via des requêtes HTTP.

### Étapes principales

- **Définition du modèle** : Utilisation du même modèle que pour l'interface Gradio.

- **Création de l'API** : Utilisation de FastAPI pour créer une API avec une route de prédiction prenant un texte en entrée et renvoyant la classe prédite. Le fichier `api.py` met en place un serveur API avec un endpoint `/predict` qui accepte les requêtes POST et retourne les prédictions du modèle.

**Endpoints** :  
- **GET `/`** : Retourne un message de bienvenue.
- **POST `/predict`** : Accepte un texte en entrée et retourne la prédiction du modèle.

- **Fonction de classification** : Réutilisation de la fonction `classifier_fn` pour transformer le texte et effectuer la prédiction.

### Exécution de l'API FastAPI

```bash
uvicorn api:app --host 0.0.0.0 --port 8080
```

## Installation des dépendances

Assurez-vous d'avoir installé toutes les dépendances requises en exécutant :

```bash
pip install -r requirements.txt
```


**Références** :
- [Transformers Library](https://huggingface.co/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Gradio Documentation](https://gradio.app/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
