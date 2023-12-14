# Stage-Marijan

## Organisation
Mon stage a duré 3 mois au sein de l'Usine Data Analyse Transverse. Le sujet était : "Etude de la génération de texte."
Le premier mois, je suis monté en compétences sur les techniques de base en NLP. J'ai suivi l'évolution chronologique des méthodes _BOW, TF-IDF, W2V, LSTM_. Ensuite, je me suis intéressé aux transformers (avec BERT en 2018). Puis aux LLMs en général, aux techniques de fine-tuning et enfin au framework LangChain pour construire des applications. Enfin, j'ai développé une application de démonstration avec Streamlit et LangChain, comme exemple de use cases.

## Présentations
J'ai présenté 4 sujets lors des partages techniques à l'équipe technique.

1. BERT : Introduction à l'architecture transformers et au mécanisme d'attention avec BERT. 
2. LoRA : Techniques de Fine-tuning de LLM
3. LLM : Vue d'ensemble des LLMs, comparaison et évaluation des modèles
4. LangChain : Framework permettant de créer des applications basés sur des LLMs (use cases)

## Notebooks
Tout au long de mon stage, j'ai pu jouer en parallèle sur google.colaboratory. Voici quelques Notebooks clés :

#### Notebook (S)BERT Classification Sentiment
Notebook de prise en main de BERT et SentenceBERT pour de la classification de sentiments (positif/negatif) de review de films.

#### Attention 
Notebook bonus qui retrace le mécanisme d'attention pas à pas avec BERT.

Sources :
- Article de référence : https://towardsdatascience.com/deep-dive-into-the-code-of-bert-model-9f618472353e 
- Comprendre le mécanisme de self-attention : https://www.youtube.com/watch?v=g2BRIuln4uc 
- D'autres sources sont disponibles dans /Présentations en PDF/BERT_prez.pdf 
####  Inférence 8bit (bitsandbytes) Finetuning LoRA (peft)
Notebook de fine-tuning de BLOOMZ-3B, chargé en 8bit (avec bitsandbytes). Fine-tuning sur une tâche spécifique avec la technique LoRA (avec peft). Préparation des données d'entraînement pour les tâches suivantes : classification de sentiment et extraction d'information.

Sources :
- Article Hugging Face original, avec le notebook original (à la fin): https://huggingface.co/blog/4bit-transformers-bitsandbytes
- Article sur Falcon, avec un notebook de fine-tuning : https://huggingface.co/blog/falcon 
#### QA PDF avec LangChain
Notebook permettant de faire du Question/Answering sur des documents en utilisant le framework LangChain qui permet de créer facilement et rapidement toute la structure. Les questions sont prises en compte une par une : il ne s'agit pas d'une conversation. La chaîne permet de renvoyer une réponse _construite_ qui cite ses sources.

Sources :
- Documentation LangChain du use case QA `RetrievalQA`: https://python.langchain.com/docs/modules/chains/popular/vector_db_qa
- Notebook d'exemple de use cases traités avec LangChain : https://github.com/gkamradt/langchain-tutorials/blob/main/LangChain%20Cookbook%20Part%202%20-%20Use%20Cases.ipynb
- Notebook de prise en main de LangChain : https://github.com/gkamradt/langchain-tutorials/blob/main/LangChain%20Cookbook%20Part%201%20-%20Fundamentals.ipynb 
- Benchmark des meilleurs modèles d'embedding open-source de Hugging Face : https://huggingface.co/spaces/mteb/leaderboard 
#### Chatbot QA PDF avec LangChain
Notebook reprenant la structure du précédent : recherche sémantique. Cependant, ici, on ajoute de la mémoire : il s'agit d'une conversation. Néanmoins, pour chaque question une recherche sémantique s'effectue. _Il faudrait peut-être envoyer dans la chaîne la conversation plutôt que juste la dernière question..._

Sources : 
- Les mêmes que précédemment.
- Documentation LangChain du use case de Chat-QA `ConversationalRetrievalChain` : https://python.langchain.com/docs/modules/chains/popular/chat_vector_db
## Projet application
J'ai développé une application de démonstration avec Streamlit LangChain, basé sur un LLM. Celui-ciu est disponible dans /projet-application-streamlit-langchain-llm/. Le fichier readme.md détaille ce projet.
L'application, basé sur un LLM téléchargé en local (nécessite un GPU) admet 3 pages :
- un chatbot : il s'agit d'une conversation avec le LLM. _Présence de mémoire_
- du Question/Answering de documents : possibilité de poser une question (ce n'est pas une conversation), et le modèle utilise de la recherche sémantique _(stuffing)_, le modèle répond en citant ses sources
- un synthétiseur de document : il génère un résumé pour un document donné (selon plusieurs méthodes : _map reduce, refine_)

## Mes sources d'informations
Dans l'optique de prolonger mes débuts de recherches et d'approfondir tout cet environnement, voici mes principaux canaux d'informations :

#### Sources classiques 
Dans un premier temps, j'utilise des sources d'informations (et de formation) assez classiques :
- Articles de recherche https://arxiv.org/ 
- Articles Medium https://medium.com/ 
- Cours Andrew NG https://www.deeplearning.ai/short-courses/ 
- Cours Google Cloud https://www.cloudskillsboost.google/course_templates/536 

#### Réseaux sociaux : Youtube
Mais j'ai aussi consommé du contenu vidéo : pour approfondir ou bien pour voir en pratique (comme la mise en place de use cases).
Grande communauté qui partage tout son savoir-faire, et est très à jour sur toutes les nouvelles technologies. Leurs projets sont open-source (sur GitHub). 
- https://www.youtube.com/@1littlecoder/videos Il commente l'actualité des LLMs et IA générative. Il accompagne ses videos avec des notebooks sur google collab pour prendre en main facilement (inférence, ou fine-tuning)
- https://www.youtube.com/@gabrielmongaras/videos Explique les papiers de recherche. Vidéos longues mais très bien expliquées. Exemple : Mécanisme d'Attention, LoRA (fine-tuning), QLoRA, LongNet (LLM avec 1 million de token d'entré), Mixture-of-Experts (utiliser pour GPT-4)
- https://www.youtube.com/@venelin_valkov/videos Video tutoriel de use case avec LangChain. Parle d'actualité.
- https://www.youtube.com/@code4AI/videos Parle d'actualité, 
- https://www.youtube.com/@DataIndependent/videos Vidéo tutoriel de use case avec LangChain. GitHub LangChain : https://github.com/gkamradt/langchain-tutorials 
- https://www.youtube.com/@jamesbriggs/videos Vidéo d'actualité sur les derniers modèles, use cases LangChain.
- https://www.youtube.com/@AemonAlgiz/videos 
- https://twitter.com/_akhaliq AK tweete les derniers papiers de recherche en IA : intéressant pour rester informer.




