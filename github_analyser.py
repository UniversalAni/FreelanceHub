import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode text using BERT
def encode_text(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
          outputs = model(**inputs)
          embeddings = outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token embedding
        return embeddings

    # Calculate cosine similarity
def calculate_similarity(text1, text2):
        embedding1 = encode_text(text1)
        embedding2 = encode_text(text2)
        similarity = cosine_similarity(embedding1.numpy(), embedding2.numpy())
        return similarity[0][0]

    # Evaluate programming languages
def evaluate_languages(task_languages, repo_languages):
    task_languages = [lang.strip().lower() for lang in task_languages]
    repo_languages = [lang.strip().lower() for lang in repo_languages]

    # Debug: Print the normalized lists
    print("Normalized Task Languages:", task_languages)
    print("Normalized Repo Languages:", repo_languages)

    # Count matches
    match_count = sum(lang in repo_languages for lang in task_languages)

    # Debug: Print match count
    print("Match Count:", match_count)

    # Calculate proportion
    return match_count / len(task_languages) if task_languages else 0

    # Combine repository descriptions
def aggregate_repositories(repo_descriptions):
        str=""
        for i in repo_descriptions:
             str+=i+" "
        return str

    # Main function to calculate suitability score
def evaluate_freelancer(task_description, task_languages, repo_descriptions, repo_languages):
        # Step 1: Calculate description similarity
        aggregated_repos = aggregate_repositories(repo_descriptions)
        description_similarity = calculate_similarity(task_description, aggregated_repos)
    
        # Step 2: Calculate language similarity
        language_similarity = evaluate_languages(task_languages, repo_languages)
    
    # Step 3: Combine similarities into a weighted score
        final_score = 0.7 * description_similarity + 0.3 * language_similarity  # Weighted combination
        return {
            "description_similarity": description_similarity,
            "language_similarity": language_similarity,
            "final_score": final_score
        }


def get_languages(user):
    url = "https://github.com/"+user+"?tab=repositories"
    response = requests.get(url,headers={'User-Agent':'Mozilla/5.0'})
    response_code=response.status_code
    if response_code != 200:
        print("Error detected")
        return
    html_content = response.content
    Ll=[]
    dom = BeautifulSoup(html_content,'html.parser')
    lang=dom.select('li div div.f6 span span[itemprop="programmingLanguage"] ')
    
    for j in lang:
        print(j.contents[0].strip())
        Ll.append(j.contents[0].strip())
    
    return Ll
    
def get_repositories(user):
    url = "https://github.com/"+user+"?tab=repositories"
    response = requests.get(url,headers={'User-Agent':'Mozilla/5.0'})
    response_code=response.status_code
    if response_code != 200:
        print("Error detected")
        return
    html_content = response.content
    dom = BeautifulSoup(html_content,'html.parser')
    desc = dom.select('li div div p')
    L=[]
    try:
    
        for i in desc:
            print(i.contents[0].strip())
            L.append(i.contents[0].strip())
    except:
        return L
        



if __name__=="__main__":
    print('Started scraping')
    L_desc = get_repositories("BharZInstein")
    L_lang = get_languages("BharZInstein")
    #r="Python"
    print(L_desc)
    print(L_lang)
    reqdesc="Write a tic tac toe game"
    reqlang =["Python","HTML"]
    scores = evaluate_freelancer(reqdesc, reqlang, L_desc, L_lang)
    print(f"Description Similarity: {scores['description_similarity']:.4f}")
    print(f"Language Similarity: {scores['language_similarity']:.4f}")
    print(f"Final Suitability Score: {scores['final_score']:.4f}")
    