import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Initialize the SentenceTransformer model
model = SentenceTransformer('paraphrase-distilroberta-base-v2')

def find_top_k_similar_sentences(sentence, sentence_categories, sentences, all_categories, k=5):
    """
    Finds the top k similar sentences based on category and similarity score.
    
    Parameters:
    - sentence (str): The reference sentence to find similarities to.
    - sentence_categories (list): Categories of the reference sentence.
    - sentences (list): List of all sentences.
    - all_categories (list): List of categories for each sentence.
    - k (int): Number of top similar sentences to find.
    
    Returns:
    - list: Top k similar sentences.
    """
    # Filter sentences by category
    filtered_sentences = [sent for sent, cats in zip(sentences, all_categories) 
                          if set(sentence_categories).intersection(set(cats)) and sent != sentence]

    # Encode sentences to embeddings
    sentence_embeddings = model.encode(filtered_sentences, convert_to_tensor=True)
    query_embedding = model.encode(sentence, convert_to_tensor=True)

    # Calculate cosine similarities
    similarities = [util.pytorch_cos_sim(query_embedding, sentence_embedding) for sentence_embedding in sentence_embeddings]
    sorted_indices = np.argsort(similarities)[::-1]
    
    return [filtered_sentences[i] for i in sorted_indices][:k]

def main(json_file_path, output_file_path, k=5):
    """
    Main function to process the JSON input file and generate related articles.

    Parameters:
    - json_file_path (str): Path to the input JSON file.
    - output_file_path (str): Path to the output text file.
    - k (int): Number of top similar sentences to find.
    """
    # Read the JSON and extract the required fields
    df = pd.read_json(json_file_path)
    sentences = df['title'].tolist()
    categories_list = df['categories'].tolist()
    post_ids = df['id'].tolist()  # Assuming a column named 'id' contains the post IDs

    output = []

    for idx, (sentence, sentence_categories) in enumerate(zip(sentences, categories_list)):
        similar_sentences = find_top_k_similar_sentences(sentence, sentence_categories, sentences, categories_list, k)
        
        # Find post IDs of the similar sentences
        related_post_ids = [post_ids[sentences.index(sim_sentence)] for sim_sentence in similar_sentences]
        
        # Format the output string
        output.append(f"{post_ids[idx]} => {','.join(map(str, related_post_ids))}")
        print(sentence)

    # Write to a text file
    with open(output_file_path, "w") as file:
        for line in output:
            file.write(line + '\n')

if __name__ == "__main__":
    json_file_path = 'input.json'
    output_file_path = 'output.txt'
    main(json_file_path, output_file_path)
