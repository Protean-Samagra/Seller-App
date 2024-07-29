from flask import Flask, request, jsonify, render_template
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)

es = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "elastic"),
    ca_certs="elasticsearch-8.14.1\\config\\certs\\http_ca.crt"
)

model = SentenceTransformer("all-mpnet-base-v2")

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0: 
        return vector
    return vector / norm

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    input_keyword = data['keyword']

    # Spell check
    spell_check_query = {
        "suggest": {
            "product-name-suggest": {
                "prefix": input_keyword,
                "completion": {
                    "field": "ProductName_suggest",
                    "fuzzy": {
                        "fuzziness": "auto"
                    }
                }
            }
        }
    }

    spell_check_res = es.search(index="all_products", body=spell_check_query)
    suggestions = spell_check_res.get('suggest', {}).get('product-name-suggest', [])[0].get('options', [])

    if suggestions:
        corrected_keyword = suggestions[0]['text']
        vector_of_input_keyword = model.encode(corrected_keyword)
    else:
        corrected_keyword = input_keyword
        vector_of_input_keyword = model.encode(input_keyword)

    # k-NN query for DescriptionVector
    knn_query_description = {
        "field": "DescriptionVector",
        "query_vector": vector_of_input_keyword,
        "k": 10,
        "num_candidates": 10000
    }

    # k-NN query for ProductNameVector
    knn_query_productname = {
        "field": "ProductNameVector",
        "query_vector": vector_of_input_keyword,
        "k": 10,
        "num_candidates": 10000
    }

    # Perform k-NN search
    res_description = es.search(index="all_products", knn=knn_query_description, _source=["ProductName", "Description", "DescriptionVector"])
    res_productname = es.search(index="all_products", knn=knn_query_productname, _source=["ProductName", "Description", "ProductNameVector"])

    # Combine results
    combined_hits = res_description["hits"]["hits"] + res_productname["hits"]["hits"]

    # Remove duplicate hits based on the document ID
    combined_hits = {hit["_id"]: hit for hit in combined_hits}.values()

    # Filter and sort the combined results using mean cosine similarity
    filtered_hits = []
    for hit in combined_hits:
        description_vector = hit["_source"].get("DescriptionVector")
        productname_vector = hit["_source"].get("ProductNameVector")
        similarity_score = 0

        if description_vector is not None and productname_vector is not None:
            similarity_score = (cosine_similarity(vector_of_input_keyword, description_vector) + cosine_similarity(vector_of_input_keyword, productname_vector)) / 2
        elif description_vector is not None:
            similarity_score = cosine_similarity(vector_of_input_keyword, description_vector)
        elif productname_vector is not None:
            similarity_score = cosine_similarity(vector_of_input_keyword, productname_vector)

        hit["_source"]["similarity_score"] = similarity_score
        filtered_hits.append(hit)

    # Sort filtered results by combined score (mean cosine similarity + Elasticsearch score)
    filtered_hits = sorted(filtered_hits, key=lambda x: (0.5 * x["_source"]["similarity_score"] + 0.5 * x["_score"]), reverse=True)

    # Apply additional filters and scoring adjustments
    final_hits = []
    for hit in filtered_hits:
        # Dynamically filter results based on keyword
        if 'shoes' in corrected_keyword.lower() and 'shoes' not in hit['_source']['ProductName'].lower():
            continue
        final_hits.append(hit)

    results = [{"ProductName": hit["_source"]["ProductName"], "Description": hit["_source"]["Description"], "Score": hit["_score"]} for hit in final_hits[:20]]

    # Multi-language query
    multi_lang_query = {
        "multi_match": {
            "query": corrected_keyword,  # Use the corrected keyword here
            "fields": ["ProductName^3", "ProductName.english", "ProductName.hindi", "Description", "Description.english", "Description.hindi"]
        }
    }

    multi_lang_res = es.search(index="all_products", body={"query": multi_lang_query})
    for hit in multi_lang_res["hits"]["hits"]:
        results.append({"ProductName": hit["_source"]["ProductName"], "Description": hit["_source"]["Description"], "Score": hit["_score"]})

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
