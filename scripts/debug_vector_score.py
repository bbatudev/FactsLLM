from sentence_transformers import SentenceTransformer, util

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)

query = "Sean Penn refused to be in any crime dramas."
doc_text = "Following his film debut in the drama Taps (1981) and a diverse range of film roles in the 1980s, including Fast Times at Ridgemont High (1982), Penn garnered critical attention for his roles in the crime dramas At Close Range (1986), State of Grace (1990), and Carlito 's Way (1993)."

query_emb = model.encode(query, convert_to_tensor=True)
doc_emb = model.encode(doc_text, convert_to_tensor=True)

score = util.cos_sim(query_emb, doc_emb)
print(f"Cosine Similarity Score: {score.item()}")
print(f"Scaled Score (x100): {int(score.item() * 100)}")
