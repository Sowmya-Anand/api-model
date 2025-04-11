from flask import Flask, request, jsonify
import re, string, math, torch, requests, os
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

# Setup
app = Flask(__name__)
stop_words = set(stopwords.words('english'))
punct = set(string.punctuation)
tokenizer_nltk = RegexpTokenizer(r'\w+')
transformer_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
transformer_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Gemini API Setup
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
API_KEY = os.getenv("GEMINI_API_KEY")  # Replace with actual key or use env var

def preprocess_text(text):
    tokens = tokenizer_nltk.tokenize(text)
    filtered = [w for w in tokens if w.lower() not in stop_words and w not in punct]
    return ' '.join(filtered)

def similarity_score(studans, teachans):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([studans, teachans])
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return sim[0][0]

def bert_embedding(text):
    ip = transformer_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        op = transformer_model(**ip)
    return op.last_hidden_state.mean(dim=1)

def contextual_similarity_score(t1, t2):
    emb1 = bert_embedding(t1)
    emb2 = bert_embedding(t2)
    sim = torch.nn.functional.cosine_similarity(emb1, emb2)
    return sim.item()

def eval_keyword(stud, keywords, marksplit):
    if not keywords:
        return 0
    studlow = stud.lower()
    found = sum(1 for kw in keywords if kw.lower() in studlow)
    rat = found / len(keywords)
    return round(rat * marksplit * 0.75, 2)

def splitwnum(text):
    pattern = r'(?:^|\n)(\d+)(?:\s*[\.\)\-:])'
    parts = re.split(pattern, text)
    answers = {}
    for i in range(1, len(parts), 2):
        qnum = int(parts[i])
        answer = parts[i + 1].strip() if i + 1 < len(parts) else ''
        answers[qnum] = answer
    return answers

def call_gemini_api(studans, teachans, split, marval):
    prompt = f"""You are an expert examiner.
Correct Answer: {teachans}
Student Answer: {studans}
Evaluate the student's answer. How correct is it? What is missing or incorrect? Give it a score out of {marval}.
Mark split: {split}. Return just the score as a number."""
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{API_URL}?key={API_KEY}", headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Gemini API Error: {response.status_code} - {response.text}")

def gemscore(response, marval):
    try:
        text = response['candidates'][0]['content']['parts'][0]['text']
        num = re.search(r'\d+(\.\d+)?', text)
        score = float(num.group()) if num else 0.0
        return min(score, marval)
    except:
        return 0.0

def grading(stud, teach):
    studans = splitwnum(stud)
    total = 0
    results = []

    for i, item in enumerate(teach):
        qno = i + 1
        rteach = item['question']
        marval = item['marksplit']
        split = item['split']
        keywords = item.get('keywords', [])
        pteach = preprocess_text(rteach)

        if qno in studans:
            rstud = studans[qno]
            pstud = preprocess_text(rstud)
            sim1 = similarity_score(pstud, pteach)
            sim2 = contextual_similarity_score(rstud, rteach)
            similarity = round((sim1 * 0.3 + sim2 * 0.7), 2)
            keyword_bonus = eval_keyword(rstud, keywords, marval)
        else:
            similarity = 0.0
            keyword_bonus = 0.0

        base_score = round(similarity * marval * 0.25, 2)
        bertaward = round(min(base_score + keyword_bonus, marval), 2)
        gem_resp = call_gemini_api(pstud, pteach, split, marval)
        gem_award = gemscore(gem_resp, marval)
        totaward = round((bertaward + gem_award) / 2, 2)

        results.append({
            'qno': qno,
            'similarity': similarity,
            'keyword_bonus': keyword_bonus,
            'bert_score': bertaward,
            'gemini_score': gem_award,
            'marks_awarded': totaward,
            'max_marks': marval
        })
        total += math.ceil(totaward * 2) / 2

    return results, total

@app.route('/')
def index():
    return "Student Evaluation API is live!"

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    student_text = data.get('student_answer')
    teacher_data = data.get('teacher_answer')

    if not student_text or not teacher_data:
        return jsonify({"error": "Missing 'student_answer' or 'teacher_answer'"}), 400

    try:
        results, total = grading(student_text, teacher_data)
        return jsonify({
            "total_score": total,
            "details": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("✅ API is starting...")
    print("🔐 GEMINI_API_KEY:", "SET" if API_KEY else "❌ NOT SET")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
