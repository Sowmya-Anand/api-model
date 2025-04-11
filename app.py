from flask import Flask, request, jsonify
import string, math, torch, re, requests, os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel


# ✅ No downloads at runtime
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# ✅ Initialize
app = Flask(__name__)
stop_words = set(stopwords.words('english'))
punct = set(string.punctuation)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
API_KEY = os.getenv("GEMINI_API_KEY")

def preprocess_text(text):
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in stop_words and w not in punct]
    return ' '.join(filtered)

def similarity_score(studans, teachans):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([studans, teachans])
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return sim[0][0]

def bert_embedding(text):
    ip = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        op = model(**ip)
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
    prompt = f"""
    You are an expert examiner.
    Correct Answer: {teachans}
    Student Answer: {studans}
    Evaluate the student's answer. Give it a score out of {marval}. Mark split: {split}.
    Return just the score as a number, no explanation.
    """
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{API_URL}?key={API_KEY}", headers=headers, json=payload)
    return response.json()

def gemscore(response_text, marval):
    try:
        score = float(response_text)
        return min(score, marval)
    except:
        return 0.0

def grading(student_text, teacher_data):
    studans = splitwnum(student_text)
    total = 0
    results = []

    for i, item in enumerate(teacher_data):
        qno = i + 1
        teach_raw = item['question']
        teach_pre = preprocess_text(teach_raw)
        marval = item['marksplit']
        split = item['split']
        keywords = item.get('keywords', [])

        if qno in studans:
            stud_raw = studans[qno]
            stud_pre = preprocess_text(stud_raw)
            sim1 = similarity_score(stud_pre, teach_pre)
            sim2 = contextual_similarity_score(stud_raw, teach_raw)
            similarity = round((sim1 * 0.3 + sim2 * 0.7), 2)
            keyword_bonus = eval_keyword(stud_raw, keywords, marval)
        else:
            similarity = 0.0
            keyword_bonus = 0.0

        base_score = round(similarity * marval * 0.25, 2)
        bertaward = round(min(base_score + keyword_bonus, marval), 2)

        gem_resp = call_gemini_api(stud_raw, teach_raw, split, marval)
        gem_text = gem_resp['candidates'][0]['content']['parts'][0]['text']
        gem_award = gemscore(gem_text, marval)

        final = round((bertaward + gem_award) / 2, 2)
        results.append({
            'qno': qno,
            'similarity': similarity,
            'keyword_bonus': keyword_bonus,
            'bert_score': bertaward,
            'gemini_score': gem_award,
            'marks_awarded': final,
            'max_marks': marval
        })
        total += math.ceil(final * 2) / 2

    return results, total

@app.route('/')
def index():
    return "Student Evaluation API is running!"

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    student_text = data.get('student_answer')
    teacher_data = data.get('teacher_answer')

    if not student_text or not teacher_data:
        return jsonify({"error": "Missing required input"}), 400

    try:
        results, total = grading(student_text, teacher_data)
        return jsonify({
            "total_score": total,
            "details": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("✅ API is starting...")
    print("🔐 GEMINI_API_KEY:", "SET" if API_KEY else "NOT SET")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
