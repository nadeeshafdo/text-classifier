from flask import Flask, request, jsonify, render_template_string
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer
MODEL_PATH = 'text_classifier_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

# arXiv category names mapping
CATEGORY_NAMES = {
    'astro-ph': 'Astrophysics',
    'astro-ph.CO': 'Cosmology and Nongalactic Astrophysics',
    'astro-ph.EP': 'Earth and Planetary Astrophysics',
    'astro-ph.GA': 'Astrophysics of Galaxies',
    'astro-ph.HE': 'High Energy Astrophysical Phenomena',
    'astro-ph.IM': 'Instrumentation and Methods for Astrophysics',
    'astro-ph.SR': 'Solar and Stellar Astrophysics',
    'cond-mat.dis-nn': 'Disordered Systems and Neural Networks',
    'cond-mat.mes-hall': 'Mesoscale and Nanoscale Physics',
    'cond-mat.mtrl-sci': 'Materials Science',
    'cond-mat.other': 'Other Condensed Matter',
    'cond-mat.quant-gas': 'Quantum Gases',
    'cond-mat.soft': 'Soft Condensed Matter',
    'cond-mat.stat-mech': 'Statistical Mechanics',
    'cond-mat.str-el': 'Strongly Correlated Electrons',
    'cond-mat.supr-con': 'Superconductivity',
    'cs.AI': 'Artificial Intelligence',
    'cs.AR': 'Hardware Architecture',
    'cs.CC': 'Computational Complexity',
    'cs.CE': 'Computational Engineering, Finance, and Science',
    'cs.CG': 'Computational Geometry',
    'cs.CL': 'Computation and Language',
    'cs.CR': 'Cryptography and Security',
    'cs.CV': 'Computer Vision and Pattern Recognition',
    'cs.CY': 'Computers and Society',
    'cs.DB': 'Databases',
    'cs.DC': 'Distributed, Parallel, and Cluster Computing',
    'cs.DL': 'Digital Libraries',
    'cs.DM': 'Discrete Mathematics',
    'cs.DS': 'Data Structures and Algorithms',
    'cs.ET': 'Emerging Technologies',
    'cs.FL': 'Formal Languages and Automata Theory',
    'cs.GL': 'General Literature',
    'cs.GR': 'Graphics',
    'cs.GT': 'Computer Science and Game Theory',
    'cs.HC': 'Human-Computer Interaction',
    'cs.IR': 'Information Retrieval',
    'cs.IT': 'Information Theory',
    'cs.LG': 'Machine Learning',
    'cs.LO': 'Logic in Computer Science',
    'cs.MA': 'Multiagent Systems',
    'cs.MM': 'Multimedia',
    'cs.MS': 'Mathematical Software',
    'cs.NA': 'Numerical Analysis',
    'cs.NE': 'Neural and Evolutionary Computing',
    'cs.NI': 'Networking and Internet Architecture',
    'cs.OH': 'Other Computer Science',
    'cs.OS': 'Operating Systems',
    'cs.PF': 'Performance',
    'cs.PL': 'Programming Languages',
    'cs.RO': 'Robotics',
    'cs.SC': 'Symbolic Computation',
    'cs.SD': 'Sound',
    'cs.SE': 'Software Engineering',
    'cs.SI': 'Social and Information Networks',
    'cs.SY': 'Systems and Control',
    'gr-qc': 'General Relativity and Quantum Cosmology',
    'hep-ex': 'High Energy Physics - Experiment',
    'hep-lat': 'High Energy Physics - Lattice',
    'hep-ph': 'High Energy Physics - Phenomenology',
    'hep-th': 'High Energy Physics - Theory',
    'math.AC': 'Commutative Algebra',
    'math.AG': 'Algebraic Geometry',
    'math.AP': 'Analysis of PDEs',
    'math.AT': 'Algebraic Topology',
    'math.CA': 'Classical Analysis and ODEs',
    'math.CO': 'Combinatorics',
    'math.CT': 'Category Theory',
    'math.CV': 'Complex Variables',
    'math.DG': 'Differential Geometry',
    'math.DS': 'Dynamical Systems',
    'math.FA': 'Functional Analysis',
    'math.GM': 'General Mathematics',
    'math.GN': 'General Topology',
    'math.GR': 'Group Theory',
    'math.GT': 'Geometric Topology',
    'math.HO': 'History and Overview',
    'math.IT': 'Information Theory',
    'math.KT': 'K-Theory and Homology',
    'math.LO': 'Logic',
    'math.MG': 'Metric Geometry',
    'math.MP': 'Mathematical Physics',
    'math.NA': 'Numerical Analysis',
    'math.NT': 'Number Theory',
    'math.OA': 'Operator Algebras',
    'math.OC': 'Optimization and Control',
    'math.PR': 'Probability',
    'math.QA': 'Quantum Algebra',
    'math.RA': 'Rings and Algebras',
    'math.RT': 'Representation Theory',
    'math.SG': 'Symplectic Geometry',
    'math.SP': 'Spectral Theory',
    'math.ST': 'Statistics Theory',
    'nlin.AO': 'Adaptation and Self-Organizing Systems',
    'nlin.CD': 'Chaotic Dynamics',
    'nlin.CG': 'Cellular Automata and Lattice Gases',
    'nlin.PS': 'Pattern Formation and Solitons',
    'nlin.SI': 'Exactly Solvable and Integrable Systems',
    'nucl-ex': 'Nuclear Experiment',
    'nucl-th': 'Nuclear Theory',
    'physics.acc-ph': 'Accelerator Physics',
    'physics.ao-ph': 'Atmospheric and Oceanic Physics',
    'physics.app-ph': 'Applied Physics',
    'physics.atm-clus': 'Atomic and Molecular Clusters',
    'physics.atom-ph': 'Atomic Physics',
    'physics.bio-ph': 'Biological Physics',
    'physics.chem-ph': 'Chemical Physics',
    'physics.class-ph': 'Classical Physics',
    'physics.comp-ph': 'Computational Physics',
    'physics.data-an': 'Data Analysis, Statistics and Probability',
    'physics.ed-ph': 'Physics Education',
    'physics.flu-dyn': 'Fluid Dynamics',
    'physics.gen-ph': 'General Physics',
    'physics.geo-ph': 'Geophysics',
    'physics.hist-ph': 'History and Philosophy of Physics',
    'physics.ins-det': 'Instrumentation and Detectors',
    'physics.med-ph': 'Medical Physics',
    'physics.optics': 'Optics',
    'physics.plasm-ph': 'Plasma Physics',
    'physics.pop-ph': 'Popular Physics',
    'physics.soc-ph': 'Physics and Society',
    'physics.space-ph': 'Space Physics',
    'q-bio.BM': 'Biomolecules',
    'q-bio.CB': 'Cell Behavior',
    'q-bio.GN': 'Genomics',
    'q-bio.MN': 'Molecular Networks',
    'q-bio.NC': 'Neurons and Cognition',
    'q-bio.OT': 'Other Quantitative Biology',
    'q-bio.PE': 'Populations and Evolution',
    'q-bio.QM': 'Quantitative Methods',
    'q-bio.SC': 'Subcellular Processes',
    'q-bio.TO': 'Tissues and Organs',
    'q-fin.CP': 'Computational Finance',
    'q-fin.EC': 'Economics',
    'q-fin.GN': 'General Finance',
    'q-fin.MF': 'Mathematical Finance',
    'q-fin.PM': 'Portfolio Management',
    'q-fin.PR': 'Pricing of Securities',
    'q-fin.RM': 'Risk Management',
    'q-fin.ST': 'Statistical Finance',
    'q-fin.TR': 'Trading and Market Microstructure',
    'quant-ph': 'Quantum Physics',
    'stat.AP': 'Applications',
    'stat.CO': 'Computation',
    'stat.ME': 'Methodology',
    'stat.ML': 'Machine Learning',
    'stat.OT': 'Other Statistics',
    'stat.TH': 'Statistics Theory',
}

def get_readable_category(category_code):
    """Convert category code to human-readable name."""
    return CATEGORY_NAMES.get(category_code, category_code)

try:
    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Model files not found!")
    print("Please run 'python main.py' first to train and save the model.")
    clf = None
    vectorizer = None

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text Classifier</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
            max-width: 700px;
            width: 100%;
        }
        h1 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-align: center;
        }
        .subtitle {
            color: #666;
            text-align: center;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            resize: vertical;
            min-height: 120px;
            font-family: inherit;
            transition: border-color 0.3s;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        button:active { transform: translateY(0); }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            display: none;
        }
        .result.show { display: block; }
        .prediction {
            font-size: 1.3em;
            color: #333;
            margin-bottom: 15px;
        }
        .category {
            color: #667eea;
            font-weight: bold;
            font-size: 1.5em;
            display: block;
            margin-bottom: 5px;
        }
        .category-code {
            color: #888;
            font-size: 0.7em;
            font-weight: normal;
            font-family: 'Courier New', monospace;
        }
        .confidence {
            color: #28a745;
            font-weight: bold;
        }
        .probabilities {
            margin-top: 20px;
        }
        .prob-item {
            margin: 10px 0;
        }
        .prob-label {
            font-weight: 600;
            color: #555;
            margin-bottom: 5px;
        }
        .prob-bar-container {
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            height: 30px;
            position: relative;
        }
        .prob-bar {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            color: white;
            font-weight: bold;
            transition: width 0.5s ease;
        }
        .error {
            color: #dc3545;
            text-align: center;
            margin-top: 20px;
        }
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        .loading.show { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Text Classifier</h1>
        <p class="subtitle">Classify scientific paper abstracts into arXiv categories</p>
        
        <textarea id="textInput" placeholder="Paste a scientific paper abstract here..."></textarea>
        
        <button onclick="classify()">🔍 Classify Text</button>
        
        <div class="loading" id="loading">
            <p>⏳ Analyzing...</p>
        </div>
        
        <div class="result" id="result">
            <div class="prediction">
                Category: <span class="category" id="category"></span>
            </div>
            <div class="prediction">
                Confidence: <span class="confidence" id="confidence"></span>
            </div>
            
            <div class="probabilities">
                <h3 style="margin-bottom: 15px; color: #555;">All Categories:</h3>
                <div id="probabilities"></div>
            </div>
        </div>
        
        <div class="error" id="error"></div>
    </div>

    <script>
        async function classify() {
            const text = document.getElementById('textInput').value.trim();
            const resultDiv = document.getElementById('result');
            const errorDiv = document.getElementById('error');
            const loadingDiv = document.getElementById('loading');
            
            // Reset
            resultDiv.classList.remove('show');
            errorDiv.textContent = '';
            
            if (!text) {
                errorDiv.textContent = '⚠️ Please enter some text to classify.';
                return;
            }
            
            // Show loading
            loadingDiv.classList.add('show');
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    errorDiv.textContent = '❌ ' + data.error;
                } else {
                    // Display results
                    document.getElementById('category').innerHTML = 
                        data.category_name + 
                        '<span class="category-code"> (' + data.category_code + ')</span>';
                    document.getElementById('confidence').textContent = 
                        (data.confidence * 100).toFixed(2) + '%';
                    
                    // Display all probabilities
                    const probsDiv = document.getElementById('probabilities');
                    probsDiv.innerHTML = '';
                    
                    // Sort probabilities
                    const sorted = Object.entries(data.probabilities)
                        .sort((a, b) => b[1] - a[1]);
                    
                    sorted.forEach(([cat, prob]) => {
                        // Parse category info
                        const parts = cat.split('|');
                        const catCode = parts[0];
                        const catName = parts[1] || catCode;
                        
                        const probItem = document.createElement('div');
                        probItem.className = 'prob-item';
                        probItem.innerHTML = `
                            <div class="prob-label">${catName} <span style="color: #999; font-size: 0.85em;">(${catCode})</span></div>
                            <div class="prob-bar-container">
                                <div class="prob-bar" style="width: ${prob * 100}%">
                                    ${(prob * 100).toFixed(1)}%
                                </div>
                            </div>
                        `;
                        probsDiv.appendChild(probItem);
                    });
                    
                    resultDiv.classList.add('show');
                }
            } catch (error) {
                errorDiv.textContent = '❌ Error: ' + error.message;
            } finally {
                loadingDiv.classList.remove('show');
            }
        }
        
        // Allow Enter key to submit (with Shift+Enter for new line)
        document.getElementById('textInput').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                classify();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Render the main page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions."""
    if clf is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Make prediction
        processed = vectorizer.transform([text])
        prediction = clf.predict(processed)[0]
        confidence = clf.predict_proba(processed).max()
        probabilities = clf.predict_proba(processed)[0]
        classes = clf.classes_
        
        # Create probability dictionary with readable names
        prob_dict = {f"{class_name}|{get_readable_category(class_name)}": float(prob) 
                     for class_name, prob in zip(classes, probabilities)}
        
        return jsonify({
            'category_code': str(prediction),
            'category_name': get_readable_category(str(prediction)),
            'confidence': float(confidence),
            'probabilities': prob_dict
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if clf is None or vectorizer is None:
        print("\n⚠️  Warning: Model not loaded!")
        print("Please run 'python main.py' first to train and save the model.\n")
    else:
        print("\n" + "=" * 70)
        print("🚀 Starting Text Classifier Web Interface")
        print("=" * 70)
        print("\n📍 Open your browser and go to: http://127.0.0.1:5000")
        print("\n💡 Press Ctrl+C to stop the server\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
