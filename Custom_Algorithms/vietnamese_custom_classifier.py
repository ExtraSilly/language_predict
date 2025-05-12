import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import os
import re
import time
import unicodedata
from typing import List, Dict, Tuple
import warnings
import pickle
warnings.filterwarnings('ignore')

# Danh sách stop words tiếng Việt
VIETNAMESE_STOP_WORDS = [
    'a', 'à', 'ả', 'ã', 'ạ', 'á', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ',
    'b', 'c', 'd', 'đ', 'e', 'è', 'ẻ', 'ẽ', 'ẹ', 'é', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ',
    'f', 'g', 'h', 'i', 'ì', 'ỉ', 'ĩ', 'ị', 'í',
    'j', 'k', 'l', 'm', 'n', 'o', 'ò', 'ỏ', 'õ', 'ọ', 'ó', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ',
    'p', 'q', 'r', 's', 't', 'u', 'ù', 'ủ', 'ũ', 'ụ', 'ú', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự',
    'v', 'w', 'x', 'y', 'ỳ', 'ỷ', 'ỹ', 'ỵ', 'ý',
    'z',
    # Các từ kết nối phổ biến
    'và', 'của', 'các', 'là', 'có', 'được', 'trong', 'cho', 'với', 'có', 'đã', 'đang', 'sẽ',
    'này', 'nọ', 'kia', 'đây', 'đó', 'kia', 'nào', 'bao', 'nhiêu', 'mấy',
    'tôi', 'chúng tôi', 'bạn', 'các bạn', 'anh', 'chị', 'em', 'họ', 'chúng', 'mình',
    'của', 'những', 'các', 'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười',
    'nhiều', 'ít', 'lớn', 'nhỏ', 'cao', 'thấp', 'dài', 'ngắn', 'rộng', 'hẹp',
    'đẹp', 'xấu', 'tốt', 'kém', 'hay', 'dở', 'đúng', 'sai', 'đủ', 'thiếu',
    'nhanh', 'chậm', 'sớm', 'muộn', 'gần', 'xa', 'trên', 'dưới', 'trong', 'ngoài',
    'trước', 'sau', 'giữa', 'bên', 'cạnh', 'đầu', 'cuối', 'giữa',
    'vì', 'nên', 'mà', 'nhưng', 'hoặc', 'hay', 'vậy', 'thế', 'nếu', 'thì',
    'để', 'cho', 'bởi', 'từ', 'theo', 'về', 'đến', 'ở', 'tại', 'trong',
    'cùng', 'với', 'cả', 'cả', 'chỉ', 'mỗi', 'mọi', 'tất cả', 'không', 'chưa',
    'đã', 'đang', 'sẽ', 'vừa', 'mới', 'vẫn', 'còn', 'hết', 'xong', 'rồi',
    'à', 'ừ', 'vâng', 'dạ', 'ạ', 'ơi', 'này', 'nọ', 'kia', 'đây', 'đó', 'kia',
    'nào', 'bao', 'nhiêu', 'mấy', 'sao', 'thế nào', 'tại sao', 'vì sao', 'làm sao',
    'bao giờ', 'khi nào', 'lúc nào', 'ở đâu', 'từ đâu', 'đến đâu', 'về đâu',
    'ai', 'gì', 'nào', 'đâu', 'bao nhiêu', 'mấy', 'sao', 'thế nào', 'tại sao',
    'vì sao', 'làm sao', 'bao giờ', 'khi nào', 'lúc nào', 'ở đâu', 'từ đâu',
    'đến đâu', 'về đâu', 'ai', 'gì', 'nào', 'đâu', 'bao nhiêu', 'mấy'
]

class VietnameseTextClassifier:
    def __init__(self):
        """Initialize the custom classifier"""
        self.categories = ['economics', 'education', 'politics', 'science', 
                          'culture', 'environment', 'health', 'psychology', 
                          'sociology', 'technology']
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words=VIETNAMESE_STOP_WORDS,
            token_pattern=r'(?u)\b\w+\b'  # Match single words
        )
        
        # Initialize Random Forest model with optimized hyperparameters
        self.model = RandomForestClassifier(
            n_estimators=300,        # More trees for better learning
            max_depth=50,            # Limit depth to prevent overfitting
            min_samples_split=2,     # Minimum samples required to split a node
            min_samples_leaf=2,      # Minimum samples required at a leaf node
            class_weight='balanced', # Automatically adjust weights inversely proportional to class frequencies
            random_state=42,         # For reproducibility
            n_jobs=-1                # Use all available cores for parallel processing
        )
        
        # Initialize probability model
        self.prob_model = {
            'classes': None,
            'probabilities': None
        }
        
        # Model paths
        self.model_dir = 'models'
        self.categories_path = os.path.join(self.model_dir, 'categories.pkl')
        self.model_path = os.path.join(self.model_dir, 'model.pkl')
        self.prob_model_path = os.path.join(self.model_dir, 'prob_model.pkl')
        self.vectorizer_path = os.path.join(self.model_dir, 'vectorizer.pkl')
        
        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def _is_vietnamese(self, text: str) -> bool:
        """Check if text is in Vietnamese using rule-based approach"""
        if not text or len(text.strip()) == 0:
            return False
            
        # Vietnamese-specific characters
        vietnamese_chars = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
        
        # Count Vietnamese characters
        viet_char_count = sum(1 for char in text if char in vietnamese_chars)
        
        # Count total word characters (excluding spaces and punctuation)
        total_char_count = sum(1 for char in text if char.isalpha())
        
        # If there are no alphabetic characters, return False
        if total_char_count == 0:
            return False
            
        # Calculate the ratio of Vietnamese characters to total alphabetic characters
        viet_char_ratio = viet_char_count / total_char_count
        
        # Common Vietnamese words
        common_viet_words = ['và', 'của', 'các', 'là', 'có', 'được', 'trong', 'cho', 'với', 
                            'đã', 'đang', 'sẽ', 'này', 'nọ', 'kia', 'đây', 'đó', 'nào', 
                            'bao', 'nhiêu', 'mấy', 'tôi', 'bạn', 'anh', 'chị', 'em', 'họ', 
                            'chúng', 'mình', 'những', 'một', 'hai', 'ba', 'bốn', 'năm', 
                            'sáu', 'bảy', 'tám', 'chín', 'mười']
        
        # Count common Vietnamese words
        word_count = 0
        for word in common_viet_words:
            if word in text.lower():
                word_count += 1
        
        # If there are Vietnamese characters or common Vietnamese words, consider it Vietnamese
        return viet_char_ratio > 0.05 or word_count >= 2
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess Vietnamese text with comprehensive normalization:
        1. Normalize Unicode (NFC form)
        2. Convert to lowercase
        3. Remove special characters, keep meaningful words
        4. Normalize whitespace
        5. Apply domain-specific markers
        """
        # Normalize Unicode to NFC form
        text = unicodedata.normalize('NFC', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters, keep Vietnamese characters, numbers, and spaces
        text = re.sub(r'[^a-zA-Z0-9áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ\s]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Store capitalized words before lowercase conversion
        capitalized_words = re.findall(r'\b[A-Z][A-ZÀ-Ỹ]+\b', text)  # Fully capitalized words
        proper_nouns = re.findall(r'\b[A-Z][a-zà-ỹ]+\b', text)  # Words starting with capital
        
        # Add special markers for originally capitalized words
        for word in capitalized_words:
            word_lower = word.lower()
            text = re.sub(
                r'\b' + word_lower + r'\b',
                f'CAPS_{word_lower}_CAPS',
                text
            )
        
        for word in proper_nouns:
            word_lower = word.lower()
            text = re.sub(
                r'\b' + word_lower + r'\b',
                f'PROPER_{word_lower}_PROPER',
                text
            )
            
        # Add domain-specific markers with enhanced keywords
        domain_terms = {
            # Economic terms with higher weights
            'kinh tế': 'ECONOMIC_TERM ECONOMIC_KEYWORD ECONOMIC_MAIN',
            'gdp': 'ECONOMIC_TERM ECONOMIC_KEYWORD ECONOMIC_MAIN',
            'tăng trưởng': 'ECONOMIC_TERM ECONOMIC_KEYWORD ECONOMIC_MAIN',
            'thị trường': 'ECONOMIC_TERM ECONOMIC_KEYWORD ECONOMIC_MAIN',
            'tài chính': 'ECONOMIC_TERM ECONOMIC_KEYWORD ECONOMIC_MAIN',
            'tiền tệ': 'ECONOMIC_TERM ECONOMIC_KEYWORD ECONOMIC_MAIN',
            'ngân hàng': 'ECONOMIC_TERM ECONOMIC_KEYWORD ECONOMIC_MAIN',
            'đầu tư': 'ECONOMIC_TERM ECONOMIC_KEYWORD ECONOMIC_MAIN',
            'chứng khoán': 'ECONOMIC_TERM ECONOMIC_KEYWORD ECONOMIC_MAIN',
            'lạm phát': 'ECONOMIC_TERM ECONOMIC_KEYWORD ECONOMIC_MAIN',
            'tỷ giá': 'ECONOMIC_TERM ECONOMIC_KEYWORD ECONOMIC_MAIN',
            'thuế': 'ECONOMIC_TERM ECONOMIC_KEYWORD ECONOMIC_MAIN',
            'doanh nghiệp': 'ECONOMIC_TERM ECONOMIC_KEYWORD ECONOMIC_MAIN',
            'thương mại': 'ECONOMIC_TERM ECONOMIC_KEYWORD ECONOMIC_MAIN',
            'xuất khẩu': 'ECONOMIC_TERM ECONOMIC_KEYWORD ECONOMIC_MAIN',
            'nhập khẩu': 'ECONOMIC_TERM ECONOMIC_KEYWORD ECONOMIC_MAIN',
            'dịch vụ tài chính': 'ECONOMIC_TERM ECONOMIC_KEYWORD ECONOMIC_MAIN',
            'chuyên gia kinh tế': 'ECONOMIC_TERM ECONOMIC_KEYWORD ECONOMIC_MAIN',
            'báo cáo kinh tế': 'ECONOMIC_TERM ECONOMIC_KEYWORD ECONOMIC_MAIN',
            'phục hồi kinh tế': 'ECONOMIC_TERM ECONOMIC_KEYWORD ECONOMIC_MAIN',
            'tăng trưởng kinh tế': 'ECONOMIC_TERM ECONOMIC_KEYWORD ECONOMIC_MAIN',
            
            # Technology terms with more specific markers
            'công nghệ': 'TECH_TERM TECH_KEYWORD',
            'kỹ thuật': 'TECH_TERM TECH_KEYWORD',
            'phần mềm': 'TECH_TERM TECH_SPECIFIC',
            'ứng dụng': 'TECH_TERM TECH_SPECIFIC',
            'máy tính': 'TECH_TERM TECH_SPECIFIC',
            'điện thoại': 'TECH_TERM TECH_SPECIFIC',
            'internet': 'TECH_TERM TECH_SPECIFIC',
            'trí tuệ nhân tạo': 'TECH_TERM TECH_SPECIFIC',
            'blockchain': 'TECH_TERM TECH_SPECIFIC',
            'dữ liệu': 'TECH_TERM TECH_SPECIFIC',
            'hệ thống': 'TECH_TERM TECH_SPECIFIC',
            'lập trình': 'TECH_TERM TECH_SPECIFIC',
            'thiết bị': 'TECH_TERM TECH_SPECIFIC',
            'mạng': 'TECH_TERM TECH_SPECIFIC',
            'bảo mật': 'TECH_TERM TECH_SPECIFIC',
            
            # Other domain terms
            'chính trị': 'POLITICAL_TERM',
            'quốc hội': 'POLITICAL_TERM',
            'chính phủ': 'POLITICAL_TERM',
            'môi trường': 'ENVIRONMENT_TERM',
            'sinh thái': 'ENVIRONMENT_TERM',
            'sức khỏe': 'HEALTH_TERM',
            'bệnh viện': 'HEALTH_TERM',
            'tâm lý': 'PSYCHOLOGY_TERM',
            'tinh thần': 'PSYCHOLOGY_TERM',
            'xã hội': 'SOCIOLOGY_TERM',
            'cộng đồng': 'SOCIOLOGY_TERM'
        }
        
        # Add context markers for important phrases
        context_phrases = {
            # Economic professions
            'nhà kinh tế': 'ECONOMIC_PROFESSION ECONOMIC_MAIN',
            'chuyên gia tài chính': 'ECONOMIC_PROFESSION ECONOMIC_MAIN',
            'nhà đầu tư': 'ECONOMIC_PROFESSION ECONOMIC_MAIN',
            'doanh nhân': 'ECONOMIC_PROFESSION ECONOMIC_MAIN',
            'chuyên viên tài chính': 'ECONOMIC_PROFESSION ECONOMIC_MAIN',
            'nhà phân tích kinh tế': 'ECONOMIC_PROFESSION ECONOMIC_MAIN',
            
            # Technology professions
            'kỹ sư công nghệ': 'TECH_PROFESSION TECH_SPECIFIC',
            'lập trình viên': 'TECH_PROFESSION TECH_SPECIFIC',
            'nhà phát triển': 'TECH_PROFESSION TECH_SPECIFIC',
            'chuyên gia công nghệ': 'TECH_PROFESSION TECH_SPECIFIC',
            'kỹ sư phần mềm': 'TECH_PROFESSION TECH_SPECIFIC'
        }
        
        # Apply domain terms
        for term, marker in domain_terms.items():
            text = text.replace(term, f'{term} {marker}')
            
        # Apply context phrases
        for phrase, marker in context_phrases.items():
            text = text.replace(phrase, f'{phrase} {marker}')
        
        # Final whitespace normalization
        text = ' '.join(text.split())
        return text.strip()
    
    def load_data(self, data_dir: str) -> Tuple[List[str], List[int]]:
        texts = []
        labels = []
        
        print("\nLoading data from categories:")
        category_counts = {category: 0 for category in self.categories}
        
        for category in self.categories:
            category_dir = os.path.join(data_dir, category)
            if not os.path.exists(category_dir):
                print(f"Warning: Directory {category_dir} does not exist")
                continue
                
            for filename in os.listdir(category_dir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(category_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            paragraphs = re.split(r'\d+\.', content)[1:]
                            for paragraph in paragraphs:
                                if paragraph.strip() and self._is_vietnamese(paragraph):
                                    processed_text = self._preprocess_text(paragraph)
                                    if processed_text and len(processed_text.split()) >= 3:  # Minimum 3 words
                                        texts.append(processed_text)
                                        labels.append(self.categories.index(category))
                                        category_counts[category] += 1
                    except Exception as e:
                        print(f"Error reading {file_path}: {str(e)}")
        
        for category, count in category_counts.items():
            print(f"- {category}: {count} samples")
        
        return texts, labels
    
    def train(self, texts: List[str], labels: List[int], save_model: bool = True) -> None:
        start_time = time.time()
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print("\nVectorizing texts...")
        X_train = self.vectorizer.fit_transform(train_texts)
        X_val = self.vectorizer.transform(val_texts)
        
        print("\nTraining Random Forest model...")
        train_labels = np.array(train_labels)
        val_labels = np.array(val_labels)
        
        # Huấn luyện mô hình
        self.model.fit(X_train, train_labels)
        
        # Tạo mô hình có khả năng dự đoán xác suất
        self.prob_model = {
            'classes': self.model.classes_,
            'probabilities': self.model.predict_proba(X_train)
        }
        
        print("\nModel Evaluation:")
        val_preds = self.model.predict(X_val)
        print(classification_report(val_labels, val_preds, target_names=self.categories))
        
        # Lưu mô hình nếu được yêu cầu
        if save_model:
            self.save_model()
            
        end_time = time.time()
        print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    
    def predict(self, text: str) -> Dict:
        if not self._is_vietnamese(text):
            return {"error": "Text is not in Vietnamese", "original_text": text}
        
        if not hasattr(self, 'model') or self.model is None:
            return {"error": "Model has not been trained yet", "original_text": text}
        
        processed_text = self._preprocess_text(text)
        if not processed_text:
            return {"error": "Text is empty after preprocessing", "original_text": text}
        
        try:
            X = self.vectorizer.transform([processed_text])
            
            # Get prediction and probabilities
            prediction = int(self.model.predict(X)[0])
            probabilities = self.model.predict_proba(X)[0]
            
            # Get top 3 categories with highest probabilities
            category_probs = list(zip(self.categories, probabilities))
            category_probs.sort(key=lambda x: x[1], reverse=True)
            top_3_categories = category_probs[:3]
            
            result = {
                "category": self.categories[prediction],
                "probabilities": {
                    category: float(prob) * 100
                    for category, prob in top_3_categories
                },
                "top_3_categories": [
                    {"category": category, "probability": float(prob) * 100}
                    for category, prob in top_3_categories
                ],
                "original_text": text
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Error during prediction: {str(e)}", "original_text": text}
    
    def save_model(self, model_dir: str = 'models') -> None:
        """Lưu mô hình đã huấn luyện vào thư mục models"""
        # Tạo thư mục models nếu chưa tồn tại
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Lưu vectorizer
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            
        # Lưu mô hình
        model_path = os.path.join(model_dir, 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
            
        # Lưu mô hình xác suất
        prob_model_path = os.path.join(model_dir, 'prob_model.pkl')
        with open(prob_model_path, 'wb') as f:
            pickle.dump(self.prob_model, f)
            
        # Lưu danh sách categories
        categories_path = os.path.join(model_dir, 'categories.pkl')
        with open(categories_path, 'wb') as f:
            pickle.dump(self.categories, f)
            
        print(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir: str = 'models') -> bool:
        """Tải mô hình đã huấn luyện từ thư mục models"""
        # Kiểm tra xem các file cần thiết có tồn tại không
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        model_path = os.path.join(model_dir, 'model.pkl')
        categories_path = os.path.join(model_dir, 'categories.pkl')
        
        if not all(os.path.exists(path) for path in [vectorizer_path, model_path, categories_path]):
            print("Model files not found. Please train the model first.")
            return False
            
        try:
            # Tải vectorizer
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
                
            # Tải danh sách categories
            with open(categories_path, 'rb') as f:
                self.categories = pickle.load(f)
                
            # Kiểm tra loại mô hình đã lưu
            with open(model_path, 'rb') as f:
                saved_model = pickle.load(f)
                
            # Nếu mô hình đã lưu không phải là RandomForestClassifier, cần huấn luyện lại
            if not isinstance(saved_model, RandomForestClassifier):
                print("Saved model is not a RandomForestClassifier. Retraining with RandomForestClassifier...")
                return False
                
            self.model = saved_model
            print(f"Model loaded from {model_dir}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def predict_paragraph(self, paragraph: str) -> List[Dict]:
        """
        Phân loại từng câu trong một đoạn văn dài.
        
        Args:
            paragraph: Đoạn văn cần phân loại
            
        Returns:
            List[Dict]: Danh sách kết quả phân loại cho từng câu
        """
        if not self._is_vietnamese(paragraph):
            return [{"error": "Text is not in Vietnamese", "original_text": paragraph}]
        
        # Tách đoạn văn thành các câu
        # Sử dụng regex để tách câu, xử lý các trường hợp đặc biệt
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        # Lọc bỏ các câu trống
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [{"error": "No valid sentences found in the paragraph", "original_text": paragraph}]
        
        results = []
        for sentence in sentences:
            # Bỏ qua các câu quá ngắn (ít hơn 3 từ)
            if len(sentence.split()) < 3:
                continue
                
            # Sử dụng phương thức predict hiện có để phân loại từng câu
            result = self.predict(sentence)
            results.append(result)
        
        return results

def main():
    print(f"\n{'='*50}")
    print("Using scikit-learn Random Forest Implementation")
    print('='*50)
    
    classifier = VietnameseTextClassifier()
    
    # Kiểm tra xem có thể tải mô hình đã huấn luyện không
    if classifier.load_model():
        print("Using pre-trained model")
    else:
        print("Training new model")
        texts, labels = classifier.load_data('data')
        
        if not texts:
            print("Error: No valid Vietnamese texts found in the data directory")
            return
        
        classifier.train(texts, labels)
    
    # Chức năng nhập văn bản từ bàn phím
    print("\n" + "="*50)
    print("Nhập văn bản để phân loại (nhấn Enter hai lần để kết thúc):")
    print("="*50)
    
    while True:
        try:
            # Nhập văn bản từ bàn phím
            print("\nNhập văn bản tiếng Việt (nhấn Enter hai lần để kết thúc):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            
            # Kiểm tra nếu người dùng muốn thoát
            if not lines:
                print("Kết thúc chương trình.")
                break
            
            # Kết hợp các dòng thành một văn bản
            user_text = "\n".join(lines)
            
            # Phân loại văn bản
            print("\nKết quả phân loại:")
            print(f"Văn bản gốc:\n{user_text}")
            
            # Kiểm tra xem văn bản có chứa nhiều câu không
            sentences = re.split(r'(?<=[.!?])\s+', user_text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) > 1:
                # Nếu có nhiều câu, sử dụng predict_paragraph
                results = classifier.predict_paragraph(user_text)
                
                for i, result in enumerate(results):
                    if "error" in result:
                        print(f"Câu {i+1}: {result['error']}")
                    else:
                        print(f"\nCâu {i+1}: {result['original_text']}")
                        print(f"Danh mục: {result['category']}")
                        print("Top 3 danh mục:")
                        for cat in result['top_3_categories']:
                            print(f"- {cat['category']}: {cat['probability']:.2f}%")
            else:
                # Nếu chỉ có một câu, sử dụng predict
                result = classifier.predict(user_text)
                
                if "error" in result:
                    print(f"Lỗi: {result['error']}")
                else:
                    print(f"Danh mục: {result['category']}")
                    print("Top 3 danh mục:")
                    for cat in result['top_3_categories']:
                        print(f"- {cat['category']}: {cat['probability']:.2f}%")
            
            print("\n" + "="*50)
            
        except KeyboardInterrupt:
            print("\nChương trình đã bị dừng bởi người dùng.")
            break
        except Exception as e:
            print(f"\nLỗi: {str(e)}")
            print("Vui lòng thử lại.")

if __name__ == "__main__":
    main()