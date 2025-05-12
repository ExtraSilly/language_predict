import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import os
import re
from typing import List, Dict, Tuple
import warnings
import pickle
warnings.filterwarnings('ignore')

# Vietnamese stop words - split compound words
VIETNAMESE_STOP_WORDS = [
    'bị', 'bởi', 'cả', 'các', 'cái', 'cần', 'càng', 'chỉ', 'chiếc', 'cho', 'chứ', 'chưa', 'chuyện',
    'có', 'có', 'thể', 'cứ', 'của', 'cùng', 'cũng', 'đã', 'đang', 'đây', 'để', 'đến', 'nỗi', 'đều', 'điều',
    'do', 'đó', 'được', 'dưới', 'gì', 'khi', 'không', 'là', 'lại', 'lên', 'lúc', 'mà', 'mỗi', 'một', 'cách',
    'này', 'nên', 'nếu', 'ngay', 'nhiều', 'như', 'nhưng', 'những', 'nơi', 'nữa', 'phải', 'qua', 'ra',
    'rằng', 'rất', 'rồi', 'sau', 'sẽ', 'so', 'sự', 'tại', 'theo', 'thì', 'trên', 'trước', 'từ', 'từng',
    'và', 'vẫn', 'vào', 'vậy', 'vì', 'việc', 'với', 'vừa', 'thể'
]

class VietnameseSVMClassifier:
    def __init__(self):
        """Initialize the SVM classifier"""
        self.categories = ['economics', 'education', 'politics', 'science', 
                          'culture', 'environment', 'health', 'psychology', 
                          'sociology', 'technology']
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words=VIETNAMESE_STOP_WORDS,
            token_pattern=r'(?u)\b\w+\b'  # Match single words
        )
        
        # Initialize SVM model
        self.model = SVC(kernel='linear', probability=True)
        
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
        """Preprocess Vietnamese text with special handling for capitalized words"""
        # Store capitalized words before lowercase conversion
        capitalized_words = re.findall(r'\b[A-Z][A-ZÀ-Ỹ]+\b', text)  # Fully capitalized words
        proper_nouns = re.findall(r'\b[A-Z][a-zà-ỹ]+\b', text)  # Words starting with capital
        
        # Remove numbers and special characters but keep spaces
        text = re.sub(r'\d+\.', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
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
            # Culture terms
            'văn hóa': 'CULTURE_TERM CULTURE_KEYWORD',
            'nghệ thuật': 'CULTURE_TERM CULTURE_KEYWORD',
            'văn học': 'CULTURE_TERM CULTURE_KEYWORD',
            'nhà văn': 'CULTURE_TERM CULTURE_KEYWORD',
            'nhà thơ': 'CULTURE_TERM CULTURE_KEYWORD',
            'tác phẩm': 'CULTURE_TERM CULTURE_KEYWORD',
            'văn chương': 'CULTURE_TERM CULTURE_KEYWORD',
            'nghệ sĩ': 'CULTURE_TERM CULTURE_KEYWORD',
            'sáng tác': 'CULTURE_TERM CULTURE_KEYWORD',
            'văn xuôi': 'CULTURE_TERM CULTURE_KEYWORD',
            'thơ ca': 'CULTURE_TERM CULTURE_KEYWORD',
            'truyện': 'CULTURE_TERM CULTURE_KEYWORD',
            'tiểu thuyết': 'CULTURE_TERM CULTURE_KEYWORD',
            'vở kịch': 'CULTURE_TERM CULTURE_KEYWORD',
            'phim ảnh': 'CULTURE_TERM CULTURE_KEYWORD',
            'âm nhạc': 'CULTURE_TERM CULTURE_KEYWORD',
            'hội họa': 'CULTURE_TERM CULTURE_KEYWORD',
            'điêu khắc': 'CULTURE_TERM CULTURE_KEYWORD',
            'kiến trúc': 'CULTURE_TERM CULTURE_KEYWORD',
            'di sản': 'CULTURE_TERM CULTURE_KEYWORD',
            
            # Science terms
            'khoa học': 'SCIENCE_TERM SCIENCE_KEYWORD',
            'nghiên cứu': 'SCIENCE_TERM SCIENCE_KEYWORD',
            'nhà khoa học': 'SCIENCE_TERM SCIENCE_KEYWORD',
            'phòng thí nghiệm': 'SCIENCE_TERM SCIENCE_KEYWORD',
            'thí nghiệm': 'SCIENCE_TERM SCIENCE_KEYWORD',
            'khám phá': 'SCIENCE_TERM SCIENCE_KEYWORD',
            'phát minh': 'SCIENCE_TERM SCIENCE_KEYWORD',
            'khoa học tự nhiên': 'SCIENCE_TERM SCIENCE_KEYWORD',
            'vật lý': 'SCIENCE_TERM SCIENCE_KEYWORD',
            'hóa học': 'SCIENCE_TERM SCIENCE_KEYWORD',
            'sinh học': 'SCIENCE_TERM SCIENCE_KEYWORD',
            
            # Education terms
            'giáo dục': 'EDUCATION_TERM EDUCATION_KEYWORD',
            'học sinh': 'EDUCATION_TERM EDUCATION_KEYWORD',
            'trường học': 'EDUCATION_TERM EDUCATION_KEYWORD',
            'giáo viên': 'EDUCATION_TERM EDUCATION_KEYWORD',
            'học tập': 'EDUCATION_TERM EDUCATION_KEYWORD',
            'giảng dạy': 'EDUCATION_TERM EDUCATION_KEYWORD',
            'đào tạo': 'EDUCATION_TERM EDUCATION_KEYWORD',
            'chương trình học': 'EDUCATION_TERM EDUCATION_KEYWORD',
            'môn học': 'EDUCATION_TERM EDUCATION_KEYWORD',
            
            # Other domain terms
            'kinh tế': 'ECONOMIC_TERM ECONOMIC_KEYWORD',
            'gdp': 'ECONOMIC_TERM ECONOMIC_KEYWORD',
            'thị trường': 'ECONOMIC_TERM ECONOMIC_KEYWORD',
            'tài chính': 'ECONOMIC_TERM ECONOMIC_KEYWORD',
            'tiền tệ': 'ECONOMIC_TERM ECONOMIC_KEYWORD',
            'ngân hàng': 'ECONOMIC_TERM ECONOMIC_KEYWORD',
            'đầu tư': 'ECONOMIC_TERM ECONOMIC_KEYWORD',
            'chứng khoán': 'ECONOMIC_TERM ECONOMIC_KEYWORD',
            'lạm phát': 'ECONOMIC_TERM ECONOMIC_KEYWORD',
            'tỷ giá': 'ECONOMIC_TERM ECONOMIC_KEYWORD',
            'thuế': 'ECONOMIC_TERM ECONOMIC_KEYWORD',
            'doanh nghiệp': 'ECONOMIC_TERM ECONOMIC_KEYWORD',
            'thương mại': 'ECONOMIC_TERM ECONOMIC_KEYWORD',
            'xuất khẩu': 'ECONOMIC_TERM ECONOMIC_KEYWORD',
            'nhập khẩu': 'ECONOMIC_TERM ECONOMIC_KEYWORD',
            'bitcoin': 'ECONOMIC_TERM CRYPTO_KEYWORD',
            'tiền mã hóa': 'ECONOMIC_TERM CRYPTO_KEYWORD',
            'blockchain': 'ECONOMIC_TERM CRYPTO_KEYWORD',
            'chính trị': 'POLITICAL_TERM',
            'quốc hội': 'POLITICAL_TERM',
            'chính phủ': 'POLITICAL_TERM',
            'công nghệ': 'TECH_TERM',
            'kỹ thuật': 'TECH_TERM',
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
            # Culture professions
            'nhà văn': 'CULTURE_PROFESSION',
            'nhà thơ': 'CULTURE_PROFESSION',
            'nghệ sĩ': 'CULTURE_PROFESSION',
            'nhạc sĩ': 'CULTURE_PROFESSION',
            'họa sĩ': 'CULTURE_PROFESSION',
            'đạo diễn': 'CULTURE_PROFESSION',
            'diễn viên': 'CULTURE_PROFESSION',
            'nhà điêu khắc': 'CULTURE_PROFESSION',
            'kiến trúc sư': 'CULTURE_PROFESSION',
            
            # Science professions
            'nhà khoa học': 'SCIENCE_PROFESSION',
            'nhà nghiên cứu': 'SCIENCE_PROFESSION',
            'nghiên cứu sinh': 'SCIENCE_ROLE',
            
            # Education professions
            'giáo viên': 'EDUCATION_PROFESSION',
            'nhà giáo': 'EDUCATION_PROFESSION',
            'giảng viên': 'EDUCATION_PROFESSION',
            'sinh viên': 'EDUCATION_ROLE',
            'học sinh': 'EDUCATION_ROLE',
            
            # Economic professions
            'nhà kinh tế': 'ECONOMIC_PROFESSION',
            'chuyên gia tài chính': 'ECONOMIC_PROFESSION',
            'nhà đầu tư': 'ECONOMIC_PROFESSION',
            'doanh nhân': 'ECONOMIC_PROFESSION',
            'học sinh': 'EDUCATION_ROLE'
        }
        
        # Apply domain terms
        for term, marker in domain_terms.items():
            text = text.replace(term, f'{term} {marker}')
            
        # Apply context phrases
        for phrase, marker in context_phrases.items():
            text = text.replace(phrase, f'{phrase} {marker}')
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def _calculate_class_weights(self, labels: List[int]) -> Dict[int, float]:
        """Calculate class weights based on the distribution of labels"""
        
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        return dict(zip(np.unique(labels), class_weights))
    
    def load_data(self, data_dir: str) -> Tuple[List[str], List[int]]:
        """Load and preprocess the data"""
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
                                    if processed_text:
                                        texts.append(processed_text)
                                        labels.append(self.categories.index(category))
                                        category_counts[category] += 1
                    except Exception as e:
                        print(f"Error reading {file_path}: {str(e)}")
        
        for category, count in category_counts.items():
            print(f"- {category}: {count} samples")
        
        return texts, labels
    
    def train(self, texts: List[str], labels: List[int], save_model: bool = True) -> None:
        """Train the SVM classifier"""
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Vectorize texts
        print("\nVectorizing texts...")
        X_train = self.vectorizer.fit_transform(train_texts)
        X_val = self.vectorizer.transform(val_texts)
        
        # Train SVM
        print("\nTraining SVM classifier...")
        self.model.fit(X_train, train_labels)
        
        # Initialize probability model after training
        self.prob_model['classes'] = self.model.classes_
        
        # Evaluate
        val_preds = self.model.predict(X_val)
        accuracy = accuracy_score(val_labels, val_preds)
        print(f"Validation Accuracy: {accuracy:.4f}")
        
        # Final evaluation
        print("\nModel Evaluation:")
        print(classification_report(val_labels, val_preds, target_names=self.categories))
        
        # Save model if requested
        if save_model:
            self.save_model()
    
    def save_model(self):
        """Save the trained model components"""
        if self.model is None or not hasattr(self.model, 'classes_'):
            print("No trained model to save!")
            return
            
        print("\nSaving model components...")
        
        # Save categories
        with open(self.categories_path, 'wb') as f:
            pickle.dump(self.categories, f)
            
        # Save vectorizer
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            
        # Save main model
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
            
        # Save probability model (for future use)
        with open(self.prob_model_path, 'wb') as f:
            pickle.dump(self.prob_model, f)
            
        print(f"All model components saved to {self.model_dir}/")
        
    def load_model(self) -> bool:
        """Load all model components"""
        required_files = [
            self.categories_path,
            self.vectorizer_path,
            self.model_path,
            self.prob_model_path
        ]
        
        if not all(os.path.exists(f) for f in required_files):
            print("Some model components are missing!")
            return False
            
        print("\nLoading model components...")
        try:
            # Load categories
            with open(self.categories_path, 'rb') as f:
                self.categories = pickle.load(f)
                
            # Load vectorizer
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
                
            # Load main model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
                
            # Load probability model
            with open(self.prob_model_path, 'rb') as f:
                self.prob_model = pickle.load(f)
                
            print("All model components loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model components: {str(e)}")
            return False
            
    def predict(self, text: str) -> Dict:
        """Predict the category of a text"""
        if not self._is_vietnamese(text):
            return {"error": "Text is not in Vietnamese"}
        
        # Preprocess and vectorize text
        processed_text = self._preprocess_text(text)
        if not processed_text:
            return {"error": "Text is empty after preprocessing"}
        
        X = self.vectorizer.transform([processed_text])
        
        # Get prediction and probabilities
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Update probability model if it exists
        if hasattr(self, 'prob_model'):
            self.prob_model['probabilities'] = probabilities.tolist()
        
        # Get top 3 categories with highest probabilities
        category_probs = list(zip(self.categories, probabilities))
        category_probs.sort(key=lambda x: x[1], reverse=True)
        top_3_categories = category_probs[:3]
        
        # Format results
        result = {
            "category": self.categories[prediction],
            "probabilities": {
                category: float(prob) * 100
                for category, prob in top_3_categories
            }
        }
        
        return result

def main():
    classifier = VietnameseSVMClassifier()
    
    # Try to load existing model first
    if classifier.load_model():
        print("Đã tải model thành công, sẵn sàng phân loại!")
    else:
        print("Không tìm thấy model đã lưu. Bắt đầu training model mới...")
        texts, labels = classifier.load_data('data')
        
        if not texts:
            print("Error: No valid Vietnamese texts found in the data directory")
            return
        
        classifier.train(texts, labels)
    
    while True:
        print("\nNhập văn bản cần phân loại (hoặc 'q' để thoát):")
        user_input = input().strip()
        
        if user_input.lower() == 'q':
            break
            
        if not user_input:
            print("Văn bản không được để trống!")
            continue
            
        # Split input into sentences
        sentences = re.split(r'[.!?]+', user_input)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            print("Không tìm thấy câu nào trong văn bản!")
            continue
            
        print("\nKết quả phân loại:")
        print("-" * 50)
        
        for i, sentence in enumerate(sentences, 1):
            result = classifier.predict(sentence)
            if "error" in result:
                print(f"\nCâu {i}: {sentence}")
                print(f"Lỗi: {result['error']}")
                continue
                
            print(f"\nCâu {i}: {sentence}")
            print(f"Phân loại: {result['category']}")
            print("Top 3 xác suất:")
            for category, prob in result['probabilities'].items():
                print(f"- {category}: {prob:.2f}%")
            print("-" * 50)

if __name__ == "__main__":
    main() 