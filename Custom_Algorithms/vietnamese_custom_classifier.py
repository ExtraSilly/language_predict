import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import os
import re
import langid
from typing import List, Dict, Tuple
import warnings
import pickle
import time
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
        self.categories = ['economics', 'education', 'politics', 'science', 'technology', 'health', 'environment', 'culture', 'psychology', 'sociology']
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # Giảm số lượng đặc trưng để tăng tốc độ
            ngram_range=(1, 2),  # Giảm xuống bigrams thay vì trigrams
            min_df=2,           
            max_df=0.95,        
            stop_words=VIETNAMESE_STOP_WORDS
        )
        
        # Sử dụng LinearSVC từ scikit-learn thay vì triển khai thủ công
        self.model = LinearSVC(
            C=1.0,
            max_iter=1000,
            dual=False,  # Sử dụng primal formulation để tăng tốc độ
            random_state=42
        )
        
        # Thêm biến để lưu trữ trạng thái đã huấn luyện
        self.is_trained = False
        
    def _is_vietnamese(self, text: str) -> bool:
        lang, _ = langid.classify(text)
        return lang == 'vi'
    
    def _preprocess_text(self, text: str) -> str:
        # Convert to lowercase
        text = text.lower()
        
        # Remove numbers and special characters but keep Vietnamese diacritics
        text = re.sub(r'\d+\.', '', text)
        text = re.sub(r'[^\w\sáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', ' ', text)
        
        # Remove extra whitespace and normalize spaces
        text = ' '.join(text.split())
        
        # Remove very short words (less than 2 characters)
        words = text.split()
        words = [word for word in words if len(word) > 1]
        
        # Thêm xử lý từ khóa quan trọng
        # Giữ lại các từ khóa quan trọng liên quan đến từng danh mục
        important_keywords = {
            'politics': {
                'chính trị': 2.0, 'chính phủ': 2.0, 'quốc hội': 2.0, 'đảng': 2.0,
                'chính sách': 1.5, 'đối ngoại': 1.5, 'nội chính': 1.5, 'pháp luật': 1.5,
                'luật': 1.5, 'tòa án': 1.5, 'luật sư': 1.5, 'quy định': 1.5,
                'vi phạm': 1.5, 'xử lý': 1.5, 'nghị quyết': 1.5, 'quyết định': 1.5,
                'chỉ thị': 1.5, 'thông tư': 1.5, 'nghị định': 1.5, 'văn bản': 1.5,
                'hành chính': 1.5, 'cải cách': 1.5, 'dân chủ': 1.5, 'nhân quyền': 1.5,
                'tự do': 1.5, 'công bằng': 1.5, 'bình đẳng': 1.5, 'đoàn kết': 1.5,
                'hòa bình': 1.5, 'an ninh': 1.5, 'quốc phòng': 1.5, 'ngoại giao': 1.5,
                'hợp tác': 1.5, 'phát triển': 1.5, 'xây dựng': 1.5, 'đổi mới': 1.5,
                'cải thiện': 1.5, 'nâng cao': 1.5, 'thúc đẩy': 1.5, 'tăng cường': 1.5,
                'mở rộng': 1.5, 'củng cố': 1.5, 'duy trì': 1.5, 'bảo vệ': 1.5,
                'đảm bảo': 1.5, 'thực hiện': 1.5, 'triển khai': 1.5, 'quản lý': 1.5,
                'điều hành': 1.5, 'lãnh đạo': 1.5, 'chỉ đạo': 1.5, 'kiểm tra': 1.5,
                'giám sát': 1.5, 'thanh tra': 1.5, 'kiểm toán': 1.5
            },
            'economics': {
                'kinh tế': 2.0, 'tài chính': 2.0, 'ngân hàng': 2.0, 'thương mại': 2.0,
                'xuất khẩu': 1.5, 'nhập khẩu': 1.5, 'đầu tư': 1.5, 'du lịch': 1.5,
                'khách sạn': 1.5, 'tour': 1.5, 'du khách': 1.5, 'thăm quan': 1.5,
                'điểm đến': 1.5, 'phượt': 1.5, 'nông nghiệp': 1.5, 'nông dân': 1.5,
                'trồng trọt': 1.5, 'chăn nuôi': 1.5, 'nông sản': 1.5, 'thu hoạch': 1.5,
                'đồng ruộng': 1.5, 'thị trường': 1.5, 'giá cả': 1.5, 'lạm phát': 1.5,
                'tăng trưởng': 1.5, 'phát triển': 1.5, 'sản xuất': 1.5, 'kinh doanh': 1.5,
                'doanh nghiệp': 1.5, 'công ty': 1.5, 'xí nghiệp': 1.5, 'nhà máy': 1.5,
                'cơ sở': 1.5, 'hộ kinh doanh': 1.5, 'tiểu thương': 1.5, 'buôn bán': 1.5,
                'mua sắm': 1.5, 'tiêu dùng': 1.5, 'chi tiêu': 1.5, 'thu nhập': 1.5,
                'lợi nhuận': 1.5, 'doanh thu': 1.5, 'chi phí': 1.5, 'thuế': 1.5,
                'phí': 1.5, 'lệ phí': 1.5, 'giá trị': 1.5, 'tài sản': 1.5,
                'bất động sản': 1.5, 'chứng khoán': 1.5, 'cổ phiếu': 1.5, 'trái phiếu': 1.5,
                'vàng': 1.5, 'ngoại tệ': 1.5, 'tiền tệ': 1.5, 'lãi suất': 1.5, 'tỷ giá': 1.5
            },
            'education': {
                'giáo dục': 2.0, 'học tập': 2.0, 'đào tạo': 2.0, 'giảng dạy': 2.0,
                'trường học': 2.0, 'đại học': 2.0, 'cao đẳng': 1.5, 'trung học': 1.5,
                'tiểu học': 1.5, 'mầm non': 1.5, 'giáo viên': 1.5, 'giảng viên': 1.5,
                'học sinh': 1.5, 'sinh viên': 1.5, 'chương trình': 1.5, 'môn học': 1.5,
                'kiểm tra': 1.5, 'thi cử': 1.5, 'bài giảng': 1.5, 'sách giáo khoa': 1.5,
                'tài liệu': 1.5, 'nghiên cứu': 1.5, 'luận văn': 1.5, 'luận án': 1.5,
                'khóa học': 1.5, 'lớp học': 1.5, 'phòng học': 1.5, 'thư viện': 1.5,
                'phòng thí nghiệm': 1.5, 'thực hành': 1.5, 'thực tập': 1.5,
                'tốt nghiệp': 1.5, 'bằng cấp': 1.5, 'chứng chỉ': 1.5, 'học bổng': 1.5,
                'học phí': 1.5, 'cơ sở vật chất': 1.5, 'phương pháp': 1.5,
                'công nghệ': 1.5, 'trực tuyến': 1.5, 'từ xa': 1.5, 'chuyên ngành': 1.5,
                'ngành học': 1.5, 'khoa': 1.5, 'viện': 1.5, 'trung tâm': 1.5,
                'học viện': 1.5, 'trường chuyên': 1.5, 'trường quốc tế': 1.5,
                'giáo trình': 1.5, 'giáo án': 1.5, 'bài tập': 1.5, 'đề thi': 1.5,
                'điểm số': 1.5, 'xếp hạng': 1.5, 'học lực': 1.5, 'hạnh kiểm': 1.5
            },
            'science': {
                'khoa học': 2.0, 'nghiên cứu': 2.0, 'phát minh': 2.0, 'sáng chế': 2.0,
                'thí nghiệm': 1.5, 'phòng thí nghiệm': 1.5, 'thiết bị': 1.5, 'dụng cụ': 1.5,
                'vật lý': 1.5, 'hóa học': 1.5, 'sinh học': 1.5, 'địa chất': 1.5,
                'thiên văn': 1.5, 'toán học': 1.5, 'tin học': 1.5, 'y học': 1.5,
                'dược học': 1.5, 'công nghệ sinh học': 1.5, 'nano': 1.5, 'robot': 1.5,
                'trí tuệ nhân tạo': 1.5, 'vũ trụ': 1.5, 'không gian': 1.5, 'hạt nhân': 1.5,
                'gen': 1.5, 'ADN': 1.5, 'tế bào': 1.5, 'vi sinh vật': 1.5,
                'vaccine': 1.5, 'thuốc': 1.5, 'bệnh': 1.5, 'điều trị': 1.5,
                'chẩn đoán': 1.5, 'phẫu thuật': 1.5, 'xét nghiệm': 1.5, 'hóa chất': 1.5,
                'vật liệu': 1.5, 'năng lượng': 1.5, 'môi trường': 1.5, 'khí hậu': 1.5,
                'sinh thái': 1.5, 'đa dạng sinh học': 1.5, 'tiến hóa': 1.5, 'di truyền': 1.5,
                'tế bào gốc': 1.5, 'protein': 1.5, 'enzyme': 1.5, 'hormone': 1.5
            },
            'technology': {
                'công nghệ': 2.0, 'kỹ thuật số': 2.0, 'trí tuệ nhân tạo': 2.0, 'AI': 2.0,
                'internet': 1.5, 'máy tính': 1.5, 'phần mềm': 1.5, 'ứng dụng': 1.5,
                'mạng': 1.5, 'website': 1.5, 'web': 1.5, 'online': 1.5,
                'trực tuyến': 1.5, 'số hóa': 1.5, 'tự động hóa': 1.5, 'robot': 1.5,
                'thiết bị': 1.5, 'điện tử': 1.5, 'viễn thông': 1.5, 'thông tin': 1.5,
                'dữ liệu': 1.5, 'bảo mật': 1.5, 'an ninh mạng': 1.5, 'cloud': 1.5,
                'đám mây': 1.5, 'big data': 1.5, 'dữ liệu lớn': 1.5, 'IoT': 1.5,
                'blockchain': 1.5, 'tiền điện tử': 1.5, 'cryptocurrency': 1.5, 'bitcoin': 1.5,
                'metaverse': 1.5, 'thực tế ảo': 1.5, 'VR': 1.5, 'AR': 1.5,
                '5G': 1.5, '6G': 1.5, 'wifi': 1.5, 'bluetooth': 1.5,
                'GPS': 1.5, 'định vị': 1.5, 'máy chủ': 1.5, 'server': 1.5,
                'database': 1.5, 'cơ sở dữ liệu': 1.5, 'phần cứng': 1.5, 'hardware': 1.5,
                'chip': 1.5, 'bán dẫn': 1.5, 'CPU': 1.5, 'GPU': 1.5,
                'RAM': 1.5, 'ROM': 1.5, 'SSD': 1.5, 'HDD': 1.5
            },
            'health': {
                'sức khỏe': 2.0, 'y tế': 2.0, 'bệnh viện': 2.0, 'bác sĩ': 2.0,
                'thuốc': 1.5, 'điều trị': 1.5, 'dịch bệnh': 1.5, 'khám bệnh': 1.5,
                'chữa bệnh': 1.5, 'phòng bệnh': 1.5, 'tiêm chủng': 1.5, 'vaccine': 1.5,
                'dược phẩm': 1.5, 'thực phẩm chức năng': 1.5, 'dinh dưỡng': 1.5, 'ăn uống': 1.5,
                'tập luyện': 1.5, 'yoga': 1.5, 'gym': 1.5, 'fitness': 1.5,
                'chạy bộ': 1.5, 'đi bộ': 1.5, 'bơi lội': 1.5, 'đạp xe': 1.5,
                'thể dục': 1.5, 'thể hình': 1.5, 'giảm cân': 1.5, 'tăng cân': 1.5,
                'phục hồi': 1.5, 'chấn thương': 1.5, 'phẫu thuật': 1.5, 'cấp cứu': 1.5,
                'cứu thương': 1.5, 'y học': 1.5, 'dược học': 1.5, 'đông y': 1.5,
                'tây y': 1.5, 'y học cổ truyền': 1.5, 'châm cứu': 1.5, 'bấm huyệt': 1.5,
                'xoa bóp': 1.5, 'vật lý trị liệu': 1.5, 'tâm lý': 1.5, 'tâm thần': 1.5,
                'stress': 1.5, 'trầm cảm': 1.5, 'lo âu': 1.5, 'mất ngủ': 1.5
            },
            'environment': {
                'môi trường': 2.0, 'ô nhiễm': 2.0, 'bảo vệ': 2.0, 'khí hậu': 2.0,
                'thiên nhiên': 1.5, 'rừng': 1.5, 'biển': 1.5, 'không khí': 1.5,
                'nước': 1.5, 'đất': 1.5, 'sinh thái': 1.5, 'hệ sinh thái': 1.5,
                'động vật': 1.5, 'thực vật': 1.5, 'tài nguyên': 1.5, 'năng lượng': 1.5,
                'năng lượng tái tạo': 1.5, 'năng lượng sạch': 1.5, 'điện mặt trời': 1.5,
                'điện gió': 1.5, 'thủy điện': 1.5, 'nhiệt điện': 1.5, 'điện hạt nhân': 1.5,
                'nhiên liệu': 1.5, 'xăng dầu': 1.5, 'khí đốt': 1.5, 'than đá': 1.5,
                'rác thải': 1.5, 'chất thải': 1.5, 'nước thải': 1.5, 'khí thải': 1.5,
                'bụi': 1.5, 'tiếng ồn': 1.5, 'ánh sáng': 1.5, 'nhiệt độ': 1.5,
                'độ ẩm': 1.5, 'mưa': 1.5, 'nắng': 1.5, 'gió': 1.5, 'bão': 1.5,
                'lũ lụt': 1.5, 'hạn hán': 1.5, 'thiên tai': 1.5, 'biến đổi khí hậu': 1.5,
                'nóng lên toàn cầu': 1.5, 'hiệu ứng nhà kính': 1.5, 'tầng ozone': 1.5,
                'tái chế': 1.5, 'tái sử dụng': 1.5, 'tiết kiệm': 1.5, 'bền vững': 1.5
            },
            'culture': {
                'văn hóa': 2.0, 'nghệ thuật': 2.0, 'âm nhạc': 2.0, 'hội họa': 2.0,
                'điêu khắc': 1.5, 'kiến trúc': 1.5, 'điện ảnh': 1.5, 'sân khấu': 1.5,
                'múa': 1.5, 'ca nhạc': 1.5, 'hát': 1.5, 'nhạc': 1.5, 'phim': 1.5,
                'kịch': 1.5, 'triển lãm': 1.5, 'bảo tàng': 1.5, 'di sản': 1.5,
                'di tích': 1.5, 'lễ hội': 1.5, 'phong tục': 1.5, 'tập quán': 1.5,
                'truyền thống': 1.5, 'dân tộc': 1.5, 'bản sắc': 1.5, 'văn học': 1.5,
                'thơ': 1.5, 'văn': 1.5, 'truyện': 1.5, 'tiểu thuyết': 1.5,
                'tác phẩm': 1.5, 'tác giả': 1.5, 'nhà văn': 1.5, 'nhà thơ': 1.5,
                'nghệ sĩ': 1.5, 'ca sĩ': 1.5, 'nhạc sĩ': 1.5, 'họa sĩ': 1.5,
                'diễn viên': 1.5, 'đạo diễn': 1.5, 'nhiếp ảnh': 1.5, 'thủ công': 1.5,
                'mỹ nghệ': 1.5, 'trang phục': 1.5, 'ẩm thực': 1.5, 'món ăn': 1.5,
                'đồ uống': 1.5, 'tín ngưỡng': 1.5, 'tôn giáo': 1.5, 'tâm linh': 1.5,
                'lễ nghi': 1.5, 'cúng bái': 1.5, 'thờ cúng': 1.5
            },
            'psychology': {
                'tâm lý': 2.0, 'tâm lý học': 2.0, 'hành vi': 2.0, 'cảm xúc': 2.0,
                'cảm giác': 1.5, 'tình cảm': 1.5, 'suy nghĩ': 1.5, 'nhận thức': 1.5,
                'trí nhớ': 1.5, 'học tập': 1.5, 'phát triển': 1.5, 'tính cách': 1.5,
                'cá tính': 1.5, 'động lực': 1.5, 'stress': 1.5, 'áp lực': 1.5,
                'lo âu': 1.5, 'trầm cảm': 1.5, 'rối loạn': 1.5, 'tâm thần': 1.5,
                'tư vấn': 1.5, 'trị liệu': 1.5, 'tham vấn': 1.5, 'tâm lý trị liệu': 1.5,
                'tâm thần học': 1.5, 'tâm bệnh học': 1.5, 'tâm lý xã hội': 1.5,
                'tâm lý giáo dục': 1.5, 'tâm lý phát triển': 1.5, 'tâm lý học tích cực': 1.5,
                'tâm lý học nhận thức': 1.5, 'tâm lý học hành vi': 1.5, 'tâm lý học xã hội': 1.5,
                'tâm lý học lâm sàng': 1.5, 'tâm lý học tư vấn': 1.5, 'tâm lý học trị liệu': 1.5,
                'tâm lý học ứng dụng': 1.5
            },
            'sociology': {
                'xã hội học': 2.0, 'xã hội': 2.0, 'cộng đồng': 2.0, 'nhóm xã hội': 2.0,
                'tổ chức xã hội': 1.5, 'cấu trúc xã hội': 1.5, 'quan hệ xã hội': 1.5,
                'tương tác xã hội': 1.5, 'hành vi xã hội': 1.5, 'vai trò xã hội': 1.5,
                'địa vị xã hội': 1.5, 'giai cấp': 1.5, 'tầng lớp': 1.5, 'phân tầng': 1.5,
                'di động xã hội': 1.5, 'biến đổi xã hội': 1.5, 'phát triển xã hội': 1.5,
                'vấn đề xã hội': 1.5, 'công bằng xã hội': 1.5, 'bất bình đẳng': 1.5,
                'phân biệt đối xử': 1.5, 'kỳ thị': 1.5, 'định kiến': 1.5, 'văn hóa xã hội': 1.5,
                'chuẩn mực xã hội': 1.5, 'giá trị xã hội': 1.5, 'niềm tin xã hội': 1.5,
                'thái độ xã hội': 1.5, 'xã hội dân sự': 1.5, 'phong trào xã hội': 1.5,
                'mạng lưới xã hội': 1.5, 'truyền thông xã hội': 1.5, 'trật tự xã hội': 1.5,
                'kiểm soát xã hội': 1.5, 'lệch lạc xã hội': 1.5, 'tội phạm học': 1.5,
                'đô thị học': 1.5, 'nông thôn học': 1.5, 'gia đình học': 1.5,
                'giới và phát triển': 1.5, 'dân số học': 1.5, 'nhân khẩu học': 1.5
            }
        }
        
        # Tăng trọng số cho các từ khóa quan trọng
        processed_words = []
        for word in words:
            processed_words.append(word)
            # Nếu từ là từ khóa quan trọng, thêm nó vào danh sách nhiều lần để tăng trọng số
            for category, keywords in important_keywords.items():
                if word in keywords.keys():  # Sửa lại cách kiểm tra từ khóa
                    # Thêm từ khóa 3 lần để tăng trọng số
                    processed_words.extend([word] * 3)
        
        text = ' '.join(processed_words)
        
        return text.strip()
    
    def _analyze_keywords(self, text: str) -> Dict[str, float]:
        """
        Phân tích từ khóa trong văn bản và trả về điểm số cho từng danh mục
        """
        try:
            text = text.lower()
            scores = {category: 0.0 for category in self.categories}
            
            # Định nghĩa từ khóa và trọng số cho từng danh mục
            keyword_weights = {
                'politics': {
                    'chính trị': 2.0, 'chính phủ': 2.0, 'quốc hội': 2.0, 'đảng': 2.0,
                    'chính sách': 1.5, 'đối ngoại': 1.5, 'nội chính': 1.5, 'pháp luật': 1.5,
                    'luật': 1.5, 'tòa án': 1.5, 'luật sư': 1.5, 'quy định': 1.5,
                    'vi phạm': 1.5, 'xử lý': 1.5, 'nghị quyết': 1.5, 'quyết định': 1.5,
                    'chỉ thị': 1.5, 'thông tư': 1.5, 'nghị định': 1.5, 'văn bản': 1.5,
                    'hành chính': 1.5, 'cải cách': 1.5, 'dân chủ': 1.5, 'nhân quyền': 1.5,
                    'tự do': 1.5, 'công bằng': 1.5, 'bình đẳng': 1.5, 'đoàn kết': 1.5,
                    'hòa bình': 1.5, 'an ninh': 1.5, 'quốc phòng': 1.5, 'ngoại giao': 1.5,
                    'hợp tác': 1.5, 'phát triển': 1.5, 'xây dựng': 1.5, 'đổi mới': 1.5,
                    'cải thiện': 1.5, 'nâng cao': 1.5, 'thúc đẩy': 1.5, 'tăng cường': 1.5,
                    'mở rộng': 1.5, 'củng cố': 1.5, 'duy trì': 1.5, 'bảo vệ': 1.5,
                    'đảm bảo': 1.5, 'thực hiện': 1.5, 'triển khai': 1.5, 'quản lý': 1.5,
                    'điều hành': 1.5, 'lãnh đạo': 1.5, 'chỉ đạo': 1.5, 'kiểm tra': 1.5,
                    'giám sát': 1.5, 'thanh tra': 1.5, 'kiểm toán': 1.5
                },
                'economics': {
                    'kinh tế': 2.0, 'tài chính': 2.0, 'ngân hàng': 2.0, 'thương mại': 2.0,
                    'xuất khẩu': 1.5, 'nhập khẩu': 1.5, 'đầu tư': 1.5, 'du lịch': 1.5,
                    'khách sạn': 1.5, 'tour': 1.5, 'du khách': 1.5, 'thăm quan': 1.5,
                    'điểm đến': 1.5, 'phượt': 1.5, 'nông nghiệp': 1.5, 'nông dân': 1.5,
                    'trồng trọt': 1.5, 'chăn nuôi': 1.5, 'nông sản': 1.5, 'thu hoạch': 1.5,
                    'đồng ruộng': 1.5, 'thị trường': 1.5, 'giá cả': 1.5, 'lạm phát': 1.5,
                    'tăng trưởng': 1.5, 'phát triển': 1.5, 'sản xuất': 1.5, 'kinh doanh': 1.5,
                    'doanh nghiệp': 1.5, 'công ty': 1.5, 'xí nghiệp': 1.5, 'nhà máy': 1.5,
                    'cơ sở': 1.5, 'hộ kinh doanh': 1.5, 'tiểu thương': 1.5, 'buôn bán': 1.5,
                    'mua sắm': 1.5, 'tiêu dùng': 1.5, 'chi tiêu': 1.5, 'thu nhập': 1.5,
                    'lợi nhuận': 1.5, 'doanh thu': 1.5, 'chi phí': 1.5, 'thuế': 1.5,
                    'phí': 1.5, 'lệ phí': 1.5, 'giá trị': 1.5, 'tài sản': 1.5,
                    'bất động sản': 1.5, 'chứng khoán': 1.5, 'cổ phiếu': 1.5, 'trái phiếu': 1.5,
                    'vàng': 1.5, 'ngoại tệ': 1.5, 'tiền tệ': 1.5, 'lãi suất': 1.5, 'tỷ giá': 1.5
                },
                'education': {
                    'giáo dục': 2.0, 'học tập': 2.0, 'đào tạo': 2.0, 'giảng dạy': 2.0,
                    'trường học': 2.0, 'đại học': 2.0, 'cao đẳng': 1.5, 'trung học': 1.5,
                    'tiểu học': 1.5, 'mầm non': 1.5, 'giáo viên': 1.5, 'giảng viên': 1.5,
                    'học sinh': 1.5, 'sinh viên': 1.5, 'chương trình': 1.5, 'môn học': 1.5,
                    'kiểm tra': 1.5, 'thi cử': 1.5, 'bài giảng': 1.5, 'sách giáo khoa': 1.5,
                    'tài liệu': 1.5, 'nghiên cứu': 1.5, 'luận văn': 1.5, 'luận án': 1.5,
                    'khóa học': 1.5, 'lớp học': 1.5, 'phòng học': 1.5, 'thư viện': 1.5,
                    'phòng thí nghiệm': 1.5, 'thực hành': 1.5, 'thực tập': 1.5,
                    'tốt nghiệp': 1.5, 'bằng cấp': 1.5, 'chứng chỉ': 1.5, 'học bổng': 1.5,
                    'học phí': 1.5, 'cơ sở vật chất': 1.5, 'phương pháp': 1.5,
                    'công nghệ': 1.5, 'trực tuyến': 1.5, 'từ xa': 1.5, 'chuyên ngành': 1.5,
                    'ngành học': 1.5, 'khoa': 1.5, 'viện': 1.5, 'trung tâm': 1.5,
                    'học viện': 1.5, 'trường chuyên': 1.5, 'trường quốc tế': 1.5,
                    'giáo trình': 1.5, 'giáo án': 1.5, 'bài tập': 1.5, 'đề thi': 1.5,
                    'điểm số': 1.5, 'xếp hạng': 1.5, 'học lực': 1.5, 'hạnh kiểm': 1.5
                },
                'science': {
                    'khoa học': 2.0, 'nghiên cứu': 2.0, 'phát minh': 2.0, 'sáng chế': 2.0,
                    'thí nghiệm': 1.5, 'phòng thí nghiệm': 1.5, 'thiết bị': 1.5, 'dụng cụ': 1.5,
                    'vật lý': 1.5, 'hóa học': 1.5, 'sinh học': 1.5, 'địa chất': 1.5,
                    'thiên văn': 1.5, 'toán học': 1.5, 'tin học': 1.5, 'y học': 1.5,
                    'dược học': 1.5, 'công nghệ sinh học': 1.5, 'nano': 1.5, 'robot': 1.5,
                    'trí tuệ nhân tạo': 1.5, 'vũ trụ': 1.5, 'không gian': 1.5, 'hạt nhân': 1.5,
                    'gen': 1.5, 'ADN': 1.5, 'tế bào': 1.5, 'vi sinh vật': 1.5,
                    'vaccine': 1.5, 'thuốc': 1.5, 'bệnh': 1.5, 'điều trị': 1.5,
                    'chẩn đoán': 1.5, 'phẫu thuật': 1.5, 'xét nghiệm': 1.5, 'hóa chất': 1.5,
                    'vật liệu': 1.5, 'năng lượng': 1.5, 'môi trường': 1.5, 'khí hậu': 1.5,
                    'sinh thái': 1.5, 'đa dạng sinh học': 1.5, 'tiến hóa': 1.5, 'di truyền': 1.5,
                    'tế bào gốc': 1.5, 'protein': 1.5, 'enzyme': 1.5, 'hormone': 1.5
                },
                'technology': {
                    'công nghệ': 2.0, 'kỹ thuật số': 2.0, 'trí tuệ nhân tạo': 2.0, 'AI': 2.0,
                    'internet': 1.5, 'máy tính': 1.5, 'phần mềm': 1.5, 'ứng dụng': 1.5,
                    'mạng': 1.5, 'website': 1.5, 'web': 1.5, 'online': 1.5,
                    'trực tuyến': 1.5, 'số hóa': 1.5, 'tự động hóa': 1.5, 'robot': 1.5,
                    'thiết bị': 1.5, 'điện tử': 1.5, 'viễn thông': 1.5, 'thông tin': 1.5,
                    'dữ liệu': 1.5, 'bảo mật': 1.5, 'an ninh mạng': 1.5, 'cloud': 1.5,
                    'đám mây': 1.5, 'big data': 1.5, 'dữ liệu lớn': 1.5, 'IoT': 1.5,
                    'blockchain': 1.5, 'tiền điện tử': 1.5, 'cryptocurrency': 1.5, 'bitcoin': 1.5,
                    'metaverse': 1.5, 'thực tế ảo': 1.5, 'VR': 1.5, 'AR': 1.5,
                    '5G': 1.5, '6G': 1.5, 'wifi': 1.5, 'bluetooth': 1.5,
                    'GPS': 1.5, 'định vị': 1.5, 'máy chủ': 1.5, 'server': 1.5,
                    'database': 1.5, 'cơ sở dữ liệu': 1.5, 'phần cứng': 1.5, 'hardware': 1.5,
                    'chip': 1.5, 'bán dẫn': 1.5, 'CPU': 1.5, 'GPU': 1.5,
                    'RAM': 1.5, 'ROM': 1.5, 'SSD': 1.5, 'HDD': 1.5
                },
                'health': {
                    'sức khỏe': 2.0, 'y tế': 2.0, 'bệnh viện': 2.0, 'bác sĩ': 2.0,
                    'thuốc': 1.5, 'điều trị': 1.5, 'dịch bệnh': 1.5, 'khám bệnh': 1.5,
                    'chữa bệnh': 1.5, 'phòng bệnh': 1.5, 'tiêm chủng': 1.5, 'vaccine': 1.5,
                    'dược phẩm': 1.5, 'thực phẩm chức năng': 1.5, 'dinh dưỡng': 1.5, 'ăn uống': 1.5,
                    'tập luyện': 1.5, 'yoga': 1.5, 'gym': 1.5, 'fitness': 1.5,
                    'chạy bộ': 1.5, 'đi bộ': 1.5, 'bơi lội': 1.5, 'đạp xe': 1.5,
                    'thể dục': 1.5, 'thể hình': 1.5, 'giảm cân': 1.5, 'tăng cân': 1.5,
                    'phục hồi': 1.5, 'chấn thương': 1.5, 'phẫu thuật': 1.5, 'cấp cứu': 1.5,
                    'cứu thương': 1.5, 'y học': 1.5, 'dược học': 1.5, 'đông y': 1.5,
                    'tây y': 1.5, 'y học cổ truyền': 1.5, 'châm cứu': 1.5, 'bấm huyệt': 1.5,
                    'xoa bóp': 1.5, 'vật lý trị liệu': 1.5, 'tâm lý': 1.5, 'tâm thần': 1.5,
                    'stress': 1.5, 'trầm cảm': 1.5, 'lo âu': 1.5, 'mất ngủ': 1.5
                },
                'environment': {
                    'môi trường': 2.0, 'ô nhiễm': 2.0, 'bảo vệ': 2.0, 'khí hậu': 2.0,
                    'thiên nhiên': 1.5, 'rừng': 1.5, 'biển': 1.5, 'không khí': 1.5,
                    'nước': 1.5, 'đất': 1.5, 'sinh thái': 1.5, 'hệ sinh thái': 1.5,
                    'động vật': 1.5, 'thực vật': 1.5, 'tài nguyên': 1.5, 'năng lượng': 1.5,
                    'năng lượng tái tạo': 1.5, 'năng lượng sạch': 1.5, 'điện mặt trời': 1.5,
                    'điện gió': 1.5, 'thủy điện': 1.5, 'nhiệt điện': 1.5, 'điện hạt nhân': 1.5,
                    'nhiên liệu': 1.5, 'xăng dầu': 1.5, 'khí đốt': 1.5, 'than đá': 1.5,
                    'rác thải': 1.5, 'chất thải': 1.5, 'nước thải': 1.5, 'khí thải': 1.5,
                    'bụi': 1.5, 'tiếng ồn': 1.5, 'ánh sáng': 1.5, 'nhiệt độ': 1.5,
                    'độ ẩm': 1.5, 'mưa': 1.5, 'nắng': 1.5, 'gió': 1.5, 'bão': 1.5,
                    'lũ lụt': 1.5, 'hạn hán': 1.5, 'thiên tai': 1.5, 'biến đổi khí hậu': 1.5,
                    'nóng lên toàn cầu': 1.5, 'hiệu ứng nhà kính': 1.5, 'tầng ozone': 1.5,
                    'tái chế': 1.5, 'tái sử dụng': 1.5, 'tiết kiệm': 1.5, 'bền vững': 1.5
                },
                'culture': {
                    'văn hóa': 2.0, 'nghệ thuật': 2.0, 'âm nhạc': 2.0, 'hội họa': 2.0,
                    'điêu khắc': 1.5, 'kiến trúc': 1.5, 'điện ảnh': 1.5, 'sân khấu': 1.5,
                    'múa': 1.5, 'ca nhạc': 1.5, 'hát': 1.5, 'nhạc': 1.5, 'phim': 1.5,
                    'kịch': 1.5, 'triển lãm': 1.5, 'bảo tàng': 1.5, 'di sản': 1.5,
                    'di tích': 1.5, 'lễ hội': 1.5, 'phong tục': 1.5, 'tập quán': 1.5,
                    'truyền thống': 1.5, 'dân tộc': 1.5, 'bản sắc': 1.5, 'văn học': 1.5,
                    'thơ': 1.5, 'văn': 1.5, 'truyện': 1.5, 'tiểu thuyết': 1.5,
                    'tác phẩm': 1.5, 'tác giả': 1.5, 'nhà văn': 1.5, 'nhà thơ': 1.5,
                    'nghệ sĩ': 1.5, 'ca sĩ': 1.5, 'nhạc sĩ': 1.5, 'họa sĩ': 1.5,
                    'diễn viên': 1.5, 'đạo diễn': 1.5, 'nhiếp ảnh': 1.5, 'thủ công': 1.5,
                    'mỹ nghệ': 1.5, 'trang phục': 1.5, 'ẩm thực': 1.5, 'món ăn': 1.5,
                    'đồ uống': 1.5, 'tín ngưỡng': 1.5, 'tôn giáo': 1.5, 'tâm linh': 1.5,
                    'lễ nghi': 1.5, 'cúng bái': 1.5, 'thờ cúng': 1.5
                },
                'psychology': {
                    'tâm lý': 2.0, 'tâm lý học': 2.0, 'hành vi': 2.0, 'cảm xúc': 2.0,
                    'cảm giác': 1.5, 'tình cảm': 1.5, 'suy nghĩ': 1.5, 'nhận thức': 1.5,
                    'trí nhớ': 1.5, 'học tập': 1.5, 'phát triển': 1.5, 'tính cách': 1.5,
                    'cá tính': 1.5, 'động lực': 1.5, 'stress': 1.5, 'áp lực': 1.5,
                    'lo âu': 1.5, 'trầm cảm': 1.5, 'rối loạn': 1.5, 'tâm thần': 1.5,
                    'tư vấn': 1.5, 'trị liệu': 1.5, 'tham vấn': 1.5, 'tâm lý trị liệu': 1.5,
                    'tâm thần học': 1.5, 'tâm bệnh học': 1.5, 'tâm lý xã hội': 1.5,
                    'tâm lý giáo dục': 1.5, 'tâm lý phát triển': 1.5, 'tâm lý học tích cực': 1.5,
                    'tâm lý học nhận thức': 1.5, 'tâm lý học hành vi': 1.5, 'tâm lý học xã hội': 1.5,
                    'tâm lý học lâm sàng': 1.5, 'tâm lý học tư vấn': 1.5, 'tâm lý học trị liệu': 1.5,
                    'tâm lý học ứng dụng': 1.5
                },
                'sociology': {
                    'xã hội học': 2.0, 'xã hội': 2.0, 'cộng đồng': 2.0, 'nhóm xã hội': 2.0,
                    'tổ chức xã hội': 1.5, 'cấu trúc xã hội': 1.5, 'quan hệ xã hội': 1.5,
                    'tương tác xã hội': 1.5, 'hành vi xã hội': 1.5, 'vai trò xã hội': 1.5,
                    'địa vị xã hội': 1.5, 'giai cấp': 1.5, 'tầng lớp': 1.5, 'phân tầng': 1.5,
                    'di động xã hội': 1.5, 'biến đổi xã hội': 1.5, 'phát triển xã hội': 1.5,
                    'vấn đề xã hội': 1.5, 'công bằng xã hội': 1.5, 'bất bình đẳng': 1.5,
                    'phân biệt đối xử': 1.5, 'kỳ thị': 1.5, 'định kiến': 1.5, 'văn hóa xã hội': 1.5,
                    'chuẩn mực xã hội': 1.5, 'giá trị xã hội': 1.5, 'niềm tin xã hội': 1.5,
                    'thái độ xã hội': 1.5, 'xã hội dân sự': 1.5, 'phong trào xã hội': 1.5,
                    'mạng lưới xã hội': 1.5, 'truyền thông xã hội': 1.5, 'trật tự xã hội': 1.5,
                    'kiểm soát xã hội': 1.5, 'lệch lạc xã hội': 1.5, 'tội phạm học': 1.5,
                    'đô thị học': 1.5, 'nông thôn học': 1.5, 'gia đình học': 1.5,
                    'giới và phát triển': 1.5, 'dân số học': 1.5, 'nhân khẩu học': 1.5
                }
            }
            
            # Tính điểm cho từng danh mục dựa trên từ khóa
            for category, weights in keyword_weights.items():
                for keyword, weight in weights.items():
                    if keyword in text:
                        scores[category] += weight
            
            # Chuẩn hóa điểm số
            total_score = sum(scores.values())
            if total_score > 0:
                for category in scores:
                    scores[category] = scores[category] / total_score
            
            return scores
            
        except Exception as e:
            # Trả về điểm số mặc định nếu có lỗi
            return {category: 0.0 for category in self.categories}
    
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
        
        print("\nTraining LinearSVC model...")
        train_labels = np.array(train_labels)
        val_labels = np.array(val_labels)
        
        # Huấn luyện mô hình
        self.model.fit(X_train, train_labels)
        
        # Tạo mô hình có khả năng dự đoán xác suất
        self.prob_model = CalibratedClassifierCV(self.model, cv=5, method='sigmoid')
        self.prob_model.fit(X_train, train_labels)
        
        print("\nModel Evaluation:")
        val_preds = self.model.predict(X_val)
        print(classification_report(val_labels, val_preds, target_names=self.categories))
        
        # Đánh dấu mô hình đã được huấn luyện
        self.is_trained = True
        
        # Lưu mô hình nếu được yêu cầu
        if save_model:
            self.save_model()
            
        end_time = time.time()
        print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    
    def predict(self, text: str) -> Dict:
        if not self._is_vietnamese(text):
            return {"error": "Text is not in Vietnamese", "original_text": text}
        
        if not hasattr(self, 'prob_model') or self.prob_model is None:
            return {"error": "Model has not been trained yet", "original_text": text}
        
        processed_text = self._preprocess_text(text)
        if not processed_text:
            return {"error": "Text is empty after preprocessing", "original_text": text}
        
        try:
            X = self.vectorizer.transform([processed_text])
            
            # Lấy dự đoán và xác suất từ mô hình
            prediction = int(self.model.predict(X)[0])
            probabilities = self.prob_model.predict_proba(X)[0]
            
            # Phân tích từ khóa để điều chỉnh xác suất
            keyword_scores = self._analyze_keywords(text)
            
            # Kết hợp xác suất từ mô hình và điểm số từ phân tích từ khóa
            adjusted_probabilities = probabilities.copy()
            for i, category in enumerate(self.categories):
                # Trọng số cho mô hình và phân tích từ khóa
                model_weight = 0.7
                keyword_weight = 0.3
                
                # Lấy điểm số từ khóa cho danh mục hiện tại
                keyword_score = keyword_scores.get(category, 0.0)
                
                # Kết hợp xác suất
                adjusted_probabilities[i] = (model_weight * probabilities[i] + 
                                            keyword_weight * keyword_score)
            
            # Chuẩn hóa xác suất
            adjusted_probabilities = adjusted_probabilities / adjusted_probabilities.sum()
            
            # Kiểm tra xác suất tối thiểu
            max_prob = np.max(adjusted_probabilities)
            if max_prob < 0.3:  # Ngưỡng xác suất tối thiểu
                return {
                    "error": "Low confidence in classification",
                    "original_text": text,
                    "max_probability": float(max_prob * 100)
                }
            
            # Sắp xếp các danh mục theo xác suất giảm dần
            sorted_indices = np.argsort(adjusted_probabilities)[::-1]
            sorted_categories = [self.categories[i] for i in sorted_indices]
            sorted_probabilities = [float(adjusted_probabilities[i]) * 100 for i in sorted_indices]
            
            # Cập nhật dự đoán dựa trên xác suất đã điều chỉnh
            prediction = sorted_indices[0]
            
            result = {
                "category": self.categories[prediction],
                "probabilities": {
                    category: prob
                    for category, prob in zip(sorted_categories, sorted_probabilities)
                },
                "top_3_categories": [
                    {"category": sorted_categories[i], "probability": sorted_probabilities[i]}
                    for i in range(min(3, len(sorted_categories)))
                ],
                "original_text": text
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Error during prediction: {str(e)}", "original_text": text}
    
    def save_model(self, model_dir: str = 'models') -> None:
        """Lưu mô hình đã huấn luyện vào thư mục models"""
        if not self.is_trained:
            print("Model has not been trained yet. Cannot save.")
            return
            
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
        prob_model_path = os.path.join(model_dir, 'prob_model.pkl')
        categories_path = os.path.join(model_dir, 'categories.pkl')
        
        if not all(os.path.exists(path) for path in [vectorizer_path, model_path, prob_model_path, categories_path]):
            print("Model files not found. Please train the model first.")
            return False
            
        # Tải vectorizer
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
            
        # Tải mô hình
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        # Tải mô hình xác suất
        with open(prob_model_path, 'rb') as f:
            self.prob_model = pickle.load(f)
            
        # Tải danh sách categories
        with open(categories_path, 'rb') as f:
            self.categories = pickle.load(f)
            
        # Đánh dấu mô hình đã được huấn luyện
        self.is_trained = True
        print(f"Model loaded from {model_dir}")
        return True

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
    print("Using scikit-learn LinearSVC Implementation")
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