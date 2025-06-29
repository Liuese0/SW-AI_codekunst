import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import google.generativeai as genai
import pandas as pd
import re
import threading
from datetime import datetime
import os

class ChatEmotionAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("채팅 메시지 감정 분석기 - 진심/빈말/비꼬기 구분")
        self.root.geometry("800x700")
        self.root.configure(bg='#f0f0f0')
        
        # Gemini API 설정
        self.api_key = "AIzaSyDPUXHrBZ3P4luxl9aTvrqsTPRZtDNAo18"
        genai.configure(api_key=self.api_key)
        # 최신 Gemini 모델 사용
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except:
            try:
                self.model = genai.GenerativeModel('gemini-1.5-pro')
            except:
                self.model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
        
        # 대화 기록 저장
        self.conversation_history = []
        
        # 풍자 데이터셋 저장
        self.dataset = None
        self.dataset_info = ""
        
        # API 연결 테스트
        self.test_api_connection()
        
        self.setup_ui()
        
    def test_api_connection(self):
        """API 연결 테스트"""
        try:
            test_response = self.model.generate_content("안녕하세요")
            print("✅ Gemini API 연결 성공!")
        except Exception as e:
            print(f"⚠️ API 연결 실패: {e}")
            messagebox.showwarning("API 경고", f"Gemini API 연결에 문제가 있을 수 있습니다:\n{str(e)[:100]}...")
        
    def setup_ui(self):
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 제목
        title_label = ttk.Label(main_frame, text="채팅 메시지 감정 분석기", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # 데이터셋 로드 섹션
        dataset_frame = ttk.LabelFrame(main_frame, text="풍자 데이터셋 로드", padding="10")
        dataset_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 데이터셋 상태 표시
        self.dataset_status = tk.StringVar()
        self.dataset_status.set("데이터셋이 로드되지 않음")
        dataset_status_label = ttk.Label(dataset_frame, textvariable=self.dataset_status)
        dataset_status_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        # 데이터셋 로드 버튼
        load_dataset_button = ttk.Button(dataset_frame, text="CSV 데이터셋 로드", 
                                        command=self.load_dataset)
        load_dataset_button.grid(row=0, column=1)
        
        # 데이터셋 정보 표시
        self.dataset_info_text = tk.Text(dataset_frame, height=3, width=70)
        self.dataset_info_text.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        self.dataset_info_text.config(state='disabled')
        
        # 입력 섹션
        input_frame = ttk.LabelFrame(main_frame, text="메시지 입력", padding="10")
        input_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 메시지 입력창
        ttk.Label(input_frame, text="분석할 메시지:").grid(row=0, column=0, sticky=tk.W)
        self.message_entry = scrolledtext.ScrolledText(input_frame, height=4, width=70)
        self.message_entry.grid(row=1, column=0, columnspan=2, pady=(5, 10))
        
        # 이전 대화 맥락 입력
        ttk.Label(input_frame, text="이전 대화 맥락 (선택사항):").grid(row=2, column=0, sticky=tk.W)
        self.context_entry = scrolledtext.ScrolledText(input_frame, height=3, width=70)
        self.context_entry.grid(row=3, column=0, columnspan=2, pady=(5, 10))
        
        # 분석 버튼
        self.analyze_button = ttk.Button(input_frame, text="감정 분석하기", 
                                        command=self.analyze_message)
        self.analyze_button.grid(row=4, column=0, pady=(0, 5))
        
        # 기록 지우기 버튼
        self.clear_button = ttk.Button(input_frame, text="기록 지우기", 
                                      command=self.clear_history)
        self.clear_button.grid(row=4, column=1, pady=(0, 5), padx=(10, 0))
        
        # 결과 섹션
        result_frame = ttk.LabelFrame(main_frame, text="분석 결과", padding="10")
        result_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # 결과 표시창
        self.result_text = scrolledtext.ScrolledText(result_frame, height=15, width=80)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 진행 상황 표시
        self.progress_var = tk.StringVar()
        self.progress_var.set("분석할 메시지를 입력하세요.")
        self.progress_label = ttk.Label(main_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=4, column=0, columnspan=2, pady=(0, 5))
        
        # 그리드 가중치 설정
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
    def load_dataset(self):
        """풍자 데이터셋 CSV 파일 로드"""
        file_path = filedialog.askopenfilename(
            title="풍자 데이터셋 CSV 파일 선택",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # CSV 파일 읽기
                self.dataset = pd.read_csv(file_path, encoding='utf-8')
                
                # 데이터셋 정보 생성
                dataset_info = f"✅ 데이터셋 로드 성공!\n"
                dataset_info += f"📁 파일: {os.path.basename(file_path)}\n"
                dataset_info += f"📊 총 {len(self.dataset)}개 데이터\n"
                dataset_info += f"📋 컬럼: {', '.join(self.dataset.columns.tolist())}\n"
                
                # 각 카테고리별 개수 확인
                if 'label' in self.dataset.columns:
                    label_counts = self.dataset['label'].value_counts()
                    dataset_info += f"🏷️ 라벨 분포: {dict(label_counts)}\n"
                elif 'category' in self.dataset.columns:
                    category_counts = self.dataset['category'].value_counts()
                    dataset_info += f"🏷️ 카테고리 분포: {dict(category_counts)}\n"
                
                # 샘플 데이터 미리보기
                sample_data = self.dataset.head(3).to_string(index=False, max_cols=3)
                dataset_info += f"\n📝 샘플 데이터:\n{sample_data[:200]}..."
                
                self.dataset_info = dataset_info
                
                # UI 업데이트
                self.dataset_status.set(f"✅ 데이터셋 로드됨 ({len(self.dataset)}개)")
                self.dataset_info_text.config(state='normal')
                self.dataset_info_text.delete(1.0, tk.END)
                self.dataset_info_text.insert(1.0, dataset_info)
                self.dataset_info_text.config(state='disabled')
                
                messagebox.showinfo("성공", f"데이터셋이 성공적으로 로드되었습니다!\n총 {len(self.dataset)}개의 데이터")
                
            except Exception as e:
                messagebox.showerror("오류", f"데이터셋 로드 중 오류가 발생했습니다:\n{str(e)}")
                self.dataset_status.set("❌ 데이터셋 로드 실패")
    
    def get_similar_examples(self, message, num_examples=3):
        """입력 메시지와 유사한 데이터셋 예시 찾기"""
        if self.dataset is None:
            return []
        
        # 텍스트 컬럼 찾기
        text_columns = []
        for col in self.dataset.columns:
            if any(keyword in col.lower() for keyword in ['text', 'message', 'content', '메시지', '내용', 'sentence']):
                text_columns.append(col)
        
        if not text_columns:
            return []
        
        text_col = text_columns[0]  # 첫 번째 텍스트 컬럼 사용
        
        # 라벨 컬럼 찾기
        label_col = None
        for col in self.dataset.columns:
            if any(keyword in col.lower() for keyword in ['label', 'category', 'class', '라벨', '분류']):
                label_col = col
                break
        
        # 간단한 유사도 계산 (공통 단어 개수 기반)
        message_words = set(message.lower().split())
        similarities = []
        
        for idx, row in self.dataset.iterrows():
            text = str(row[text_col]).lower()
            text_words = set(text.split())
            
            # 공통 단어 개수로 유사도 계산
            common_words = len(message_words.intersection(text_words))
            similarity = common_words / (len(message_words) + len(text_words) - common_words + 1)
            
            similarities.append({
                'index': idx,
                'text': row[text_col],
                'label': row[label_col] if label_col else 'Unknown',
                'similarity': similarity
            })
        
        # 유사도 순으로 정렬하여 상위 예시 반환
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:num_examples]
    
    def preprocess_text(self, text):
        """텍스트 전처리"""
        # 특수문자 및 이모티콘 처리
        text = re.sub(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ.,!?~]', ' ', text)
        
        # 반복 문자 처리 (ㅋㅋㅋ, ㅠㅠ 등)
        text = re.sub(r'([ㅋㅎㅠㅜ])\1+', r'\1\1', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, text):
        """텍스트 특성 추출"""
        features = {}
        
        # 언어적 특성
        features['길이'] = len(text)
        features['감탄사_개수'] = len(re.findall(r'[!?~]', text))
        features['반복_표현'] = len(re.findall(r'([ㅋㅎㅠㅜ])\1+', text))
        features['줄임말_사용'] = len(re.findall(r'[ㄱ-ㅎ]', text))
        
        # 존댓말/반말 판별
        honorific_patterns = ['습니다', '해요', '세요', '께서', '님']
        casual_patterns = ['야', '어', '지', '다', '냐']
        
        honorific_count = sum(1 for pattern in honorific_patterns if pattern in text)
        casual_count = sum(1 for pattern in casual_patterns if text.endswith(pattern))
        
        if honorific_count > casual_count:
            features['문체'] = '존댓말'
        else:
            features['문체'] = '반말'
            
        return features
    
    def create_prompt(self, message, context="", features=None):
        """분석을 위한 프롬프트 생성 (데이터셋 참고)"""
        
        # 유사한 예시 찾기
        similar_examples = self.get_similar_examples(message) if self.dataset is not None else []
        
        prompt = f"""
다음 채팅 메시지를 분석하여 진심, 빈말, 비꼬기 중 어느 것인지 판단해주세요.

분석할 메시지: "{message}"

{f"이전 대화 맥락: {context}" if context else ""}

{f"추출된 특성: {features}" if features else ""}

"""
        
        # 데이터셋 참고 예시 추가
        if similar_examples:
            prompt += f"""
📚 참고할 유사한 예시들 (데이터셋에서):
"""
            for i, example in enumerate(similar_examples, 1):
                prompt += f"{i}. 텍스트: \"{example['text']}\"\n   분류: {example['label']}\n"
            
            prompt += f"""
위 예시들을 참고하여 분석해주세요.

"""
        
        # 데이터셋 정보 추가
        if self.dataset is not None:
            prompt += f"""
🎯 사용 가능한 데이터셋 정보:
{self.dataset_info}

"""
        
        prompt += f"""
다음 기준으로 분석해주세요:

1. **진심 (Sincere)**:
   - 말하는 사람의 진짜 감정이나 의도가 담긴 메시지
   - 직설적이고 솔직한 표현
   - 맥락과 일치하는 감정 표현

2. **빈말 (Empty words)**:
   - 사회적 예의나 관례로 하는 말
   - 진정성이 부족한 표현
   - 형식적이거나 의무적인 말

3. **비꼬기 (Sarcasm)**:
   - 반어적 표현
   - 겉으로는 긍정적이지만 실제로는 부정적 의미
   - 맥락과 반대되는 감정 표현

다음 형식으로 답변해주세요:
**판단 결과**: [진심/빈말/비꼬기]
**확신도**: [1-10점]
**근거**: [구체적인 분석 근거]
**주요 단서**: [판단에 중요한 언어적 특성들]
**데이터셋 참고**: [유사 예시와의 비교 분석]
"""
        return prompt
    
    def analyze_with_gemini(self, message, context=""):
        """Gemini API를 사용하여 감정 분석"""
        try:
            # 텍스트 전처리
            processed_message = self.preprocess_text(message)
            
            # 특성 추출
            features = self.extract_features(processed_message)
            
            # 프롬프트 생성
            prompt = self.create_prompt(processed_message, context, features)
            
            # Gemini API 호출 - 안전 설정 추가
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            return response.text, features
            
        except Exception as e:
            # API 키 유효성 검사
            if "API_KEY" in str(e) or "authentication" in str(e).lower():
                return "API 키가 유효하지 않습니다. 설정을 확인해주세요.", None
            elif "quota" in str(e).lower() or "limit" in str(e).lower():
                return "API 사용 한도를 초과했습니다. 잠시 후 다시 시도해주세요.", None
            else:
                return f"분석 중 오류가 발생했습니다: {str(e)}", None
    
    def analyze_message(self):
        """메시지 분석 실행"""
        message = self.message_entry.get(1.0, tk.END).strip()
        context = self.context_entry.get(1.0, tk.END).strip()
        
        if not message:
            messagebox.showwarning("경고", "분석할 메시지를 입력해주세요.")
            return
        
        # 버튼 비활성화
        self.analyze_button.config(state='disabled')
        self.progress_var.set("분석 중...")
        
        # 별도 스레드에서 분석 실행
        thread = threading.Thread(target=self.run_analysis, args=(message, context))
        thread.daemon = True
        thread.start()
    
    def run_analysis(self, message, context):
        """분석 실행 (별도 스레드)"""
        try:
            # Gemini API 분석
            analysis_result, features = self.analyze_with_gemini(message, context)
            
            # UI 업데이트 (메인 스레드에서)
            self.root.after(0, self.update_results, message, context, analysis_result, features)
            
        except Exception as e:
            self.root.after(0, self.handle_error, str(e))
    
    def update_results(self, message, context, analysis_result, features):
        """결과 업데이트"""
        # 현재 시간
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 결과 텍스트 구성
        result_text = f"""
{'='*80}
📅 분석 일시: {timestamp}
💬 분석 메시지: "{message}"
{f"📋 대화 맥락: {context}" if context else ""}
{f"📊 사용된 데이터셋: {len(self.dataset)}개 데이터 참고" if self.dataset is not None else "📊 데이터셋: 사용되지 않음"}

🔍 추출된 특성:
{self.format_features(features) if features else "특성 추출 실패"}

🤖 AI 분석 결과:
{analysis_result}

{'='*80}

"""
        
        # 결과창에 추가
        self.result_text.insert(tk.END, result_text)
        self.result_text.see(tk.END)
        
        # 대화 기록에 저장
        self.conversation_history.append({
            'timestamp': timestamp,
            'message': message,
            'context': context,
            'result': analysis_result,
            'features': features
        })
        
        # UI 상태 복원
        self.analyze_button.config(state='normal')
        self.progress_var.set(f"분석 완료! (총 {len(self.conversation_history)}개 분석)")
        
        # 입력창 초기화
        self.message_entry.delete(1.0, tk.END)
    
    def format_features(self, features):
        """특성을 보기 좋게 포맷"""
        if not features:
            return "특성 정보 없음"
        
        formatted = []
        for key, value in features.items():
            formatted.append(f"  • {key}: {value}")
        
        return "\n".join(formatted)
    
    def handle_error(self, error_message):
        """오류 처리"""
        messagebox.showerror("오류", f"분석 중 오류가 발생했습니다:\n{error_message}")
        self.analyze_button.config(state='normal')
        self.progress_var.set("오류 발생")
    
    def clear_history(self):
        """기록 지우기"""
        if messagebox.askyesno("확인", "모든 분석 기록을 지우시겠습니까?"):
            self.result_text.delete(1.0, tk.END)
            self.conversation_history.clear()
            self.progress_var.set("기록이 지워졌습니다.")

def main():
    root = tk.Tk()
    app = ChatEmotionAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()