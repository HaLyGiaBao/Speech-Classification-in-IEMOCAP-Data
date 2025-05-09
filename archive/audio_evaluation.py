# Tên file: audio_analyzer.py

import os
import torch
import torchaudio
import joblib # Để load mô hình phân loại (ví dụ: SVM)
import csv
# Bạn cần import các thư viện Hugging Face cần thiết cho processor và model nhúng âm thanh
# Ví dụ:
# from transformers import AutoProcessor, AutoModel

class AudioMisclassificationAnalyzer:
    """
    Một lớp để phân tích dữ liệu âm thanh, dự đoán nhãn
    và báo cáo các trường hợp bị phân loại sai.
    """

    def __init__(self, audio_base_path, class_names, classification_model, embedding_processor, embedding_model):
        """
        Khởi tạo đối tượng phân tích.

        Args:
            audio_base_path (str): Đường dẫn gốc chứa các thư mục con,
                                   mỗi thư mục con là một lớp (nhãn thật).
            class_names (list): Danh sách các tên lớp (tương ứng với tên thư mục con).
            classification_model: Mô hình đã được load (ví dụ: joblib.load(...))
                                  có phương thức .predict().
            embedding_processor: Processor từ thư viện Hugging Face (hoặc tương tự)
                                 để chuẩn bị dữ liệu audio cho mô hình nhúng.
            embedding_model: Mô hình nhúng âm thanh từ thư viện Hugging Face (hoặc tương tự)
                             có phương thức .forward().
        """
        self.audio_base_path = audio_base_path
        self.class_names = class_names
        self.classification_model = classification_model
        self.embedding_processor = embedding_processor
        self.embedding_model = embedding_model

        # List để lưu các trường hợp bị sai, sẽ được populate sau khi chạy analyze()
        self.misclassified_samples = []

        print("AudioMisclassificationAnalyzer initialized.")
        print(f"Base audio path: {self.audio_base_path}")
        print(f"Expected classes: {self.class_names}")

    def _get_embedding_from_audio(self, filepath):
        """
        Phương thức trợ giúp (private) để trích xuất embedding từ file audio.
        Dựa trên hàm get_embedding_from_audio ban đầu của bạn.
        """
        try:
            # Tải và tiền xử lý audio
            waveform, sample_rate = torchaudio.load(filepath)  # waveform: [1, time]
            waveform = waveform.squeeze(0)  # từ [1, T] -> [T]

            # Sử dụng processor và model nhúng đã được truyền vào lớp
            inputs = self.embedding_processor(waveform, sampling_rate=sample_rate, return_tensors="pt")

            # Đưa vào thiết bị (CPU/GPU) nếu cần
            # if torch.cuda.is_available():
            #     inputs = {k: v.cuda() for k, v in inputs.items()}
            #     self.embedding_model.cuda() # Chỉ cần gọi 1 lần khi khởi tạo model

            # Lấy embedding
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)

            # Pooling (trung bình qua chiều seq_len)
            hidden_states = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_size]
            pooled = torch.mean(hidden_states, dim=0)  # [hidden_size]

            # Trả về dưới dạng numpy array để phù hợp với mô hình phân loại (thường là scikit-learn)
            return pooled.numpy()

        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            return None # Trả về None nếu có lỗi

    def analyze(self):
        """
        Thực hiện phân tích toàn bộ dataset để tìm các file bị phân loại sai.
        Kết quả được lưu vào self.misclassified_samples.
        """
        self.misclassified_samples = [] # Reset list trước khi bắt đầu phân tích
        print("\nStarting dataset analysis...")

        # Duyệt qua từng lớp (thư mục)
        for true_class in self.class_names:
            print(f"Analyzing class: {true_class}")
            class_dir_path = os.path.join(self.audio_base_path, true_class)

            if not os.path.isdir(class_dir_path):
                print(f"Warning: Directory not found for class '{true_class}' at '{class_dir_path}'. Skipping.")
                continue

            try:
                # Lấy danh sách các file trong thư mục lớp
                audio_files = [f for f in os.listdir(class_dir_path) if os.path.isfile(os.path.join(class_dir_path, f))]
            except Exception as e:
                 print(f"Error listing files in {class_dir_path}: {e}. Skipping class.")
                 continue


            # Duyệt qua từng file audio trong lớp đó
            for audio_file in audio_files:
                audio_path = os.path.join(class_dir_path, audio_file)

                # Lấy embedding bằng phương thức nội bộ của lớp
                embedding = self._get_embedding_from_audio(audio_path)

                if embedding is None:
                    # Đã có thông báo lỗi bên trong _get_embedding_from_audio
                    continue # Bỏ qua file nếu không lấy được embedding

                try:
                    # Chuẩn hoá và dự đoán bằng mô hình phân loại đã được truyền vào
                    # Đảm bảo embedding có shape phù hợp cho .predict (thường là [1, -1])
                    embedding_reshaped = embedding.reshape(1, -1)
                    predicted_label = self.classification_model.predict(embedding_reshaped)[0]

                    # print(f"File: {audio_file}, True: {true_class}, Predicted: {predicted_label}") # Debugging print

                    # So sánh nhãn dự đoán với nhãn thật
                    if predicted_label != true_class:
                        print(f"  Misclassified: {audio_file} (True: {true_class}, Predicted: {predicted_label})")
                        # Lưu thông tin vào list
                        self.misclassified_samples.append([audio_path, predicted_label, true_class])

                except Exception as e:
                     print(f"Error during prediction for {audio_path}: {e}")


        print("\nDataset analysis finished.")
        print(f"Found {len(self.misclassified_samples)} misclassified samples.")

    def save_report_to_csv(self, output_csv_path='misclassification_report.csv'):
        """
        Lưu danh sách các file bị phân loại sai vào một file CSV.

        Args:
            output_csv_path (str): Đường dẫn đầy đủ hoặc tương đối đến file CSV sẽ tạo.
        """
        if not self.misclassified_samples:
            print("No misclassified samples found. CSV file will not be created.")
            return

        try:
            # Mở file CSV để ghi
            # 'newline=''' quan trọng để tránh thêm dòng trống giữa các hàng trên Windows
            # 'encoding='utf-8'' để hỗ trợ các ký tự đặc biệt trong tên file/nhãn
            with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)

                # Ghi tiêu đề
                writer.writerow(["audio_path", "predicted_class", "true_class"])

                # Ghi dữ liệu từ list đã thu thập
                writer.writerows(self.misclassified_samples)

            print(f"Misclassification report saved to '{output_csv_path}'")

        except Exception as e:
            print(f"Error saving report to CSV '{output_csv_path}': {e}")


# Example of how to use this class (optional, for testing within the module file)
if __name__ == "__main__":
    print("--- Testing AudioMisclassificationAnalyzer (if run directly) ---")
    # This block will only run if you execute audio_analyzer.py directly.
    # You'll need dummy models and data for this test.

    # --- Dummy/Mock Objects for Testing ---
    class DummyProcessor:
        def __call__(self, waveform, sampling_rate, return_tensors):
            print("DummyProcessor processing...")
            # Simulate returning dummy inputs tensor
            return {"input_values": torch.randn(1, waveform.shape[0])}

    class DummyEmbeddingModel:
        def __init__(self):
            # Simulate a hidden state size
            self.config = type('Config', (object,), {'hidden_size': 768})() # Mock config
            # self.cuda = lambda: None # Mock cuda method if needed

        def __call__(self, input_values):
             print("DummyEmbeddingModel forwarding...")
             # Simulate returning dummy output with last_hidden_state
             seq_len = input_values.shape[1] // 100 # Simulate some sequence length
             return type('Outputs', (object,), {'last_hidden_state': torch.randn(1, seq_len, 768)})() # Mock outputs

        @torch.no_grad() # Apply decorator to the mock
        def forward(self, input_values):
             return self(input_values) # Call __call__

    class DummyClassificationModel:
        def predict(self, embedding):
            print("DummyClassificationModel predicting...")
            # Simulate a simple prediction logic for testing
            # Replace with your actual prediction logic if testing seriously
            import random
            return random.choice(["class1", "class2", "class3"]) # Predict random class

    # --- Create Dummy Data Structure ---
    dummy_base_path = "dummy_audio_data"
    dummy_classes = ["class1", "class2", "class3"]
    os.makedirs(os.path.join(dummy_base_path, "class1"), exist_ok=True)
    os.makedirs(os.path.join(dummy_base_path, "class2"), exist_ok=True)
    os.makedirs(os.path.join(dummy_base_path, "class3"), exist_ok=True)

    # Create dummy audio files (empty or small binary files are fine for path testing)
    for cls in dummy_classes:
        for i in range(3):
            dummy_audio_file = os.path.join(dummy_base_path, cls, f"audio_{cls}_{i+1}.wav")
            # Create a small valid-ish wav header or just touch the file
            try:
                 # Attempt to create a minimal WAV header for torchaudio load test
                 import wave
                 with wave.open(dummy_audio_file, 'wb') as wf:
                     wf.setnchannels(1)
                     wf.setsampwidth(2) # 2 bytes = 16 bits
                     wf.setframerate(16000)
                     wf.setnframes(16000 * 1) # 1 second of audio
                     wf.writeframes(b'\x00' * (16000 * 1 * 2)) # Silent audio data
            except ImportError:
                 # If wave module is not available, just create an empty file
                 with open(dummy_audio_file, 'wb') as f:
                     f.write(b'\x00' * 100) # Write some dummy bytes

    # --- Initialize and Run Analyzer ---
    # You would replace these with your actual loaded objects
    dummy_processor_obj = DummyProcessor()
    dummy_embedding_model_obj = DummyEmbeddingModel()
    dummy_classification_model_obj = DummyClassificationModel()

    analyzer = AudioMisclassificationAnalyzer(
        audio_base_path=dummy_base_path,
        class_names=dummy_classes,
        classification_model=dummy_classification_model_obj,
        embedding_processor=dummy_processor_obj,
        embedding_model=dummy_embedding_model_obj
    )

    analyzer.analyze()
    analyzer.save_report_to_csv("dummy_misclassification_report.csv")

    print("\n--- Test finished ---")
    # Clean up dummy data (optional)
    # import shutil
    # shutil.rmtree(dummy_base_path)
    # if os.path.exists("dummy_misclassification_report.csv"):
    #     os.remove("dummy_misclassification_report.csv")