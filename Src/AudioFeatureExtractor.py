# Tên file: AudioFeatureExtractor.py (Đã cấu trúc lại)

import os
import torch
import torchaudio
import numpy as np
import pandas as pd
# Import các lớp model cụ thể và AutoFeatureExtractor
from transformers import (
    WavLMModel,
    HubertModel,
    Wav2Vec2Model,
    AutoFeatureExtractor, # Vẫn sử dụng AutoFeatureExtractor cho Processor
    AutoModel # Vẫn giữ AutoModel như fallback hoặc cho các model khác
)

# Import các module cho lớp trừu tượng
from abc import ABC, abstractmethod

# Cần import lớp AudioDataset và hàm collate_fn từ file AudioDataset.py
# (Giả định file AudioDataset.py đã được sửa và định nghĩa audio_collate_fn)
from AudioDataset import AudioDataset, audio_collate_fn


class BaseFeatureExtractor(ABC):
    """
    Lớp cơ sở trừu tượng cho các bộ trích xuất đặc trưng âm thanh
    sử dụng mô hình từ Hugging Face Transformers.
    Chứa logic chung như thiết lập thiết bị và các phương thức static
    cho việc lưu/nạp đặc trưng.
    """
    def __init__(self, model_name_or_path, device=None):
        """
        Khởi tạo lớp cơ sở.

        Args:
            model_name_or_path (str): Tên hoặc đường dẫn cục bộ đến mô hình HF.
            device (str, optional): Thiết bị sử dụng ('cuda' hoặc 'cpu'). Mặc định tự chọn GPU nếu có.
        """
        self.device = self._set_device(device) # Sử dụng _set_device nội bộ
        print(f"Đang sử dụng thiết bị: {self.device}")

        self.processor = None # Sẽ được nạp trong lớp kế thừa
        self.model = None     # Sẽ được nạp trong lớp kế thừa
        self.model_name_or_path = model_name_or_path

        print(f"Khởi tạo BaseFeatureExtractor cho mô hình: {model_name_or_path}")


    def _set_device(self, device=None):
        """
        Thiết lập và trả về thiết bị (GPU hoặc CPU).
        Nếu device là None, tự động chọn 'cuda' nếu khả dụng, ngược lại là 'cpu'.
        (Phương thức nội bộ)
        """
        if device is not None:
             device_lower = str(device).lower() # Convert to string for comparison
             if 'cuda' in device_lower:
                  if torch.cuda.is_available():
                       try:
                            # Handle potential cuda:X format
                            gpu_id_str = device_lower.split('cuda:')
                            if len(gpu_id_str) > 1 and gpu_id_str[1].isdigit():
                                 gpu_id = int(gpu_id_str[1])
                                 if gpu_id < torch.cuda.device_count():
                                      return torch.device(device_lower)
                                 else:
                                      print(f"Cảnh báo: GPU index {gpu_id} không hợp lệ (chỉ có {torch.cuda.device_count()} GPU). Sử dụng GPU mặc định hoặc CPU.")
                                      return torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            else:
                                 # If just 'cuda' or format not standard, use default cuda
                                 return torch.device("cuda" if torch.cuda.is_available() else "cpu")
                       except Exception: # Catch any parsing errors
                             print(f"Cảnh báo: Định dạng thiết bị '{device}' không chuẩn. Sử dụng thiết bị mặc định.")
                             return torch.device("cuda" if torch.cuda.is_available() else "cpu")

                  else:
                       print(f"Cảnh báo: Yêu cầu thiết bị 'cuda' nhưng GPU không khả dụng. Sử dụng CPU.")
                       return torch.device("cpu")
             elif device_lower == 'cpu':
                  return torch.device("cpu")
             else:
                  print(f"Cảnh báo: Tên thiết bị '{device}' không hợp lệ. Sử dụng thiết bị mặc định.")
                  return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")


    @abstractmethod
    def extract_dataset_embeddings(self, dataset: AudioDataset, batch_size: int):
        """
        Phương thức trừu tượng: Trích xuất embedding (pooled) cho toàn bộ dataset.
        Mỗi lớp kế thừa phải triển khai phương thức này.

        Args:
            dataset (AudioDataset): Đối tượng AudioDataset.
            batch_size (int): Kích thước batch cho DataLoader.

        Returns:
            list: Danh sách các tuple (embedding_np_array, label, filepath).
        """
        pass # Phương thức này sẽ được triển khai trong các lớp kế thừa


    @staticmethod
    def save_features_to_csv(features_labels_filepaths_list, csv_path):
        """
        STATIC METHOD: Lưu danh sách (embedding_np_array, label, filepath) vào file CSV.
        Cột 'label' và 'audio_path'.

        Args:
            features_labels_filepaths_list (list): Danh sách các tuple (embedding_np_array, label, filepath).
            csv_path (str): Đường dẫn đến file CSV sẽ lưu.
        """
        if not features_labels_filepaths_list:
            print("Không có đặc trưng nào để lưu vào CSV.")
            return

        print(f"Đang lưu đặc trưng và metadata vào {csv_path}...")
        try:
            features_list = [item[0] for item in features_labels_filepaths_list]
            labels_list = [item[1] for item in features_labels_filepaths_list]
            filepaths_list = [item[2] for item in features_labels_filepaths_list]

            if not features_list or not isinstance(features_list[0], np.ndarray):
                 print("Lỗi: Không có mảng đặc trưng hợp lệ để lưu.")
                 return

            # Kiểm tra shape nhất quán trước khi tạo numpy array
            first_shape = features_list[0].shape
            if not all(isinstance(arr, np.ndarray) and arr.shape == first_shape for arr in features_list):
                 print("Lỗi: Các mảng đặc trưng có shape không nhất quán trước khi lưu CSV.")
                 return

            features_np = np.array(features_list)

            df_features = pd.DataFrame(features_np)
            df_features['label'] = labels_list
            df_features['audio_path'] = filepaths_list

            df_features.to_csv(csv_path, index=False, encoding='utf-8')

            print("Đặc trưng và metadata đã được lưu thành công.")
        except Exception as e:
            print(f"Lỗi khi lưu đặc trưng vào CSV {csv_path}: {e}")


    @staticmethod
    def load_features_from_csv(csv_path):
        """
        STATIC METHOD: Nạp đặc trưng, nhãn, VÀ filepath từ file CSV.
        Cấu trúc CSV mong đợi: các cột số là đặc trưng, cột 'label', cột 'audio_path'.

        Args:
            csv_path (str): Đường dẫn đến file CSV nguồn.

        Returns:
            tuple: (features_np_array, labels_list, filepaths_list).
        """
        if not os.path.exists(csv_path):
             print(f"Lỗi: Không tìm thấy file CSV đặc trưng tại {csv_path}.")
             return None, None, None

        print(f"Đang nạp đặc trưng và metadata từ {csv_path}...")
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')

            if 'label' not in df.columns or 'audio_path' not in df.columns:
                 print(f"Lỗi: File CSV '{csv_path}' phải chứa cột 'label' và 'audio_path'.")
                 return None, None, None

            labels_list = df['label'].tolist()
            filepaths_list = df['audio_path'].tolist()

            feature_columns = [col for col in df.columns if col not in ['label', 'audio_path']]
            if not feature_columns:
                print(f"Lỗi: Không tìm thấy cột đặc trưng nào trong file CSV '{csv_path}'.")
                return None, None, None

            features_np = df[feature_columns].values

            if not (len(labels_list) == len(filepaths_list) == features_np.shape[0]):
                 print(f"Lỗi: Số lượng mẫu không nhất quán giữa đặc trưng ({features_np.shape[0]}), nhãn ({len(labels_list)}), và đường dẫn ({len(filepaths_list)}) khi nạp từ CSV.")
                 return None, None, None

            print(f"Đã nạp thành công {features_np.shape[0]} mẫu với {features_np.shape[1]} đặc trưng mỗi mẫu.")

            return features_np, labels_list, filepaths_list

        except Exception as e:
            print(f"Lỗi khi nạp đặc trưng từ CSV {csv_path}: {e}")
            return None, None, None


class WavLMFeatureExtractor(BaseFeatureExtractor):
    """Bộ trích xuất đặc trưng sử dụng mô hình WavLM."""
    def __init__(self, model_name_or_path, device=None):
        # Gọi lớp cha để thiết lập thiết bị
        super().__init__(model_name_or_path, device)

        # Nạp Processor cụ thể (WavLM thường dùng Wav2Vec2FeatureExtractor hoặc Auto)
        print(f"Đang nạp Processor: {model_name_or_path}")
        try:
            # AutoFeatureExtractor là cách linh hoạt nhất và được khuyến nghị
            self.processor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        except Exception as e:
            print(f"Lỗi khi nạp Processor {model_name_or_path}: {e}")
            self.processor = None
            raise # Ném lỗi

        # Nạp Mô hình cụ thể WavLM
        print(f"Đang nạp Mô hình WavLM: {model_name_or_path}")
        try:
            # Sử dụng lớp mô hình WavLMModel cụ thể
            self.model = WavLMModel.from_pretrained(model_name_or_path)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Lỗi khi nạp Mô hình WavLM {model_name_or_path}: {e}")
            self.model = None
            raise # Ném lỗi

        # Kiểm tra cơ bản cấu trúc đầu ra (có last_hidden_state không)
        self._check_model_output()
        print("WavLMFeatureExtractor đã được khởi tạo thành công.")


    @torch.no_grad()
    def extract_dataset_embeddings(self, dataset: AudioDataset, batch_size: int):
        """Trích xuất embedding (pooled) cho dataset sử dụng mô hình WavLM."""
        if self.processor is None or self.model is None:
             print("Lỗi: Processor hoặc mô hình WavLM chưa được nạp.")
             return []
        if not isinstance(dataset, AudioDataset):
             print("Lỗi: Dataset phải là instance của AudioDataset.")
             return []

        print(f"\nBắt đầu trích xuất đặc trưng (WavLM) với batch size {batch_size}...")

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=audio_collate_fn, # Sử dụng collate_fn đã sửa
            num_workers=os.cpu_count() // 2 or 1,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        all_pooled_tensors = []
        all_labels = []
        all_filepaths = []

        processed_count = 0
        total_samples = len(dataset)

        for batch_idx, (waveforms_list, sample_rate_batch, labels_list, filepaths_list) in enumerate(dataloader):
            if waveforms_list is None:
                # Cập nhật processed_count để tính cả các mẫu lỗi đã bị bỏ qua
                processed_count += len(labels_list) # labels_list vẫn có item cho mẫu lỗi
                continue # Bỏ qua batch lỗi


            try:
                waveforms_np_list = [w.cpu().numpy() for w in waveforms_list]

                inputs = self.processor(
                     waveforms_np_list,
                     sampling_rate=sample_rate_batch,
                     return_tensors="pt",
                     padding=True
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)

                if not hasattr(outputs, 'last_hidden_state'):
                     print(f"\nCảnh báo: Batch {batch_idx}: Mô hình WavLM output không có 'last_hidden_state'. Bỏ qua batch.")
                     continue

                last_hidden_states = outputs.last_hidden_state # Shape [batch_size, seq_len, hidden_size]
                pooled_embeddings_batch = torch.mean(last_hidden_states, dim=1) # Shape [batch_size, hidden_size]

                all_pooled_tensors.append(pooled_embeddings_batch)
                all_labels.extend(labels_list)
                all_filepaths.extend(filepaths_list)

                processed_count += pooled_embeddings_batch.shape[0] # Chỉ tăng processed_count cho mẫu xử lý thành công
                print(f"  Đã xử lý {processed_count}/{total_samples} mẫu...", end='\r')

            except Exception as e:
                 print(f"\nLỗi khi xử lý batch {batch_idx} (WavLM): {e}. Bỏ qua batch này.")
                 # Các mẫu trong batch này bị lỗi, nên bỏ qua filepaths và labels tương ứng của batch này khỏi kết quả cuối cùng


        print("\nHoàn thành duyệt các batch (WavLM).")

        if not all_pooled_tensors:
             print("Không có tensor embedding nào được trích xuất thành công (WavLM).")
             return []

        try:
            final_embeddings_tensor = torch.cat(all_pooled_tensors, dim=0)
            final_embeddings_np = final_embeddings_tensor.cpu().numpy()

            # Ghép cặp kết quả chỉ từ các mẫu đã xử lý thành công
            all_features_labels_filepaths = list(zip(final_embeddings_np, all_labels, all_filepaths))

            print(f"Tổng số mẫu đã trích xuất đặc trưng thành công (WavLM): {len(all_features_labels_filepaths)}")
            return all_features_labels_filepaths

        except Exception as e:
            print(f"\nLỗi khi nối hoặc chuyển đổi tensor cuối cùng (WavLM): {e}")
            return []

    def _check_model_output(self):
        """Kiểm tra cấu trúc đầu ra của mô hình cụ thể."""
        if self.processor is None or self.model is None:
             print("Không thể kiểm tra cấu trúc đầu ra: Processor hoặc mô hình chưa được nạp.")
             return

        dummy_input = torch.randn(1, 16000).to(self.device)
        try:
            inputs = self.processor([dummy_input.cpu().numpy()], sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                dummy_outputs = self.model(**inputs)

            if not hasattr(dummy_outputs, 'last_hidden_state'):
                print("Cảnh báo: Mô hình cụ thể không có 'last_hidden_state' trong đầu ra. Phương thức trích xuất có thể cần điều chỉnh.")
            else:
                dummy_pooled = torch.mean(dummy_outputs.last_hidden_state, dim=1)
                print(f"Kiểm tra dummy output (WavLM): pooled embedding shape {dummy_pooled.shape}")

        except Exception as e:
            print(f"Cảnh báo: Không thể kiểm tra cấu trúc đầu ra của mô hình WavLM bằng dummy input: {e}")


class HubertFeatureExtractor(BaseFeatureExtractor):
    """Bộ trích xuất đặc trưng sử dụng mô hình HuBERT."""
    def __init__(self, model_name_or_path, device=None):
        super().__init__(model_name_or_path, device)

        # Nạp Processor (HuBERT thường dùng Wav2Vec2FeatureExtractor hoặc Auto)
        print(f"Đang nạp Processor: {model_name_or_path}")
        try:
            self.processor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        except Exception as e:
            print(f"Lỗi khi nạp Processor {model_name_or_path}: {e}")
            self.processor = None
            raise # Ném lỗi

        # Nạp Mô hình cụ thể HuBERT
        print(f"Đang nạp Mô hình HuBERT: {model_name_or_path}")
        try:
            # Sử dụng lớp mô hình HubertModel cụ thể
            self.model = HubertModel.from_pretrained(model_name_or_path)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Lỗi khi nạp Mô hình HuBERT {model_name_or_path}: {e}")
            self.model = None
            raise # Ném lỗi

        # Kiểm tra cơ bản cấu trúc đầu ra
        self._check_model_output()
        print("HubertFeatureExtractor đã được khởi tạo thành công.")

    @torch.no_grad()
    def extract_dataset_embeddings(self, dataset: AudioDataset, batch_size: int):
        """Trích xuất embedding (pooled) cho dataset sử dụng mô hình HuBERT."""
        if self.processor is None or self.model is None:
             print("Lỗi: Processor hoặc mô hình HuBERT chưa được nạp.")
             return []
        if not isinstance(dataset, AudioDataset):
             print("Lỗi: Dataset phải là instance của AudioDataset.")
             return []

        print(f"\nBắt đầu trích xuất đặc trưng (HuBERT) với batch size {batch_size}...")

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=audio_collate_fn,
            num_workers=os.cpu_count() // 2 or 1,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        all_pooled_tensors = []
        all_labels = []
        all_filepaths = []

        processed_count = 0
        total_samples = len(dataset)

        for batch_idx, (waveforms_list, sample_rate_batch, labels_list, filepaths_list) in enumerate(dataloader):
            if waveforms_list is None:
                 processed_count += len(labels_list)
                 continue

            try:
                waveforms_np_list = [w.cpu().numpy() for w in waveforms_list]

                inputs = self.processor(
                     waveforms_np_list,
                     sampling_rate=sample_rate_batch,
                     return_tensors="pt",
                     padding=True
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)

                # HuBERT cũng thường có last_hidden_state
                if not hasattr(outputs, 'last_hidden_state'):
                     print(f"\nCảnh báo: Batch {batch_idx}: Mô hình HuBERT output không có 'last_hidden_state'. Bỏ qua batch.")
                     continue

                last_hidden_states = outputs.last_hidden_state # Shape [batch_size, seq_len, hidden_size]
                pooled_embeddings_batch = torch.mean(last_hidden_states, dim=1) # Shape [batch_size, hidden_size]

                all_pooled_tensors.append(pooled_embeddings_batch)
                all_labels.extend(labels_list)
                all_filepaths.extend(filepaths_list)

                processed_count += pooled_embeddings_batch.shape[0]
                print(f"  Đã xử lý {processed_count}/{total_samples} mẫu...", end='\r')

            except Exception as e:
                 print(f"\nLỗi khi xử lý batch {batch_idx} (HuBERT): {e}. Bỏ qua batch này.")


        print("\nHoàn thành duyệt các batch (HuBERT).")

        if not all_pooled_tensors:
             print("Không có tensor embedding nào được trích xuất thành công (HuBERT).")
             return []

        try:
            final_embeddings_tensor = torch.cat(all_pooled_tensors, dim=0)
            final_embeddings_np = final_embeddings_tensor.cpu().numpy()

            all_features_labels_filepaths = list(zip(final_embeddings_np, all_labels, all_filepaths))

            print(f"Tổng số mẫu đã trích xuất đặc trưng thành công (HuBERT): {len(all_features_labels_filepaths)}")
            return all_features_labels_filepaths

        except Exception as e:
            print(f"\nLỗi khi nối hoặc chuyển đổi tensor cuối cùng (HuBERT): {e}")
            return []

    def _check_model_output(self):
        """Kiểm tra cấu trúc đầu ra của mô hình cụ thể."""
        if self.processor is None or self.model is None:
             print("Không thể kiểm tra cấu trúc đầu ra: Processor hoặc mô hình chưa được nạp.")
             return

        dummy_input = torch.randn(1, 16000).to(self.device)
        try:
            inputs = self.processor([dummy_input.cpu().numpy()], sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                dummy_outputs = self.model(**inputs)

            if not hasattr(dummy_outputs, 'last_hidden_state'):
                print("Cảnh báo: Mô hình cụ thể không có 'last_hidden_state' trong đầu ra. Phương thức trích xuất có thể cần điều chỉnh.")
            else:
                dummy_pooled = torch.mean(dummy_outputs.last_hidden_state, dim=1)
                print(f"Kiểm tra dummy output (HuBERT): pooled embedding shape {dummy_pooled.shape}")

        except Exception as e:
            print(f"Cảnh báo: Không thể kiểm tra cấu trúc đầu ra của mô hình HuBERT bằng dummy input: {e}")


class Wav2Vec2EncoderFeatureExtractor(BaseFeatureExtractor):
    """Bộ trích xuất đặc trưng sử dụng mô hình Wav2Vec2 (phần Encoder)."""
    def __init__(self, model_name_or_path, device=None):
        super().__init__(model_name_or_path, device)

        # Nạp Processor (Wav2Vec2 dùng Wav2Vec2FeatureExtractor hoặc Auto)
        print(f"Đang nạp Processor: {model_name_or_path}")
        try:
            self.processor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        except Exception as e:
            print(f"Lỗi khi nạp Processor {model_name_or_path}: {e}")
            self.processor = None
            raise # Ném lỗi

        # Nạp Mô hình cụ thể Wav2Vec2 (chỉ phần encoder)
        # Lưu ý: Nếu mô hình là Wav2Vec2ForCTC, việc nạp bằng Wav2Vec2Model chỉ lấy phần Encoder.
        # Nếu bạn muốn trích xuất từ các layer cụ thể, cần tùy chỉnh thêm.
        print(f"Đang nạp Mô hình Wav2Vec2 (Encoder): {model_name_or_path}")
        try:
            # Sử dụng lớp mô hình Wav2Vec2Model cụ thể
            self.model = Wav2Vec2Model.from_pretrained(model_name_or_path)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Lỗi khi nạp Mô hình Wav2Vec2 {model_name_or_path}: {e}")
            self.model = None
            raise # Ném lỗi

        # Kiểm tra cơ bản cấu trúc đầu ra
        self._check_model_output()
        print("Wav2Vec2EncoderFeatureExtractor đã được khởi tạo thành công.")

    @torch.no_grad()
    def extract_dataset_embeddings(self, dataset: AudioDataset, batch_size: int):
        """Trích xuất embedding (pooled) cho dataset sử dụng mô hình Wav2Vec2."""
        if self.processor is None or self.model is None:
             print("Lỗi: Processor hoặc mô hình Wav2Vec2 chưa được nạp.")
             return []
        if not isinstance(dataset, AudioDataset):
             print("Lỗi: Dataset phải là instance của AudioDataset.")
             return []

        print(f"\nBắt đầu trích xuất đặc trưng (Wav2Vec2) với batch size {batch_size}...")

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=audio_collate_fn,
            num_workers=os.cpu_count() // 2 or 1,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        all_pooled_tensors = []
        all_labels = []
        all_filepaths = []

        processed_count = 0
        total_samples = len(dataset)

        for batch_idx, (waveforms_list, sample_rate_batch, labels_list, filepaths_list) in enumerate(dataloader):
            if waveforms_list is None:
                 processed_count += len(labels_list)
                 continue

            try:
                waveforms_np_list = [w.cpu().numpy() for w in waveforms_list]

                inputs = self.processor(
                     waveforms_np_list,
                     sampling_rate=sample_rate_batch,
                     return_tensors="pt",
                     padding=True
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)

                # Wav2Vec2 cũng có last_hidden_state
                if not hasattr(outputs, 'last_hidden_state'):
                     print(f"\nCảnh báo: Batch {batch_idx}: Mô hình Wav2Vec2 output không có 'last_hidden_state'. Bỏ qua batch.")
                     continue

                last_hidden_states = outputs.last_hidden_state # Shape [batch_size, seq_len, hidden_size]
                pooled_embeddings_batch = torch.mean(last_hidden_states, dim=1) # Shape [batch_size, hidden_size]


                all_pooled_tensors.append(pooled_embeddings_batch)
                all_labels.extend(labels_list)
                all_filepaths.extend(filepaths_list)

                processed_count += pooled_embeddings_batch.shape[0]
                print(f"  Đã xử lý {processed_count}/{total_samples} mẫu...", end='\r')

            except Exception as e:
                 print(f"\nLỗi khi xử lý batch {batch_idx} (Wav2Vec2): {e}. Bỏ qua batch này.")


        print("\nHoàn thành duyệt các batch (Wav2Vec2).")

        if not all_pooled_tensors:
             print("Không có tensor embedding nào được trích xuất thành công (Wav2Vec2).")
             return []

        try:
            final_embeddings_tensor = torch.cat(all_pooled_tensors, dim=0)
            final_embeddings_np = final_embeddings_tensor.cpu().numpy()

            all_features_labels_filepaths = list(zip(final_embeddings_np, all_labels, all_filepaths))

            print(f"Tổng số mẫu đã trích xuất đặc trưng thành công (Wav2Vec2): {len(all_features_labels_filepaths)}")
            return all_features_labels_filepaths

        except Exception as e:
            print(f"\nLỗi khi nối hoặc chuyển đổi tensor cuối cùng (Wav2Vec2): {e}")
            return []

    def _check_model_output(self):
        """Kiểm tra cấu trúc đầu ra của mô hình cụ thể."""
        if self.processor is None or self.model is None:
             print("Không thể kiểm tra cấu trúc đầu ra: Processor hoặc mô hình chưa được nạp.")
             return

        dummy_input = torch.randn(1, 16000).to(self.device)
        try:
            inputs = self.processor([dummy_input.cpu().numpy()], sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                dummy_outputs = self.model(**inputs)

            if not hasattr(dummy_outputs, 'last_hidden_state'):
                print("Cảnh báo: Mô hình cụ thể không có 'last_hidden_state' trong đầu ra. Phương thức trích xuất có thể cần điều chỉnh.")
            else:
                dummy_pooled = torch.mean(dummy_outputs.last_hidden_state, dim=1)
                print(f"Kiểm tra dummy output (Wav2Vec2): pooled embedding shape {dummy_pooled.shape}")

        except Exception as e:
            print(f"Cảnh báo: Không thể kiểm tra cấu trúc đầu ra của mô hình Wav2Vec2 bằng dummy input: {e}")


# --- Cách sử dụng (ví dụ đơn giản trong cùng file) ---
# (Giữ nguyên như trước)
if __name__ == "__main__":
    print("\n--- Test các lớp FeatureExtractor (đã cấu trúc lại) ---")

    # Sử dụng lại lớp AudioDataset và dummy data
    dummy_audio_dir = "dummy_audio_data_for_extractor_test_v3"
    os.makedirs(os.path.join(dummy_audio_dir, "test_A"), exist_ok=True)
    os.makedirs(os.path.join(dummy_audio_dir, "test_B"), exist_ok=True)
    dummy_classes_ext = ["test_A", "test_B"]
    dummy_csv_features_ext_wavlm = "dummy_features_wavlm_test_v3.csv"
    dummy_csv_features_ext_hubert = "dummy_features_hubert_test_v3.csv"
    dummy_csv_features_ext_w2v2 = "dummy_features_w2v2_test_v3.csv"


    sr_test = 16000
    duration_sec_test = 1
    for cls in dummy_classes_ext:
        for i in range(2):
            dummy_file_path = os.path.join(dummy_audio_dir, cls, f"{cls}_ext_{i+1}.wav")
            sample_rate_to_use = sr_test
            num_frames = sample_rate_to_use * duration_sec_test
            try:
                 import wave
                 with wave.open(dummy_file_path, 'wb') as wf:
                     wf.setnchannels(1)
                     wf.setsampwidth(2)
                     wf.setframerate(sample_rate_to_use)
                     wf.setnframes(int(num_frames))
                     wf.writeframes(b'\x00' * int(num_frames) * 2)
            except ImportError:
                 with open(dummy_file_path, 'wb') as f:
                     f.write(b'')


    audio_files_list_ext = AudioDataset.collect_audio_files(dummy_audio_dir, dummy_classes_ext)
    dataset_for_extraction_test = AudioDataset(audio_files_list=audio_files_list_ext)
    print(f"Dataset cho trích xuất test có {len(dataset_for_extraction_test)} file.")

    # --- Test với từng loại Feature Extractor ---

    # Test WavLM
    print("\n--- Testing WavLMFeatureExtractor ---")
    model_name_wavlm = "microsoft/wavlm-base" # Sử dụng base model cho test nhanh
    try:
        extractor_wavlm = WavLMFeatureExtractor(
            model_name_or_path=model_name_wavlm,
            device='cpu' # Chỉ định CPU cho test nếu cần
        )
        extracted_data_wavlm = extractor_wavlm.extract_dataset_embeddings(
            dataset=dataset_for_extraction_test,
            batch_size=2
        )
        if extracted_data_wavlm:
            print(f"Test WavLM: Đã trích xuất thành công cho {len(extracted_data_wavlm)} mẫu.")
            BaseFeatureExtractor.save_features_to_csv(extracted_data_wavlm, dummy_csv_features_ext_wavlm)
            loaded_data_wavlm = BaseFeatureExtractor.load_features_from_csv(dummy_csv_features_ext_wavlm)
            if loaded_data_wavlm and loaded_data_wavlm[0] is not None:
                 print(f"Test WavLM: Đã nạp thành công {loaded_data_wavlm[0].shape[0]} mẫu từ CSV.")
                 # Thêm kiểm tra data consistency nếu cần
        else:
             print("Test WavLM: Trích xuất thất bại.")

    except Exception as e:
        print(f"Lỗi xảy ra khi test WavLMFeatureExtractor: {e}")


    # Test HuBERT
    print("\n--- Testing HubertFeatureExtractor ---")
    model_name_hubert = "facebook/hubert-base-ls960" # Sử dụng base model cho test nhanh
    try:
        extractor_hubert = HubertFeatureExtractor(
            model_name_or_path=model_name_hubert,
            device='cpu'
        )
        extracted_data_hubert = extractor_hubert.extract_dataset_embeddings(
            dataset=dataset_for_extraction_test,
            batch_size=2
        )
        if extracted_data_hubert:
            print(f"Test HuBERT: Đã trích xuất thành công cho {len(extracted_data_hubert)} mẫu.")
            BaseFeatureExtractor.save_features_to_csv(extracted_data_hubert, dummy_csv_features_ext_hubert)
            loaded_data_hubert = BaseFeatureExtractor.load_features_from_csv(dummy_csv_features_ext_hubert)
            if loaded_data_hubert and loaded_data_hubert[0] is not None:
                 print(f"Test HuBERT: Đã nạp thành công {loaded_data_hubert[0].shape[0]} mẫu từ CSV.")
        else:
             print("Test HuBERT: Trích xuất thất bại.")

    except Exception as e:
        print(f"Lỗi xảy ra khi test HubertFeatureExtractor: {e}")


    # Test Wav2Vec2Encoder
    print("\n--- Testing Wav2Vec2EncoderFeatureExtractor ---")
    model_name_w2v2 = "facebook/wav2vec2-base-960h" # Sử dụng base model cho test nhanh
    try:
        extractor_w2v2 = Wav2Vec2EncoderFeatureExtractor(
            model_name_or_path=model_name_w2v2,
            device='cpu'
        )
        extracted_data_w2v2 = extractor_w2v2.extract_dataset_embeddings(
            dataset=dataset_for_extraction_test,
            batch_size=2
        )
        if extracted_data_w2v2:
            print(f"Test Wav2Vec2: Đã trích xuất thành công cho {len(extracted_data_w2v2)} mẫu.")
            BaseFeatureExtractor.save_features_to_csv(extracted_data_w2v2, dummy_csv_features_ext_w2v2)
            loaded_data_w2v2 = BaseFeatureExtractor.load_features_from_csv(dummy_csv_features_ext_w2v2)
            if loaded_data_w2v2 and loaded_data_w2v2[0] is not None:
                 print(f"Test Wav2Vec2: Đã nạp thành công {loaded_data_w2v2[0].shape[0]} mẫu từ CSV.")
        else:
             print("Test Wav2Vec2: Trích xuất thất bại.")

    except Exception as e:
        print(f"Lỗi xảy ra khi test Wav2Vec2EncoderFeatureExtractor: {e}")


    # Clean up dummy data (optional)
    # import shutil
    # if os.path.exists(dummy_audio_dir):
    #      shutil.rmtree(dummy_audio_dir)
    # if os.path.exists(dummy_csv_features_ext_wavlm):
    #      os.remove(dummy_csv_features_ext_wavlm)
    # if os.path.exists(dummy_csv_features_ext_hubert):
    #      os.remove(dummy_csv_features_ext_hubert)
    # if os.path.exists(dummy_csv_features_ext_w2v2):
    #      os.remove(dummy_csv_features_ext_w2v2)


    print("\n--- Test các lớp FeatureExtractor (đã cấu trúc lại) hoàn thành ---")