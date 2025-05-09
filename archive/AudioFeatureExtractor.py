# Tên file: AudioFeatureExtractor.py (Đã sửa)

import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoFeatureExtractor

# Cần import lớp AudioDataset và hàm collate_fn từ file AudioDataset.py
from AudioDataset import AudioDataset, audio_collate_fn


class AudioFeatureExtractor:
    """
    Trích xuất đặc trưng (embedding) từ dữ liệu âm thanh
    sử dụng các mô hình từ Hugging Face Transformers.
    Quản lý thiết bị (GPU/CPU), thực hiện trích xuất theo batch,
    và xử lý lưu/nạp đặc trưng đã trích xuất VÀ filepath gốc vào/từ file CSV.
    """
    def __init__(self, processor_name_or_path, model_name_or_path, device=None):
        """
        Khởi tạo Feature Extractor.

        Args:
            processor_name_or_path (str): Tên hoặc đường dẫn cục bộ đến processor của mô hình HF.
            model_name_or_path (str): Tên hoặc đường dẫn cục bộ đến mô hình nhúng âm thanh HF.
            device (str, optional): Thiết bị sử dụng để chạy mô hình ('cuda' hoặc 'cpu').
                                    Mặc định: tự động chọn GPU nếu có, ngược lại là CPU.
        """
        self.device = self.set_device(device)
        print(f"Đang sử dụng thiết bị: {self.device}")

        print(f"Đang nạp Processor: {processor_name_or_path}")
        try:
            self.processor = AutoFeatureExtractor.from_pretrained(processor_name_or_path)
        except Exception as e:
            print(f"Lỗi khi nạp Processor {processor_name_or_path}: {e}")
            self.processor = None
            raise # Ném lỗi

        print(f"Đang nạp Mô hình: {model_name_or_path}")
        try:
            self.model = AutoModel.from_pretrained(model_name_or_path)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Lỗi khi nạp Mô hình {model_name_or_path}: {e}")
            self.model = None
            raise # Ném lỗi

        # Kiểm tra cơ bản cấu trúc đầu ra
        dummy_input = torch.randn(1, 16000).to(self.device)
        try:
            inputs = self.processor([dummy_input.cpu().numpy()], sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                dummy_outputs = self.model(**inputs)

            if not hasattr(dummy_outputs, 'last_hidden_state'):
                print("Cảnh báo: Mô hình output không có 'last_hidden_state'. Phương thức trích xuất có thể cần điều chỉnh.")
            else:
                dummy_pooled = torch.mean(dummy_outputs.last_hidden_state, dim=1)
                print(f"Kiểm tra dummy output: pooled embedding shape {dummy_pooled.shape}")

        except Exception as e:
            print(f"Cảnh báo: Không thể kiểm tra cấu trúc đầu ra của mô hình bằng dummy input: {e}")


        print("AudioFeatureExtractor đã được khởi tạo thành công.")


    def set_device(self, device=None):
        """Thiết lập và trả về thiết bị (GPU hoặc CPU)."""
        if device is not None:
             device_lower = device.lower()
             if 'cuda' in device_lower:
                  if torch.cuda.is_available():
                       try:
                            gpu_id_str = device_lower.split('cuda:')
                            if len(gpu_id_str) > 1 and gpu_id_str[1].isdigit():
                                 gpu_id = int(gpu_id_str[1])
                                 if gpu_id < torch.cuda.device_count():
                                      return torch.device(device_lower)
                                 else:
                                      print(f"Cảnh báo: GPU index {gpu_id} không hợp lệ. Sử dụng GPU mặc định hoặc CPU.")
                                      return torch.device("cuda" if torch.cuda.is_available() else "cpu")
                       except Exception:
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


    @torch.no_grad()
    def extract_dataset_embeddings(self, dataset, batch_size=32):
        """
        Trích xuất embedding (pooled) cho toàn bộ dataset bằng DataLoader.

        Args:
            dataset (AudioDataset): Đối tượng AudioDataset đã được khởi tạo.
                                    Mỗi item từ dataset là (waveform, sr, label, filepath).
            batch_size (int): Kích thước batch cho DataLoader.

        Returns:
            list: Danh sách các tuple (embedding_np_array, label, filepath)
                  cho tất cả các mẫu đã xử lý thành công trong dataset.
                  Trả về list rỗng nếu không có mẫu nào xử lý được hoặc có lỗi.
        """
        if self.processor is None or self.model is None:
             print("Lỗi: Processor hoặc mô hình chưa được nạp. Không thể trích xuất đặc trưng cho dataset.")
             return []

        if not isinstance(dataset, AudioDataset):
             print("Lỗi: Đối tượng dataset đầu vào phải là một instance của AudioDataset.")
             return []

        print(f"\nBắt đầu trích xuất đặc trưng cho dataset với batch size {batch_size}...")

        # Tạo DataLoader sử dụng hàm collate_fn tùy chỉnh từ AudioDataset.py
        # Hàm collate_fn cần được định nghĩa hoặc import ở đây
        # from AudioDataset import audio_collate_fn # Uncomment nếu cần
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=audio_collate_fn, # Sử dụng collate_fn đã sửa để trả về filepath
            num_workers=os.cpu_count() // 2 or 1,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        all_pooled_tensors = [] # Lưu tensor embeddings trên device
        all_labels = []         # Lưu labels
        all_filepaths = []      # Lưu filepaths # Sửa: List để lưu filepaths

        processed_count = 0
        total_samples = len(dataset)

        for batch_idx, (waveforms_list, sample_rate_batch, labels_list, filepaths_list) in enumerate(dataloader):
            # collate_fn trả về None nếu batch rỗng sau lọc lỗi
            if waveforms_list is None:
                continue # Bỏ qua batch lỗi


            try:
                # Chuyển waveform tensors sang numpy arrays cho processor
                waveforms_np_list = [w.cpu().numpy() for w in waveforms_list]

                inputs = self.processor(
                     waveforms_np_list,
                     sampling_rate=sample_rate_batch,
                     return_tensors="pt",
                     padding=True
                )

                # Chuyển inputs sang thiết bị
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Đưa inputs vào mô hình
                outputs = self.model(**inputs)

                # Lấy last_hidden_state
                if not hasattr(outputs, 'last_hidden_state'):
                     print(f"\nCảnh báo: Batch {batch_idx}: Mô hình output không có 'last_hidden_state'. Bỏ qua batch.")
                     continue

                last_hidden_states = outputs.last_hidden_state

                # Pooling
                pooled_embeddings_batch = torch.mean(last_hidden_states, dim=1)

                # Thêm tensor pooled, labels, VÀ filepaths của batch vào list thu thập
                all_pooled_tensors.append(pooled_embeddings_batch)
                all_labels.extend(labels_list) # extend list labels
                all_filepaths.extend(filepaths_list) # Sửa: extend list filepaths

                processed_count += pooled_embeddings_batch.shape[0]
                print(f"  Đã xử lý {processed_count}/{total_samples} mẫu...", end='\r')

            except Exception as e:
                 print(f"\nLỗi khi xử lý batch {batch_idx}: {e}. Bỏ qua batch này.")
                 # Cần lưu ý rằng các mẫu bị lỗi trong batch này sẽ bị bỏ qua,
                 # và filepaths của chúng cũng sẽ không được lưu trữ.


        # --- Sau khi xử lý tất cả các batch ---
        print("\nHoàn thành duyệt các batch.")

        if not all_pooled_tensors:
             print("Không có tensor embedding nào được trích xuất thành công.")
             return [] # Trả về list rỗng

        try:
            # Nối tất cả các tensor pooled lại
            final_embeddings_tensor = torch.cat(all_pooled_tensors, dim=0)

            # Chuyển tensor kết quả sang CPU và numpy
            final_embeddings_np = final_embeddings_tensor.cpu().numpy()

            # Ghép cặp numpy array embedding, nhãn, VÀ filepath tương ứng
            # all_labels và all_filepaths đã có cùng thứ tự với các mẫu trong final_embeddings_np
            # sau khi lọc lỗi và nối batch
            all_features_labels_filepaths = list(zip(final_embeddings_np, all_labels, all_filepaths)) # Sửa: Ghép cặp 3 thứ

            print(f"Tổng số mẫu đã trích xuất đặc trưng thành công: {len(all_features_labels_filepaths)}")
            # Sửa: all_features_labels_filepaths là list các tuple (np_array, label, filepath)
            return all_features_labels_filepaths

        except Exception as e:
            print(f"\nLỗi khi nối hoặc chuyển đổi tensor cuối cùng: {e}")
            return []


    @staticmethod # Sửa: Đặt là staticmethod
    def save_features_to_csv(features_labels_filepaths_list, csv_path):
        """
        Lưu danh sách (embedding_np_array, label, filepath) vào file CSV.
        Mỗi chiều của vector embedding sẽ là một cột riêng biệt.
        Thêm cột 'label' và 'audio_path'.

        Args:
            features_labels_filepaths_list (list): Danh sách các tuple (embedding_np_array, label, filepath)
                                                 như trả về từ extract_dataset_embeddings.
            csv_path (str): Đường dẫn đầy đủ hoặc tương đối đến file CSV sẽ lưu.
        """
        if not features_labels_filepaths_list:
            print("Không có đặc trưng nào để lưu vào CSV.")
            return

        print(f"Đang lưu đặc trưng và metadata vào {csv_path}...")
        try:
            # Tách list of (array, label, filepath) thành 3 list riêng
            features_list = [item[0] for item in features_labels_filepaths_list]
            labels_list = [item[1] for item in features_labels_filepaths_list]
            filepaths_list = [item[2] for item in features_labels_filepaths_list] # Sửa: Lấy list filepaths

            # Chuyển list các numpy array (embedding) thành một numpy 2D array duy nhất
            if not all(isinstance(arr, np.ndarray) and arr.shape == features_list[0].shape for arr in features_list):
                 print("Lỗi: Các mảng đặc trưng có shape không nhất quán trước khi lưu CSV.")
                 return # Không lưu nếu có lỗi shape

            features_np = np.array(features_list)

            # Tạo DataFrame từ numpy array đặc trưng
            df_features = pd.DataFrame(features_np)

            # Thêm cột nhãn và cột đường dẫn file
            df_features['label'] = labels_list
            df_features['audio_path'] = filepaths_list # Sửa: Thêm cột audio_path

            # Lưu vào CSV
            df_features.to_csv(csv_path, index=False, encoding='utf-8')

            print("Đặc trưng và metadata đã được lưu thành công.")
        except Exception as e:
            print(f"Lỗi khi lưu đặc trưng vào CSV {csv_path}: {e}")


    @staticmethod # Sửa: Đặt là staticmethod
    def load_features_from_csv(csv_path):
        """
        Nạp đặc trưng, nhãn, VÀ filepath từ file CSV đã được lưu bởi save_features_to_csv.
        Cấu trúc CSV mong đợi: các cột số là đặc trưng, cột 'label', cột 'audio_path'.

        Args:
            csv_path (str): Đường dẫn đến file CSV nguồn.

        Returns:
            tuple: (features_np_array, labels_list, filepaths_list)
                   features_np_array: numpy array shape [số_lượng_mẫu, kích thước_đặc_trưng]
                   labels_list: list các nhãn (string)
                   filepaths_list: list các đường dẫn file (string)
                   Trả về (None, None, None) nếu có lỗi hoặc file không tồn tại.
        """
        if not os.path.exists(csv_path):
             print(f"Lỗi: Không tìm thấy file CSV đặc trưng tại {csv_path}.")
             return None, None, None

        print(f"Đang nạp đặc trưng và metadata từ {csv_path}...")
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')

            # Kiểm tra các cột bắt buộc
            if 'label' not in df.columns or 'audio_path' not in df.columns: # Sửa: Kiểm tra cả audio_path
                 print(f"Lỗi: File CSV đặc trưng '{csv_path}' phải chứa cột 'label' và 'audio_path'.")
                 return None, None, None

            # Tách cột nhãn và đường dẫn file
            labels_list = df['label'].tolist()
            filepaths_list = df['audio_path'].tolist() # Sửa: Lấy list filepaths

            # Lấy các cột còn lại làm đặc trưng. Loại bỏ cột 'label' và 'audio_path'.
            feature_columns = [col for col in df.columns if col not in ['label', 'audio_path']] # Sửa: Loại cả audio_path
            if not feature_columns:
                print(f"Lỗi: Không tìm thấy cột đặc trưng nào trong file CSV '{csv_path}'.")
                return None, None, None

            features_np = df[feature_columns].values # Lấy giá trị dưới dạng numpy array

            # Kiểm tra số lượng mẫu có nhất quán giữa đặc trưng, nhãn, và đường dẫn không
            if not (len(labels_list) == len(filepaths_list) == features_np.shape[0]):
                 print(f"Lỗi: Số lượng mẫu không nhất quán giữa đặc trưng ({features_np.shape[0]}), nhãn ({len(labels_list)}), và đường dẫn ({len(filepaths_list)}) khi nạp từ CSV.")
                 return None, None, None


            print(f"Đã nạp thành công {features_np.shape[0]} mẫu với {features_np.shape[1]} đặc trưng mỗi mẫu.")

            # Sửa: Trả về numpy array đặc trưng, list nhãn, VÀ list filepaths
            return features_np, labels_list, filepaths_list

        except Exception as e:
            print(f"Lỗi khi nạp đặc trưng từ CSV {csv_path}: {e}")
            return None, None, None


# --- Cách sử dụng (ví dụ đơn giản trong cùng file) ---
# (Giữ nguyên như trước)
if __name__ == "__main__":
    print("\n--- Test lớp AudioFeatureExtractor (đã sửa) ---")

    # Sử dụng lại lớp AudioDataset và dummy data
    dummy_audio_dir = "dummy_audio_data_for_extractor_test_v2"
    os.makedirs(os.path.join(dummy_audio_dir, "happy"), exist_ok=True)
    os.makedirs(os.path.join(dummy_audio_dir, "sad"), exist_ok=True)
    dummy_classes_ext = ["happy", "sad"]
    dummy_csv_features_ext = "dummy_features_test_v2.csv"

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

    model_name_test = "hf-internal-testing/tiny-random-WavLMForXVector"


    try:
         extractor_test = AudioFeatureExtractor(
             processor_name_or_path=model_name_test,
             model_name_or_path=model_name_test,
         )

         print("\nTesting extract_dataset_embeddings...")
         # Sửa: extract_dataset_embeddings giờ trả về list (np_array, label, filepath)
         extracted_data_test = extractor_test.extract_dataset_embeddings(
             dataset=dataset_for_extraction_test,
             batch_size=2
         )

         if extracted_data_test:
              print(f"\nTest: Đã trích xuất thành công cho {len(extracted_data_test)} mẫu.")
              if extracted_data_test:
                  print("Test: Dữ liệu mẫu đầu tiên (embedding shape, label, filepath):", extracted_data_test[0][0].shape, extracted_data_test[0][1], extracted_data_test[0][2])


              # Test lưu đặc trưng vào CSV
              print("\nTesting save_features_to_csv...")
              # Sửa: Truyền list (np_array, label, filepath)
              AudioFeatureExtractor.save_features_to_csv(extracted_data_test, dummy_csv_features_ext)
              print(f"Test: Đã lưu đặc trưng vào '{dummy_csv_features_ext}'.")

              # Test nạp đặc trưng từ CSV
              print("\nTesting load_features_from_csv...")
              # Sửa: load_features_from_csv giờ trả về features, labels, filepaths
              loaded_features_test, loaded_labels_test, loaded_filepaths_test = AudioFeatureExtractor.load_features_from_csv(dummy_csv_features_ext)

              if loaded_features_test is not None and loaded_labels_test is not None and loaded_filepaths_test is not None:
                   print(f"\nTest: Đã nạp thành công {len(loaded_labels_test)} mẫu với {loaded_features_test.shape[1]} đặc trưng mỗi mẫu từ CSV.")
                   print("Test: Shape mảng đặc trưng đã nạp:", loaded_features_test.shape)
                   print("Test: Kiểu dữ liệu mảng đặc trưng đã nạp:", type(loaded_features_test))
                   print("Test: Kiểu dữ liệu list nhãn đã nạp:", type(loaded_labels_test))
                   print("Test: Kiểu dữ liệu list filepaths đã nạp:", type(loaded_filepaths_test))
                   print("Test: Đặc trưng mẫu đầu tiên đã nạp (5 phần tử đầu):", loaded_features_test[0, :5])
                   print("Test: Nhãn mẫu đầu tiên đã nạp:", loaded_labels_test[0])
                   print("Test: Filepath mẫu đầu tiên đã nạp:", loaded_filepaths_test[0])

                   # Kiểm tra tính nhất quán số lượng
                   assert len(loaded_labels_test) == len(loaded_filepaths_test) == loaded_features_test.shape[0], "Số lượng mẫu sau nạp không khớp!"

                   # Kiểm tra xem dữ liệu nạp có khớp với dữ liệu gốc không
                   original_features_np = np.array([f[0] for f in extracted_data_test])
                   original_labels_list = [f[1] for f in extracted_data_test]
                   original_filepaths_list = [f[2] for f in extracted_data_test]

                   is_features_match = np.allclose(loaded_features_test, original_features_np)
                   is_labels_match = loaded_labels_test == original_labels_list
                   is_filepaths_match = loaded_filepaths_test == original_filepaths_list

                   print(f"Test: Đặc trưng nạp có khớp với gốc không? {is_features_match}")
                   print(f"Test: Nhãn nạp có khớp với gốc không? {is_labels_match}")
                   print(f"Test: Filepaths nạp có khớp với gốc không? {is_filepaths_match}")
                   assert is_features_match and is_labels_match and is_filepaths_match, "Dữ liệu nạp từ CSV không khớp với dữ liệu gốc đã trích xuất!"
                   print("Test: Dữ liệu nạp từ CSV khớp với dữ liệu gốc đã trích xuất.")


              else:
                   print("\nTest: Nạp đặc trưng từ CSV thất bại.")
         else:
              print("\nTest: Trích xuất đặc trưng dataset thất bại hoặc trả về rỗng.")


    except Exception as e:
         print(f"\nMột lỗi xảy ra trong quá trình test FeatureExtractor: {e}")

    # Clean up dummy data (optional)
    # import shutil
    # if os.path.exists(dummy_audio_dir):
    #      shutil.rmtree(dummy_audio_dir)
    # if os.path.exists(dummy_csv_features_ext):
    #      os.remove(dummy_csv_features_ext)


    print("\n--- Test lớp AudioFeatureExtractor (đã sửa) hoàn thành ---")