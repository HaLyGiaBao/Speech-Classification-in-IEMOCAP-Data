# Tên file: AudioDataset.py (Đã sửa)

import os
import torchaudio
import torch
import csv
import pandas as pd

class AudioDataset(torch.utils.data.Dataset):
    """
    Quản lý dữ liệu âm thanh: load từ danh sách files/thư mục,
    load/save danh sách files/labels từ/đến CSV,
    và cung cấp waveform/sr/label VÀ filepath gốc cho DataLoader sau tiền xử lý cơ bản.
    """
    def __init__(self, audio_files_list=None):
        """
        Khởi tạo Dataset.

        Args:
            audio_files_list (list, optional): Danh sách các tuple (filepath, label).
                                               Nếu None, dataset sẽ rỗng ban đầu, cần load từ CSV hoặc dùng collect_audio_files.
        """
        if audio_files_list is None:
             self.audio_files = []
        else:
            if not isinstance(audio_files_list, list):
                 raise TypeError("Đối số 'audio_files_list' phải là một list.")
            self.audio_files = [item for item in audio_files_list if isinstance(item, tuple) and len(item) == 2]
            if len(self.audio_files) != len(audio_files_list):
                 print(f"Warning: Đã lọc bỏ {len(audio_files_list) - len(self.audio_files)} phần tử không hợp lệ từ audio_files_list ban đầu.")


        print(f"AudioDataset được khởi tạo với {len(self.audio_files)} file.")


    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        """
        Lấy waveform, sample rate, label VÀ filepath gốc cho mẫu tại index idx.
        Bao gồm tiền xử lý cơ bản (mono, squeeze, resample về 16kHz).
        Trả về None cho waveform và sr nếu có lỗi khi load/xử lý file.
        """
        if idx >= len(self.audio_files):
             raise IndexError("Index ngoài phạm vi của dataset.")

        filepath, label = self.audio_files[idx]

        try:
            # Load audio file
            waveform, sr = torchaudio.load(filepath) # waveform có shape [channels, time]

            # Chuyển đổi sang mono nếu cần (chỉ lấy kênh đầu tiên)
            if waveform.shape[0] > 1:
                waveform = waveform[0, :]
            waveform = waveform.squeeze(0)


            # Resample về 16kHz nếu sample rate khác
            TARGET_SAMPLE_RATE = 16000
            if sr != TARGET_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
                waveform = resampler(waveform)
                sr = TARGET_SAMPLE_RATE


            # Sửa: Trả về cả filepath
            return waveform, sr, label, filepath # Trả về waveform, sample rate, nhãn, ĐƯỜNG DẪN FILE

        except Exception as e:
            print(f"Lỗi khi load hoặc tiền xử lý file audio {filepath}: {e}")
            # Sửa: Trả về None cho waveform/sr nhưng giữ lại label và filepath
            return None, None, label, filepath # Giữ lại nhãn và đường dẫn ngay cả khi lỗi audio


    @staticmethod
    def collect_audio_files(base_dir, class_names):
        """
        Thu thập danh sách các file audio và nhãn từ cấu trúc thư mục.
        Mỗi thư mục con trong base_dir được coi là một lớp.

        Args:
            base_dir (str): Thư mục gốc chứa các thư mục con là tên lớp.
            class_names (list): Danh sách tên các lớp (tên thư mục con) cần thu thập.

        Returns:
            list: Danh sách các tuple (filepath, label). Trả về list rỗng nếu có lỗi hoặc không tìm thấy file.
        """
        print(f"Đang thu thập file audio từ thư mục: {base_dir}")
        audio_files = []
        if not os.path.isdir(base_dir):
             print(f"Lỗi: Thư mục gốc '{base_dir}' không tồn tại.")
             return audio_files

        for class_name in class_names:
            class_path = os.path.join(base_dir, class_name)
            if os.path.isdir(class_path):
                try:
                    for file in os.listdir(class_path):
                         full_filepath = os.path.join(class_path, file)
                         if os.path.isfile(full_filepath) and file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                              audio_files.append((full_filepath, class_name))

                except Exception as e:
                     print(f"Cảnh báo: Lỗi khi liệt kê file trong thư mục {class_path}: {e}")
            else:
                print(f"Cảnh báo: Thư mục lớp không tồn tại: {class_path}. Bỏ qua.")

        print(f"Tìm thấy tổng cộng {len(audio_files)} file audio hợp lệ.")
        return audio_files

    def save_to_csv(self, csv_path):
        """
        Lưu danh sách (filepath, label) hiện có trong dataset vào file CSV.
        Cột 'audio_path', 'label'.

        Args:
            csv_path (str): Đường dẫn đầy đủ hoặc tương đối đến file CSV sẽ lưu.
        """
        if not self.audio_files:
            print("Không có file audio nào trong dataset để lưu vào CSV.")
            return

        print(f"Đang lưu danh sách file audio vào {csv_path}...")
        try:
            df = pd.DataFrame(self.audio_files, columns=['audio_path', 'label'])
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print("Danh sách file audio đã được lưu thành công.")
        except Exception as e:
            print(f"Lỗi khi lưu danh sách file audio vào CSV {csv_path}: {e}")


    def load_from_csv(self, csv_path):
        """
        Nạp danh sách (filepath, label) vào dataset từ file CSV.
        Cấu trúc CSV mong đợi: phải có cột 'audio_path' và cột 'label'.

        Args:
            csv_path (str): Đường dẫn đến file CSV nguồn.
        """
        if not os.path.exists(csv_path):
             print(f"Lỗi: Không tìm thấy file CSV tại {csv_path}. Dataset sẽ rỗng.")
             self.audio_files = []
             return

        print(f"Đang nạp danh sách file audio từ {csv_path}...")
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')

            if 'audio_path' not in df.columns or 'label' not in df.columns:
                 print(f"Lỗi: File CSV '{csv_path}' phải chứa cột 'audio_path' và 'label'. Dataset sẽ rỗng.")
                 self.audio_files = []
                 return

            self.audio_files = list(zip(df['audio_path'].tolist(), df['label'].tolist()))
            print(f"Đã nạp thành công {len(self.audio_files)} file audio từ CSV.")
        except Exception as e:
            print(f"Lỗi khi nạp danh sách file audio từ CSV {csv_path}: {e}. Dataset sẽ rỗng.")
            self.audio_files = []


# Hàm collate_fn tùy chỉnh cho DataLoader của AudioDataset
# Sửa: Xử lý việc trả về filepath
def audio_collate_fn(batch):
    """
    Hàm collate cho DataLoader của AudioDataset.
    Xử lý các mẫu bị lỗi (__getitem__ trả về None cho waveform) và chuẩn bị dữ liệu cho processor HF.
    Trả về list các waveform, sample rate, list các labels, VÀ list các filepaths.
    """
    # Lọc bỏ các mẫu bị lỗi (item[0] là waveform, item[1] là sr)
    # Giữ lại item[2] (label) và item[3] (filepath) ngay cả khi lỗi load audio
    valid_batch = [item for item in batch if item[0] is not None and item[1] is not None]
    invalid_batch = [item for item in batch if item[0] is None or item[1] is None] # Các mẫu bị lỗi


    # Xử lý các mẫu bị lỗi (chỉ in cảnh báo và bỏ qua)
    if invalid_batch:
        # print(f"Warning: {len(invalid_batch)} mẫu trong batch bị lỗi khi load/tiền xử lý.")
        # Optional: Log which files failed based on invalid_batch[i][3] (filepath)
        pass # Tạm thời chỉ bỏ qua


    # Nếu không còn mẫu hợp lệ nào sau khi lọc
    if not valid_batch:
        # Trả về None cho tất cả để báo hiệu batch không hợp lệ
        return None, None, None, None

    # Tách waveform, sample rate, labels VÀ filepaths từ các mẫu hợp lệ
    waveforms, srs, labels, filepaths = zip(*valid_batch)

    # Các processor của Hugging Face cần list các waveform (Tensor)
    waveforms_list = list(waveforms)

    # Sample rate đã được chuẩn hóa về 16kHz, lấy giá trị đầu tiên
    sample_rate_batch = srs[0]

    # Labels và Filepaths
    labels_list = list(labels)
    filepaths_list = list(filepaths) # Sửa: Thu thập list các filepaths

    # Trả về list các waveform tensors, sample rate duy nhất, list labels, VÀ list filepaths
    return waveforms_list, sample_rate_batch, labels_list, filepaths_list

# --- Cách sử dụng (ví dụ đơn giản trong cùng file) ---
# (Giữ nguyên như trước để test lớp AudioDataset)
if __name__ == "__main__":
    print("--- Test lớp AudioDataset (đã sửa) ---")

    # --- 1. Tạo dữ liệu dummy (giả lập) ---
    dummy_audio_dir = "dummy_audio_data_for_dataset_test_v2"
    os.makedirs(os.path.join(dummy_audio_dir, "cat"), exist_ok=True)
    os.makedirs(os.path.join(dummy_audio_dir, "dog"), exist_ok=True)
    dummy_classes = ["cat", "dog"]
    dummy_csv_audio_list = "dummy_audio_list_test_v2.csv"

    sr_1 = 16000
    sr_2 = 44100
    duration_sec = 0.5
    for cls in dummy_classes:
        for i in range(3):
            dummy_file_path = os.path.join(dummy_audio_dir, cls, f"{cls}_{i+1}.wav")
            sample_rate_to_use = sr_1 if i % 2 == 0 else sr_2
            num_frames = sample_rate_to_use * duration_sec
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

    # --- 2. Test các phương thức của AudioDataset ---
    print("\nTesting collect_audio_files...")
    audio_files_list = AudioDataset.collect_audio_files(dummy_audio_dir, dummy_classes)
    print(f"Tổng số file thu thập được: {len(audio_files_list)}")

    print("\nTesting AudioDataset initialization with list...")
    dataset_from_list = AudioDataset(audio_files_list=audio_files_list)
    print("Dataset size:", len(dataset_from_list))

    print("\nTesting __getitem__ (including filepath)...")
    if len(dataset_from_list) > 0:
        waveform, sr, label, filepath = dataset_from_list[0]
        print(f"First item from __getitem__: waveform shape={waveform.shape}, sr={sr}, label={label}, filepath={filepath}")
        assert filepath == dataset_from_list.audio_files[0][0], "Filepath từ __getitem__ không khớp!"


    print("\nTesting save_to_csv...")
    dataset_from_list.save_to_csv(dummy_csv_audio_list)
    print(f"Đã lưu danh sách file vào '{dummy_csv_audio_list}'.")

    print("\nTesting load_from_csv...")
    dataset_from_csv = AudioDataset()
    dataset_from_csv.load_from_csv(dummy_csv_audio_list)
    print("Dataset size sau khi nạp từ CSV:", len(dataset_from_csv))
    if len(dataset_from_csv) > 0:
        print("File đầu tiên nạp từ CSV:", dataset_from_csv.audio_files[0])
        assert dataset_from_csv.audio_files == audio_files_list, "Dữ liệu nạp từ CSV không khớp với dữ liệu gốc!"
        print("Dữ liệu nạp từ CSV khớp với dữ liệu gốc.")


    print("\nTesting __getitem__ và audio_collate_fn với DataLoader...")
    dataloader_test = torch.utils.data.DataLoader(
        dataset_from_csv,
        batch_size=2,
        shuffle=False,
        collate_fn=audio_collate_fn
    )

    try:
        for i, (waveforms_list, sr_batch, labels_list, filepaths_list) in enumerate(dataloader_test):
             if waveforms_list is None:
                  print(f"Batch {i} bị lỗi hoặc rỗng sau khi lọc.")
                  continue

             print(f"Batch {i}:")
             print(f"  Số lượng waveform trong batch: {len(waveforms_list)}")
             print(f"  Sample rate của batch (mong đợi 16000): {sr_batch}")
             print(f"  Labels trong batch: {labels_list}")
             print(f"  Filepaths trong batch: {filepaths_list}") # Sửa: In filepaths
             if waveforms_list:
                 print(f"  Shape của waveform đầu tiên trong batch: {waveforms_list[0].shape}")

             break

    except Exception as e:
         print(f"Lỗi trong quá trình test DataLoader: {e}")


    # Clean up dummy data (optional)
    # import shutil
    # if os.path.exists(dummy_audio_dir):
    #      shutil.rmtree(dummy_audio_dir)
    # if os.path.exists(dummy_csv_audio_list):
    #      os.remove(dummy_csv_audio_list)


    print("\n--- Test AudioDataset (đã sửa) hoàn thành ---")