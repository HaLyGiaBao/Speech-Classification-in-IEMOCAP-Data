# Tên file: MisclassificationReporter.py

import os
import csv
import pandas as pd
import numpy as np

class MisclassificationReporter:
    """
    Lớp để tạo các báo cáo về các mẫu bị phân loại sai.
    Hỗ trợ tạo báo cáo chỉ chứa các mẫu sai hoặc báo cáo so sánh cho tất cả mẫu test.
    """
    def __init__(self, base_output_dir="reports"):
        """
        Khởi tạo MisclassificationReporter.

        Args:
            base_output_dir (str): Thư mục gốc để lưu các báo cáo.
                                   Nếu thư mục chưa tồn tại, nó sẽ được tạo.
        """
        self.base_output_dir = base_output_dir
        os.makedirs(self.base_output_dir, exist_ok=True)
        print(f"MisclassificationReporter initialized. Reports will be saved in '{self.base_output_dir}'.")


    def generate_misclassified_csv(self, y_true, y_pred, sample_identifiers, report_name):
        """
        So sánh nhãn thật và nhãn dự đoán, sau đó lưu các mẫu bị sai
        vào một file CSV riêng biệt chỉ chứa các mẫu đó.
        Tên file sẽ là <report_name>_misclassified.csv trong base_output_dir.

        Args:
            y_true (list hoặc np.ndarray): Nhãn thật của các mẫu.
            y_pred (list hoặc np.ndarray): Nhãn dự đoán của các mẫu.
            sample_identifiers (list): Danh sách các định danh duy nhất cho mỗi mẫu
                                       (ví dụ: đường dẫn file audio).
                                       Thứ tự phải khớp với y_true và y_pred.
            report_name (str): Tên mô tả cho báo cáo này (ví dụ: "SVC_WavLM", "UL_Ensemble").
                               Sẽ dùng để tạo tên file.
        Returns:
            str or None: Đường dẫn đến file CSV đã lưu nếu thành công, None nếu có lỗi.
        """
        print(f"\nĐang tạo báo cáo misclassification (chỉ mẫu sai) cho: {report_name}")

        if len(y_true) != len(y_pred) or len(y_true) != len(sample_identifiers):
            print(f"Lỗi: Số lượng nhãn thật ({len(y_true)}), dự đoán ({len(y_pred)}), và định danh mẫu ({len(sample_identifiers)}) không khớp.")
            return None

        misclassified_samples_data = []
        for i in range(len(y_true)):
            # Chuyển sang string để so sánh an toàn hơn, đề phòng np.array so sánh không đúng type
            if str(y_pred[i]) != str(y_true[i]):
                misclassified_samples_data.append([sample_identifiers[i], y_pred[i], y_true[i]])

        # Tạo đường dẫn file output
        output_filename = f"{report_name}_misclassified.csv"
        output_csv_path = os.path.join(self.base_output_dir, output_filename)

        if not misclassified_samples_data:
            print(f"Không tìm thấy mẫu nào bị phân loại sai cho {report_name}. Tạo file rỗng.")
            # Tạo file CSV rỗng chỉ với header
            try:
                 with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
                      writer = csv.writer(file)
                      writer.writerow(["sample_identifier", "predicted_class", "true_class"]) # Header
                 print(f"Đã tạo file CSV rỗng '{output_csv_path}'.")
                 return output_csv_path
            except Exception as e:
                 print(f"Lỗi khi cố gắng tạo file CSV rỗng '{output_csv_path}': {e}")
                 return None

        try:
            with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["sample_identifier", "predicted_class", "true_class"]) # Header
                writer.writerows(misclassified_samples_data) # Data

            print(f"Đã lưu báo cáo misclassification (chỉ mẫu sai) cho {report_name} vào '{output_csv_path}'")
            return output_csv_path

        except Exception as e:
            print(f"Lỗi khi lưu báo cáo misclassification CSV '{output_csv_path}': {e}")
            return None


    def generate_comparison_report(self, sample_identifiers, y_true, predictions_dict, output_filename="comparison_report.csv"):
        """
        Tạo hoặc cập nhật một file báo cáo CSV duy nhất chứa tất cả các mẫu,
        với các cột cho nhãn thật và dự đoán từ nhiều mô hình/chiến lược.

        Args:
            sample_identifiers (list): Danh sách các định danh duy nhất cho MỌI MẪU trong tập kiểm tra.
                                       Sẽ là cột đầu tiên.
            y_true (list hoặc np.ndarray): Nhãn thật cho MỌI MẪU trong tập kiểm tra.
                                          Thứ tự phải khớp với sample_identifiers.
            predictions_dict (dict): Một dictionary nơi key là tên mô hình/chiến lược (str)
                                     và value là np.ndarray hoặc list các nhãn dự đoán
                                     cho MỌI MẪU trong tập kiểm tra.
                                     Số lượng dự đoán trong mỗi value phải khớp với len(sample_identifiers).
            output_filename (str): Tên file CSV sẽ tạo/cập nhật trong base_output_dir.
                                   Mặc định là 'comparison_report.csv'.
        Returns:
            str or None: Đường dẫn đến file CSV đã lưu nếu thành công, None nếu có lỗi.
        """
        print(f"\nĐang tạo/cập nhật báo cáo so sánh: {output_filename}")

        if len(sample_identifiers) == 0:
             print("Lỗi: Không có định danh mẫu nào được cung cấp để tạo báo cáo so sánh.")
             return None
        if len(sample_identifiers) != len(y_true):
             print(f"Lỗi: Số lượng định danh mẫu ({len(sample_identifiers)}) không khớp với số lượng nhãn thật ({len(y_true)}).")
             return None
        if not predictions_dict:
             print("Cảnh báo: Không có dự đoán nào được cung cấp trong predictions_dict để thêm vào báo cáo so sánh.")
             # Vẫn tạo báo cáo cơ sở nếu có mẫu
             pass


        # Tạo DataFrame cơ sở với định danh mẫu và nhãn thật
        report_data = {'sample_identifier': sample_identifiers, 'true_class': y_true}

        # Thêm các cột dự đoán từ dictionary
        for name, y_pred in predictions_dict.items():
            if len(y_pred) != len(sample_identifiers):
                 print(f"Warning: Số lượng dự đoán cho '{name}' ({len(y_pred)}) không khớp với số lượng mẫu ({len(sample_identifiers)}). Bỏ qua cột này.")
                 continue # Bỏ qua cột dự đoán này nếu số lượng không khớp

            # Sử dụng tên cột rõ ràng hơn
            column_name = f"predicted_{name}"
            # Kiểm tra xem tên cột có bị trùng không và thêm suffix nếu cần
            original_col_name = column_name
            counter = 1
            while column_name in report_data:
                 column_name = f"{original_col_name}_{counter}"
                 counter += 1

            report_data[column_name] = y_pred # Thêm cột dự đoán vào dictionary


        # Tạo DataFrame từ dictionary
        try:
            df_report = pd.DataFrame(report_data)
        except Exception as e:
            print(f"Lỗi khi tạo DataFrame cho báo cáo so sánh: {e}")
            return None


        # Lưu DataFrame vào CSV
        output_csv_path = os.path.join(self.base_output_dir, output_filename)

        try:
            # index=False để không ghi index của DataFrame vào CSV
            # encoding='utf-8' để hỗ trợ các ký tự đặc biệt
            df_report.to_csv(output_csv_path, index=False, encoding='utf-8')
            print(f"Đã lưu báo cáo so sánh vào '{output_csv_path}'")
            return output_csv_path

        except Exception as e:
            print(f"Lỗi khi lưu báo cáo so sánh CSV '{output_csv_path}': {e}")
            return None


# --- Cách sử dụng (ví dụ đơn giản trong cùng file) ---
if __name__ == "__main__":
    print("--- Test lớp MisclassificationReporter ---")

    # Dữ liệu dummy
    y_true_dummy = np.array(['cat', 'dog', 'cat', 'dog', 'cat', 'dog'])
    y_pred_model1_dummy = np.array(['cat', 'cat', 'dog', 'dog', 'cat', 'cat']) # Mẫu 1, 2, 5 bị sai
    y_pred_model2_dummy = np.array(['dog', 'dog', 'cat', 'dog', 'cat', 'dog']) # Mẫu 0, 2 bị sai
    ids_dummy = [f'audio_{i}.wav' for i in range(len(y_true_dummy))] # Định danh mẫu dummy

    reporter = MisclassificationReporter(base_output_dir="dummy_reports_test")

    # Test generate_misclassified_csv
    print("\nTesting generate_misclassified_csv (Model 1)...")
    report1_path = reporter.generate_misclassified_csv(
        y_true=y_true_dummy,
        y_pred=y_pred_model1_dummy,
        sample_identifiers=ids_dummy,
        report_name="DummyModel1"
    )
    if report1_path:
         print(f"Generated misclassified report 1 at: {report1_path}")
         # Kiểm tra nội dung file (tùy chọn)
         df1 = pd.read_csv(report1_path)
         print("Content of report 1:")
         print(df1)
         assert len(df1) == 3 # 3 mẫu bị sai


    print("\nTesting generate_misclassified_csv (Model 2)...")
    report2_path = reporter.generate_misclassified_csv(
        y_true=y_true_dummy,
        y_pred=y_pred_model2_dummy,
        sample_identifiers=ids_dummy,
        report_name="DummyModel2"
    )
    if report2_path:
         print(f"Generated misclassified report 2 at: {report2_path}")
         df2 = pd.read_csv(report2_path)
         print("Content of report 2:")
         print(df2)
         assert len(df2) == 2 # 2 mẫu bị sai


    # Test generate_comparison_report
    print("\nTesting generate_comparison_report...")
    predictions_for_comparison = {
        "DummyModel1": y_pred_model1_dummy,
        "DummyModel2": y_pred_model2_dummy,
        # Thêm các dự đoán khác nếu có
    }

    comparison_report_path = reporter.generate_comparison_report(
        sample_identifiers=ids_dummy,
        y_true=y_true_dummy,
        predictions_dict=predictions_for_comparison,
        output_filename="dummy_comparison_all_samples.csv"
    )
    if comparison_report_path:
        print(f"Generated comparison report at: {comparison_report_path}")
        df_comp = pd.read_csv(comparison_report_path)
        print("Content of comparison report:")
        print(df_comp)
        assert len(df_comp) == len(y_true_dummy) # Báo cáo so sánh có tất cả các mẫu


    # Clean up dummy data (optional)
    # import shutil
    # if os.path.exists("dummy_reports_test"):
    #      shutil.rmtree("dummy_reports_test")

    print("\n--- Test lớp MisclassificationReporter hoàn thành ---")