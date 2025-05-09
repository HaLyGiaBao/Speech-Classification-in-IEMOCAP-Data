# Tên file: SingleModel.py

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
from scipy.stats import entropy # Để tính toán độ bất định

class SingleModel:
    """
    Một lớp bao bọc (wrapper) cho một mô hình phân loại đơn lẻ.
    Quản lý quá trình huấn luyện, dự đoán, lấy xác suất, tính độ bất định,
    và đánh giá mô hình.
    """

    def __init__(self, name, model, X_train, X_test, y_train, y_test, class_names):
        """
        Khởi tạo đối tượng SingleModel.

        Args:
            name (str): Tên mô tả cho mô hình này (ví dụ: "SVM RBF", "Random Forest").
            model: Đối tượng mô hình scikit-learn hoặc tương tự,
                   phải có các phương thức .fit(), .predict(), và .predict_proba().
            X_train (np.ndarray): Tập dữ liệu huấn luyện (đặc trưng).
            X_test (np.ndarray): Tập dữ liệu kiểm tra (đặc trưng).
            y_train (np.ndarray): Nhãn thật của tập huấn luyện.
            y_test (np.ndarray): Nhãn thật của tập kiểm tra.
            class_names (list hoặc np.ndarray): Danh sách các tên lớp (string).
        """
        if not hasattr(model, 'fit') or not hasattr(model, 'predict'):
             raise TypeError("Đối tượng 'model' phải có phương thức .fit() và .predict().")
        # Kiểm tra predict_proba riêng vì nó là tùy chọn nhưng cần cho ensemble/uncertainty
        if not hasattr(model, 'predict_proba'):
             print(f"Warning: Mô hình '{name}' không có phương thức .predict_proba(). Sẽ không thể tính xác suất và độ bất định.")


        self.name = name
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.class_names = class_names # Lưu tên lớp để sử dụng khi báo cáo/đánh giá

        # Các thuộc tính sẽ được populate sau khi chạy analyze()
        self.y_pred = None
        self.y_proba = None
        self.uncertainty = None # Độ bất định cho từng mẫu trong X_test


        print(f"Đã khởi tạo SingleModel: {self.name}")


    def train(self):
        """Huấn luyện mô hình trên dữ liệu huấn luyện."""
        print(f"Đang huấn luyện mô hình: {self.name}...")
        self.model.fit(self.X_train, self.y_train)
        print(f"Hoàn thành huấn luyện mô hình: {self.name}.")

    def predict(self):
        """Thực hiện dự đoán trên tập dữ liệu kiểm tra."""
        if self.model is None:
            print(f"Lỗi: Mô hình '{self.name}' chưa được khởi tạo.")
            return None
        if self.X_test is None:
            print(f"Lỗi: Không có dữ liệu kiểm tra cho mô hình '{self.name}'.")
            return None

        print(f"Đang dự đoán với mô hình: {self.name}...")
        self.y_pred = self.model.predict(self.X_test)
        print(f"Hoàn thành dự đoán với mô hình: {self.name}.")
        return self.y_pred

    def predict_proba(self):
        """Lấy xác suất dự đoán cho từng lớp trên tập dữ liệu kiểm tra."""
        if self.model is None:
             print(f"Lỗi: Mô hình '{self.name}' chưa được khởi tạo.")
             return None
        if not hasattr(self.model, 'predict_proba'):
            # Thông báo đã được đưa ra khi khởi tạo
            return None
        if self.X_test is None:
            print(f"Lỗi: Không có dữ liệu kiểm tra cho mô hình '{self.name}'.")
            return None


        print(f"Đang lấy xác suất dự đoán với mô hình: {self.name}...")
        self.y_proba = self.model.predict_proba(self.X_test)
        print(f"Hoàn thành lấy xác suất dự đoán với mô hình: {self.name}.")
        return self.y_proba


    def calculate_uncertainty(self):
        """
        Tính toán độ bất định (sử dụng Entropy) từ xác suất dự đoán.
        Yêu cầu phương thức predict_proba() phải được chạy trước.
        """
        if self.y_proba is None:
            print(f"Lỗi: Không có xác suất dự đoán cho mô hình '{self.name}'. Vui lòng chạy predict_proba() trước.")
            self.uncertainty = None
            return None

        # Tính entropy cho từng mẫu. Entropy là -sum(p * log2(p)).
        # scipy.stats.entropy với base=2 tính entropy theo bit.
        # Thêm một epsilon nhỏ để tránh log(0) nếu có xác suất bằng 0 tuyệt đối.
        epsilon = 1e-9
        probabilities = self.y_proba + epsilon
        # Chuẩn hóa lại sau khi thêm epsilon (đảm bảo tổng xác suất = 1)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)


        # Tính entropy cho mỗi hàng (mỗi mẫu)
        self.uncertainty = entropy(probabilities, base=2, axis=1)

        print(f"Đã tính toán độ bất định cho mô hình: {self.name}.")
        return self.uncertainty


    def evaluate(self):
        """
        Đánh giá hiệu suất của mô hình trên tập kiểm tra.
        Yêu cầu phương thức predict() phải được chạy trước.
        In ra báo cáo phân loại, balanced accuracy và accuracy.
        Trả về một dictionary chứa các chỉ số đánh giá.
        """
        if self.y_pred is None:
            print(f"Lỗi: Không có dự đoán cho mô hình '{self.name}'. Vui lòng chạy predict() trước.")
            return None
        if self.y_test is None:
             print(f"Lỗi: Không có nhãn thật cho dữ liệu kiểm tra.")
             return None
        if self.class_names is None or len(self.class_names) == 0:
             print(f"Lỗi: Không có danh sách tên lớp (class_names) để đánh giá.")
             # Thử suy luận tên lớp từ dữ liệu nếu không được cung cấp
             # unique_labels = np.unique(np.concatenate((self.y_test, self.y_pred)))
             # print(f"Sử dụng các nhãn duy nhất từ dữ liệu: {unique_labels}")
             # target_names_for_report = unique_labels
             # Tốt hơn là yêu cầu người dùng cung cấp class_names khi khởi tạo
             return None


        print(f"\n--- Báo cáo Đánh giá cho Mô hình: {self.name} ---")

        # Báo cáo phân loại (precision, recall, f1-score theo từng lớp)
        # target_names cần khớp thứ tự hoặc giá trị duy nhất của nhãn thật/dự đoán
        try:
            # Sử dụng output_dict=True để dễ dàng lấy các chỉ số riêng lẻ
            report_dict = classification_report(self.y_test, self.y_pred, target_names=self.class_names, output_dict=True)
            print(classification_report(self.y_test, self.y_pred, target_names=self.class_names))
        except Exception as e:
             print(f"Lỗi khi tạo báo cáo phân loại cho '{self.name}': {e}. Đảm bảo class_names khớp với nhãn trong y_test/y_pred.")
             report_dict = {} # Trả về dict rỗng nếu có lỗi

        # Balanced Accuracy (Unweighted Accuracy)
        # Tính trung bình độ chính xác trên từng lớp
        try:
            bal_acc = balanced_accuracy_score(self.y_test, self.y_pred)
            print(f"Balanced Accuracy (Unweighted): {bal_acc:.4f}")
            report_dict['balanced_accuracy'] = bal_acc
        except Exception as e:
             print(f"Lỗi khi tính Balanced Accuracy cho '{self.name}': {e}")
             report_dict['balanced_accuracy'] = None


        # Accuracy (Weighted Accuracy)
        # Độ chính xác tổng thể
        try:
            acc = accuracy_score(self.y_test, self.y_pred)
            print(f"Accuracy (Weighted): {acc:.4f}")
            report_dict['accuracy'] = acc
        except Exception as e:
             print(f"Lỗi khi tính Accuracy cho '{self.name}': {e}")
             report_dict['accuracy'] = None

        # Ma trận nhầm lẫn (Confusion Matrix)
        # Để có ma trận nhầm lẫn có thứ tự lớp nhất quán, nên truyền 'labels=self.class_names'
        try:
            cm = confusion_matrix(self.y_test, self.y_pred, labels=self.class_names)
            print("\nMa trận nhầm lẫn:")
            print(cm)
            report_dict['confusion_matrix'] = cm
        except Exception as e:
             print(f"Lỗi khi tính ma trận nhầm lẫn cho '{self.name}': {e}")
             report_dict['confusion_matrix'] = None


        print("----------------------------------------------------")
        return report_dict

    def run_pipeline(self):
        """
        Chạy toàn bộ quy trình: huấn luyện, dự đoán, lấy xác suất, tính độ bất định.
        """
        print(f"\n--- Bắt đầu pipeline cho mô hình: {self.name} ---")
        self.train()
        self.predict()
        # Kiểm tra predict_proba trước khi gọi
        if hasattr(self.model, 'predict_proba'):
             self.predict_proba()
             self.calculate_uncertainty()
        else:
             print(f"Bỏ qua lấy xác suất và tính độ bất định cho '{self.name}' vì mô hình không có predict_proba.")

        print(f"--- Hoàn thành pipeline cho mô hình: {self.name} ---")


    # --- Các phương thức để truy cập kết quả sau khi chạy pipeline ---
    def get_name(self):
        return self.name

    def get_predictions(self):
        return self.y_pred

    def get_probabilities(self):
        return self.y_proba

    def get_uncertainty(self):
        return self.uncertainty

    def get_true_labels(self):
        return self.y_test

    def get_model(self):
        # Cần cẩn thận khi trả về đối tượng model, không nên thay đổi nó từ bên ngoài
        return self.model

# --- Cách sử dụng (ví dụ đơn giản trong cùng file) ---
if __name__ == "__main__":
    print("\n--- Test lớp SingleModel ---")

    # Tạo dữ liệu dummy (giả lập) để test
    X_dummy = np.random.rand(100, 10) # 100 mẫu, 10 đặc trưng
    y_dummy = np.random.choice(['cat', 'dog', 'bird'], size=100) # 3 lớp

    # Chia dữ liệu
    X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = train_test_split(X_dummy, y_dummy, test_size=0.3, random_state=42)

    # Tên các lớp
    class_names_dummy = ['cat', 'dog', 'bird']

    # Tạo một mô hình dummy (ví dụ: Decision Tree)
    from sklearn.tree import DecisionTreeClassifier
    # Decision Tree có predict_proba
    dummy_model = DecisionTreeClassifier(random_state=42)

    # Khởi tạo SingleModel
    single_dt_model = SingleModel(
        name="Dummy Decision Tree",
        model=dummy_model,
        X_train=X_train_dummy,
        X_test=X_test_dummy,
        y_train=y_train_dummy,
        y_test=y_test_dummy,
        class_names=class_names_dummy
    )

    # Chạy pipeline
    single_dt_model.run_pipeline()

    # Lấy kết quả và đánh giá
    predictions = single_dt_model.get_predictions()
    probabilities = single_dt_model.get_probabilities()
    uncertainty = single_dt_model.get_uncertainty()

    print("\nKết quả từ SingleModel:")
    print("Số lượng dự đoán:", len(predictions) if predictions is not None else "N/A")
    print("Shape xác suất:", probabilities.shape if probabilities is not None else "N/A")
    print("Shape độ bất định:", uncertainty.shape if uncertainty is not None else "N/A")

    # Đánh giá mô hình đơn lẻ này
    evaluation_metrics = single_dt_model.evaluate()
    print("\nMetrics:", evaluation_metrics)

    # Test với mô hình không có predict_proba (ví dụ: LinearSVC)
    from sklearn.svm import LinearSVC
    # LinearSVC KHÔNG có predict_proba theo mặc định
    dummy_model_no_proba = LinearSVC(random_state=42)

    single_svc_no_proba = SingleModel(
         name="Dummy Linear SVC (No Proba)",
         model=dummy_model_no_proba,
         X_train=X_train_dummy,
         X_test=X_test_dummy,
         y_train=y_train_dummy,
         y_test=y_test_dummy,
         class_names=class_names_dummy
    )
    single_svc_no_proba.run_pipeline() # Sẽ cảnh báo và bỏ qua predict_proba/uncertainty
    single_svc_no_proba.evaluate() # Vẫn có thể đánh giá dựa trên predict()