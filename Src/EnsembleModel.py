# Tên file: EnsembleModel.py

import numpy as np
from scipy.stats import entropy # Để tính toán độ bất định (đã dùng trong SingleModel)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
import collections # Có thể hữu ích cho voting nếu cần, nhưng dùng numpy/proba hiệu quả hơn

# Cần import lớp SingleModel để có thể sử dụng các đối tượng của nó
from SingleModel import SingleModel

class EnsembleModel:
    """
    Một lớp để kết hợp dự đoán và thông tin độ bất định
    từ nhiều mô hình đơn lẻ (đối tượng SingleModel).
    """

    def __init__(self, models, class_names, y_true):
        """
        Khởi tạo đối tượng EnsembleModel từ danh sách các đối tượng SingleModel đã train.

        Args:
            models (list): Danh sách các đối tượng SingleModel đã được huấn luyện
                           và chạy pipeline (đã có predictions, probabilities, uncertainty).
                           Mỗi mô hình phải có phương thức get_predictions(), get_probabilities(), get_uncertainty().
                           Các mô hình cần có predict_proba=True nếu dùng chiến lược dựa trên xác suất/độ bất định.
            class_names (list hoặc np.ndarray): Danh sách các tên lớp (string).
                                              Phải nhất quán với các mô hình đơn lẻ và y_true.
            y_true (np.ndarray): Nhãn thật của tập dữ liệu kiểm tra.
        """
        # Kiểm tra đầu vào
        if not isinstance(models, list) or not models:
            raise ValueError("Đối số 'models' phải là một danh sách không rỗng các đối tượng SingleModel.")
        if not all(isinstance(m, SingleModel) for m in models):
             raise TypeError("Tất cả các phần tử trong danh sách 'models' phải là đối tượng SingleModel.")

        self.models = models
        self.class_names = class_names
        self.y_true = y_true

        # Kiểm tra tính nhất quán của shape dữ liệu kiểm tra giữa các mô hình
        if models:
             first_model_test_samples = models[0].X_test.shape[0]
             if not all(m.X_test.shape[0] == first_model_test_samples for m in models):
                  print("Warning: Số lượng mẫu trong tập kiểm tra (X_test.shape[0]) không nhất quán giữa các mô hình. Kết quả ensemble có thể không chính xác.")
             if len(y_true) != first_model_test_samples:
                  print(f"Warning: Số lượng nhãn thật ({len(y_true)}) không khớp với số lượng mẫu trong tập kiểm tra của mô hình ({first_model_test_samples}). Đánh giá ensemble có thể không chính xác.")


        # Thu thập kết quả từ tất cả các mô hình đơn lẻ đã được chạy pipeline
        # Lọc ra các mô hình không có dữ liệu xác suất hoặc độ bất định nếu cần
        self.all_y_pred = np.array([m.get_predictions() for m in self.models]) # Shape: [n_models, n_samples]

        # Lấy xác suất và độ bất định chỉ từ các mô hình CÓ chúng
        self._proba_models = [m for m in self.models if m.get_probabilities() is not None]
        if self._proba_models:
             # self.all_y_proba = np.array([m.get_probabilities() for m in self._proba_models]) # Shape: [n_valid_models_proba, n_samples, n_classes]
             # Lưu dưới dạng list các array numpy thay vì 3D array nếu có nguy cơ shape n_classes khác nhau (ít xảy ra với cùng task)
             # Hoặc đảm bảo shape n_classes giống nhau. Giả định n_classes giống nhau.
             self.all_y_proba = np.array([m.get_probabilities() for m in self._proba_models])

             # Lập index mapping từ mô hình ban đầu sang mô hình có proba
             self._proba_model_indices = [i for i, m in enumerate(self.models) if m.get_probabilities() is not None]
             # Kiểm tra tính nhất quán của shape n_classes
             if self.all_y_proba.shape[0] > 0:
                  first_n_classes = self.all_y_proba.shape[2]
                  if not all(p.shape[1] == first_n_classes for p in [m.get_probabilities() for m in self._proba_models]): # Check shape[1] of original proba arrays
                       print("Warning: Số lượng lớp (n_classes) không nhất quán trong dữ liệu xác suất giữa các mô hình có proba.")
                       # Điều này sẽ gây lỗi khi np.array([m.get_probabilities() ...])
                       # Xử lý an toàn hơn là không tạo self.all_y_proba dưới dạng numpy array nếu shape khác nhau.
                       # Hoặc buộc n_classes phải giống nhau. Ta buộc n_classes phải giống nhau.


        else:
             self.all_y_proba = None
             self._proba_model_indices = []
             print("Warning: Không có mô hình nào cung cấp dữ liệu xác suất dự đoán.")


        self._uncertainty_models = [m for m in self.models if m.get_uncertainty() is not None]
        if self._uncertainty_models:
            # self.all_uncertainty = np.array([m.get_uncertainty() for m in self._uncertainty_models]) # Shape: [n_valid_models_unc, n_samples]
            # Lưu dưới dạng numpy array
            self.all_uncertainty = np.array([m.get_uncertainty() for m in self._uncertainty_models])
            self._uncertainty_model_indices = [i for i, m in enumerate(self.models) if m.get_uncertainty() is not None]
             # Kiểm tra tính nhất quán của shape n_samples
            if self.all_uncertainty.shape[0] > 0:
                 first_n_samples_unc = self.all_uncertainty.shape[1]
                 if not all(u.shape[0] == first_n_samples_unc for u in [m.get_uncertainty() for m in self._uncertainty_models]): # Check shape[0] of original uncertainty arrays
                      print("Warning: Số lượng mẫu (n_samples) không nhất quán trong dữ liệu độ bất định giữa các mô hình có uncertainty.")
                      # Điều này sẽ gây lỗi khi np.array([...])
                      # Xử lý an toàn hơn: không tạo self.all_uncertainty dưới dạng numpy array nếu shape khác nhau.
                      # Hoặc buộc n_samples phải giống nhau. Ta buộc n_samples phải giống nhau.

        else:
             self.all_uncertainty = None
             self._uncertainty_model_indices = []
             print("Warning: Không có mô hình nào cung cấp dữ liệu độ bất định.")


        # Mapping từ tên lớp sang index và ngược lại
        self.label_to_index = {name: i for i, name in enumerate(self.class_names)}
        self.index_to_label = {i: name for name, i in self.label_to_index.items()}
        # Kiểm tra tính nhất quán giữa số lớp từ class_names và n_classes từ proba
        if self.all_y_proba is not None and self.all_y_proba.shape[2] != len(self.class_names):
             print(f"Warning: Số lượng lớp trong class_names ({len(self.class_names)}) không khớp với số cột xác suất từ mô hình ({self.all_y_proba.shape[2]}). Việc ánh xạ nhãn có thể sai.")


        print(f"Đã khởi tạo EnsembleModel với {len(self.models)} mô hình cơ sở.")
        if self.all_y_proba is not None:
             print(f"  Có {self.all_y_proba.shape[0]} mô hình cung cấp dữ liệu xác suất.")
        if self.all_uncertainty is not None:
             print(f"  Có {self.all_uncertainty.shape[0]} mô hình cung cấp dữ liệu độ bất định.")


    def _get_weighted_probabilities(self, weights):
        """
        Phương thức trợ giúp để tính xác suất trung bình có trọng số.
        Trung bình hóa trên các mô hình CÓ dữ liệu xác suất, sử dụng trọng số tương ứng.

        Args:
            weights (np.ndarray): Mảng trọng số TƯƠNG ỨNG VỚI CÁC MÔ HÌNH CÓ PROBA.
                                  Shape: [n_valid_models_proba, n_samples].
        Returns:
            np.ndarray: Xác suất trung bình có trọng số, shape [n_samples, n_classes].
                        Trả về None nếu không có dữ liệu xác suất.
        """
        if self.all_y_proba is None:
             print("Lỗi: Không có dữ liệu xác suất để tính trung bình có trọng số.")
             return None

        # Trọng số phải có cùng số lượng mô hình với self.all_y_proba
        if weights.shape[0] != self.all_y_proba.shape[0]:
             print(f"Lỗi: Số lượng mô hình trong mảng trọng số ({weights.shape[0]}) không khớp với số lượng mô hình có xác suất ({self.all_y_proba.shape[0]}).")
             return None

        # Đảm bảo trọng số hợp lệ (>= 0)
        weights[weights < 0] = 0

        # Tính tổng trọng số cho mỗi mẫu
        sum_weights = weights.sum(axis=0, keepdims=True) # Shape [1, n_samples]

        # Tránh chia cho 0 nếu tổng trọng số là 0 cho một mẫu nào đó
        sum_weights[sum_weights == 0] = 1e-9 # Thay thế 0 bằng giá trị nhỏ để tránh lỗi


        # Chuẩn hóa trọng số trên mỗi mẫu
        normalized_weights = weights / sum_weights # Shape [n_valid_models_proba, n_samples]

        # Mở rộng kích thước trọng số để phù hợp với shape của xác suất
        # normalized_weights shape: [n_valid_models_proba, n_samples]
        # self.all_y_proba shape: [n_valid_models_proba, n_samples, n_classes]
        # Mở rộng normalized_weights thành [n_valid_models_proba, n_samples, 1]
        weighted_proba = self.all_y_proba * normalized_weights[:, :, np.newaxis]

        # Tính tổng xác suất có trọng số trên các mô hình (chiều 0)
        avg_proba = weighted_proba.sum(axis=0) # Shape [n_samples, n_classes]

        return avg_proba

    def _predict_from_probabilities(self, probabilities):
         """Phương thức trợ giúp để chuyển mảng xác suất thành nhãn dự đoán."""
         if probabilities is None:
              return None
         # np.argmax tìm index của giá trị lớn nhất trên chiều lớp (chiều 1)
         predicted_indices = np.argmax(probabilities, axis=1)
         # Ánh xạ các index trở lại tên lớp
         predicted_labels = np.array([self.index_to_label[idx] for idx in predicted_indices])
         return predicted_labels


    # --- Các Chiến lược Kết hợp (Ensemble Strategies) ---

    def predict_ul(self):
        """
        Chiến lược Uncertainty Lowest (UL):
        Đối với mỗi mẫu, chọn dự đoán của mô hình CÓ dữ liệu độ bất định và có độ bất định thấp nhất.
        """
        print("Đang thực hiện chiến lược Ensemble: Uncertainty Lowest (UL)")
        if self.all_uncertainty is None or self.all_uncertainty.shape[0] == 0:
             print("Lỗi: Không có dữ liệu độ bất định từ các mô hình để dùng chiến lược UL.")
             return None
        # Cần ensure self.all_y_pred có cùng số lượng mẫu với self.all_uncertainty
        if self.all_y_pred.shape[1] != self.all_uncertainty.shape[1]:
             print("Lỗi: Số lượng mẫu trong dự đoán và độ bất định không khớp.")
             return None

        # Tìm index của mô hình CÓ độ bất định và có độ bất định nhỏ nhất cho từng mẫu
        min_uncertainty_model_relative_indices = np.argmin(self.all_uncertainty, axis=0) # Shape [n_samples], index tương ứng với all_uncertainty

        # Lấy dự đoán từ mô hình TƯƠNG ỨNG TRONG DANH SÁCH MODELS BAN ĐẦU
        # Index trong all_uncertainty -> index trong self.models
        min_uncertainty_model_original_indices = np.array([self._uncertainty_model_indices[i] for i in min_uncertainty_model_relative_indices]) # Shape [n_samples]

        # Lấy dự đoán từ mô hình tương ứng trong all_y_pred (đã có tất cả các mô hình)
        n_samples = self.all_y_pred.shape[1]
        ensemble_predictions = np.array([
            # all_y_pred[index_mo_hinh_tot_nhat_cho_mau_i (trong danh sách gốc), index_mau_i]
            self.all_y_pred[min_uncertainty_model_original_indices[i], i]
            for i in range(n_samples)
        ])
        return ensemble_predictions

    def predict_ut(self, threshold=None):
        """
        Chiến lược Uncertainty Threshold (UT):
        Đối với mỗi mẫu:
        - Nếu độ bất định thấp nhất (trong các mô hình CÓ uncertainty) < ngưỡng, dùng dự đoán của mô hình đó (như UL).
        - Ngược lại, dùng trung bình xác suất (Mean Probability Voting) của TẤT CẢ các mô hình CÓ xác suất.

        Args:
            threshold (float): Ngưỡng độ bất định. Phải được cung cấp.
        Returns:
            np.ndarray: Dự đoán ensemble, shape [n_samples]. Trả về None nếu lỗi hoặc thiếu dữ liệu.
        """
        print(f"Đang thực hiện chiến lược Ensemble: Uncertainty Threshold (UT) với ngưỡng = {threshold}")
        if self.all_uncertainty is None or self.all_uncertainty.shape[0] == 0:
             print("Lỗi: Không có dữ liệu độ bất định để dùng chiến lược UT.")
             return None
        if self.all_y_proba is None or self.all_y_proba.shape[0] == 0:
             print("Lỗi: Không có dữ liệu xác suất để dùng chiến lược UT (cần cho Average Voting).")
             return None
        # Cần ensure số lượng mẫu nhất quán giữa pred, unc, proba
        if not (self.all_y_pred.shape[1] == self.all_uncertainty.shape[1] == self.all_y_proba.shape[1]):
             print("Lỗi: Số lượng mẫu trong dự đoán, độ bất định, xác suất không khớp giữa các dữ liệu.")
             return None

        if threshold is None:
             print("Lỗi: Vui lòng cung cấp giá trị ngưỡng (threshold) cho chiến lược UT.")
             return None


        n_samples = self.all_y_pred.shape[1]
        ensemble_predictions = np.empty(n_samples, dtype=self.all_y_pred.dtype) # Khởi tạo mảng kết quả

        # Tìm index mô hình có độ bất định nhỏ nhất (trong các mô hình CÓ uncertainty)
        min_uncertainty_model_relative_indices = np.argmin(self.all_uncertainty, axis=0) # Index tương ứng với all_uncertainty
        min_uncertainties = np.min(self.all_uncertainty, axis=0) # Giá trị độ bất định thấp nhất

        # Lập index mapping từ index tương đối trong all_uncertainty sang index trong self.models
        min_uncertainty_model_original_indices = np.array([self._uncertainty_model_indices[i] for i in min_uncertainty_model_relative_indices])


        # Tạo mask cho các mẫu có độ bất định thấp nhất < ngưỡng
        below_threshold_mask = min_uncertainties < threshold

        # Đối với các mẫu DƯỚI ngưỡng: Lấy dự đoán từ mô hình có độ bất định thấp nhất
        # Lấy index của các mẫu dưới ngưỡng
        indices_below = np.where(below_threshold_mask)[0]
        if len(indices_below) > 0:
             # Lấy index mô hình gốc cho các mẫu này
             original_indices_for_below = min_uncertainty_model_original_indices[below_threshold_mask]
             # Lấy dự đoán từ all_y_pred bằng cách sử dụng fancy indexing
             ensemble_predictions[below_threshold_mask] = self.all_y_pred[original_indices_for_below, indices_below]


        # Đối với các mẫu TRÊN hoặc BẰNG ngưỡng: Dùng Mean Probability Voting
        above_threshold_mask = ~below_threshold_mask
        if np.any(above_threshold_mask):
             # Chỉ lấy dữ liệu xác suất cho các mẫu trên ngưỡng (across models with proba)
             # all_y_proba shape: [n_valid_models_proba, n_samples, n_classes]
             proba_above_threshold = self.all_y_proba[:, above_threshold_mask, :] # Shape [n_valid_models_proba, n_samples_above, n_classes]

             # Trung bình hóa xác suất trên các mô hình CÓ xác suất
             avg_proba_above_threshold = np.mean(proba_above_threshold, axis=0) # Shape [n_samples_above, n_classes]

             # Dự đoán từ xác suất trung bình
             predictions_above_threshold = self._predict_from_probabilities(avg_proba_above_threshold)
             # Gán kết quả dự đoán vào các vị trí tương ứng
             ensemble_predictions[above_threshold_mask] = predictions_above_threshold

        return ensemble_predictions


    def predict_uw(self):
        """
        Chiến lược Uncertainty Weighted (UW):
        Kết hợp dự đoán sử dụng nghịch đảo độ bất định (1 / uncertainty) làm trọng số
        cho việc trung bình hóa xác suất.
        Chỉ sử dụng các mô hình CÓ dữ liệu xác suất và độ bất định, đảm bảo chúng khớp nhau.
        """
        print("Đang thực hiện chiến lược Ensemble: Uncertainty Weighted (UW)")
        # Lọc chỉ lấy các mô hình có cả xác suất VÀ độ bất định
        valid_models_for_weighted = [m for m in self.models if m.get_probabilities() is not None and m.get_uncertainty() is not None]

        if not valid_models_for_weighted:
             print("Lỗi: Không có mô hình nào cung cấp cả dữ liệu xác suất và độ bất định để dùng chiến lược UW.")
             return None

        # Lấy dữ liệu xác suất và độ bất định chỉ từ các mô hình hợp lệ này
        proba_data = np.array([m.get_probabilities() for m in valid_models_for_weighted]) # Shape [n_valid, n_samples, n_classes]
        uncertainty_data = np.array([m.get_uncertainty() for m in valid_models_for_weighted]) # Shape [n_valid, n_samples]

        # Kiểm tra tính nhất quán shape lần nữa
        if proba_data.shape[0] != uncertainty_data.shape[0] or proba_data.shape[1] != uncertainty_data.shape[1]:
             print("Lỗi: Dữ liệu xác suất và độ bất định không khớp shape sau khi lọc các mô hình hợp lệ cho UW.")
             return None


        # Tính nghịch đảo độ bất định làm trọng số. Thêm epsilon nhỏ để tránh chia cho 0 nếu uncertainty = 0.
        epsilon = 1e-9
        inverse_uncertainty_weights = 1.0 / (uncertainty_data + epsilon) # Shape [n_valid, n_samples]

        # Tính xác suất trung bình có trọng số
        # Sử dụng _get_weighted_probabilities helper. Helper này mong đợi weights tương ứng với proba.
        avg_proba = self._get_weighted_probabilities(inverse_uncertainty_weights)

        # Dự đoán từ xác suất trung bình có trọng số
        ensemble_predictions = self._predict_from_probabilities(avg_proba)

        return ensemble_predictions


    def predict_cw(self):
        """
        Chiến lược Confidence Weighted (CW):
        Kết hợp dự đoán sử dụng độ tin cậy (Confidence) làm trọng số
        cho việc trung bình hóa xác suất. Độ tin cậy được tính là 1 - độ bất định.
        Chỉ sử dụng các mô hình CÓ dữ liệu xác suất VÀ độ bất định.
        """
        print("Đang thực hiện chiến lược Ensemble: Confidence Weighted (CW)")
        # Lọc chỉ lấy các mô hình có cả xác suất VÀ độ bất định
        valid_models_for_weighted = [m for m in self.models if m.get_probabilities() is not None and m.get_uncertainty() is not None]

        if not valid_models_for_weighted:
             print("Lỗi: Không có mô hình nào cung cấp cả dữ liệu xác suất và độ bất định để dùng chiến lược CW.")
             return None

        # Lấy dữ liệu xác suất và độ bất định chỉ từ các mô hình hợp lệ này
        proba_data = np.array([m.get_probabilities() for m in valid_models_for_weighted]) # Shape [n_valid, n_samples, n_classes]
        uncertainty_data = np.array([m.get_uncertainty() for m in valid_models_for_weighted]) # Shape [n_valid, n_samples]

         # Kiểm tra tính nhất quán shape lần nữa
        if proba_data.shape[0] != uncertainty_data.shape[0] or proba_data.shape[1] != uncertainty_data.shape[1]:
             print("Lỗi: Dữ liệu xác suất và độ bất định không khớp shape sau khi lọc các mô hình hợp lệ cho CW.")
             return None


        # Tính trọng số dựa trên độ tin cậy (1 - Uncertainty)
        confidence_weights = 1.0 - uncertainty_data # Shape [n_valid, n_samples]

        # Đảm bảo trọng số không âm (entropy >= 0, max entropy = log2(n_classes), 1-entropy có thể < 0 nếu entropy > 1)
        # Nếu entropy được tính đúng theo base 2 và n_classes > 2, entropy có thể > 1.
        # Tốt nhất là chuẩn hóa uncertainty về [0, 1] trước khi tính confidence, hoặc đảm bảo confidence >= 0.
        # Tạm thời, đảm bảo confidence >= 0
        confidence_weights[confidence_weights < 0] = 0


        # Tính xác suất trung bình có trọng số
        # Sử dụng _get_weighted_probabilities helper. Helper này mong đợi weights tương ứng với proba.
        avg_proba = self._get_weighted_probabilities(confidence_weights)

        # Dự đoán từ xác suất trung bình có trọng số
        ensemble_predictions = self._predict_from_probabilities(avg_proba)

        return ensemble_predictions

    # --- Thêm hai chiến lược dựa trên xác suất đơn giản ---

    def predict_mean_proba_voting(self):
        """
        Chiến lược Mean Probability Voting:
        Kết hợp dự đoán bằng cách trung bình cộng xác suất từ TẤT CẢ các mô hình CÓ dữ liệu xác suất.
        Chọn lớp có xác suất trung bình cao nhất.
        """
        print("Đang thực hiện chiến lược Ensemble: Mean Probability Voting")
        if self.all_y_proba is None or self.all_y_proba.shape[0] == 0:
             print("Lỗi: Không có dữ liệu xác suất từ các mô hình để dùng chiến lược Mean Probability Voting.")
             return None

        # Average probabilities across the first dimension (models)
        # all_y_proba shape: [n_valid_models_proba, n_samples, n_classes]
        avg_proba = np.mean(self.all_y_proba, axis=0) # Shape: [n_samples, n_classes]

        # Dự đoán từ xác suất trung bình
        ensemble_predictions = self._predict_from_probabilities(avg_proba)

        return ensemble_predictions


    def predict_max_proba_voting(self):
        """
        Chiến lược Max Probability Voting:
        Đối với mỗi mẫu và mỗi lớp, lấy xác suất CAO NHẤT mà bất kỳ mô hình CÓ dữ liệu xác suất nào dự đoán.
        Chọn lớp có xác suất tối đa này cao nhất.
        """
        print("Đang thực hiện chiến lược Ensemble: Max Probability Voting")
        if self.all_y_proba is None or self.all_y_proba.shape[0] == 0:
             print("Lỗi: Không có dữ liệu xác suất từ các mô hình để dùng chiến lược Max Probability Voting.")
             return None

        # Lấy xác suất lớn nhất cho mỗi lớp trên các mô hình (chiều 0)
        # all_y_proba shape: [n_valid_models_proba, n_samples, n_classes]
        max_proba = np.max(self.all_y_proba, axis=0) # Shape: [n_samples, n_classes]

        # Dự đoán từ xác suất tối đa
        ensemble_predictions = self._predict_from_probabilities(max_proba)

        return ensemble_predictions


    def evaluate_ensemble(self, y_pred_ensemble, strategy_name="Kết quả Ensemble"):
        """
        Đánh giá hiệu suất của kết quả ensemble trên tập kiểm tra.

        Args:
            y_pred_ensemble (np.ndarray): Dự đoán kết quả từ một chiến lược ensemble.
            strategy_name (str): Tên của chiến lược ensemble (để in báo cáo).
        Returns:
            dict: Dictionary chứa các chỉ số đánh giá.
        """
        if y_pred_ensemble is None:
            print(f"Lỗi: Không có dự đoán ensemble cho chiến lược '{strategy_name}'.")
            return {}
        if self.y_true is None:
            print(f"Lỗi: Không có nhãn thật (y_true) để đánh giá ensemble.")
            return {}
        if self.class_names is None or len(self.class_names) == 0:
             print(f"Lỗi: Không có danh sách tên lớp (class_names) để đánh giá ensemble.")
             return {}
        if len(y_pred_ensemble) != len(self.y_true):
            print(f"Lỗi: Số lượng dự đoán ensemble ({len(y_pred_ensemble)}) không khớp với số lượng nhãn thật ({len(self.y_true)}) cho chiến lược '{strategy_name}'.")
            return {}


        print(f"\n--- Báo cáo Đánh giá cho {strategy_name} Ensemble ---")

        # Báo cáo phân loại
        try:
            report_dict = classification_report(self.y_true, y_pred_ensemble, target_names=self.class_names, output_dict=True, zero_division=0) # zero_division=0 để tránh cảnh báo chia cho 0
            print(classification_report(self.y_true, y_pred_ensemble, target_names=self.class_names, zero_division=0))
        except Exception as e:
             print(f"Lỗi khi tạo báo cáo phân loại cho '{strategy_name}': {e}. Đảm bảo class_names khớp với nhãn trong y_true/y_pred_ensemble.")
             report_dict = {}

        # Balanced Accuracy
        try:
            bal_acc = balanced_accuracy_score(self.y_true, y_pred_ensemble)
            print(f"Balanced Accuracy (Unweighted): {bal_acc:.4f}")
            report_dict['balanced_accuracy'] = bal_acc
        except Exception as e:
             print(f"Lỗi khi tính Balanced Accuracy cho '{strategy_name}': {e}")
             report_dict['balanced_accuracy'] = None


        # Accuracy
        try:
            acc = accuracy_score(self.y_true, y_pred_ensemble)
            print(f"Accuracy (Weighted): {acc:.4f}")
            report_dict['accuracy'] = acc
        except Exception as e:
             print(f"Lỗi khi tính Accuracy cho '{strategy_name}': {e}")
             report_dict['accuracy'] = None

        # Ma trận nhầm lẫn
        try:
            # labels=self.class_names để đảm bảo thứ tự và bao gồm cả các lớp không được dự đoán
            cm = confusion_matrix(self.y_true, y_pred_ensemble, labels=self.class_names)
            print("\nMa trận nhầm lẫn:")
            print(cm)
            report_dict['confusion_matrix'] = cm
        except Exception as e:
             print(f"Lỗi khi tính ma trận nhầm lẫn cho '{strategy_name}': {e}")
             report_dict['confusion_matrix'] = None


        print("----------------------------------------------------")
        return report_dict


    # --- Các phương thức để lấy dữ liệu đã thu thập ---
    # Các phương thức này trả về dữ liệu từ các mô hình đơn lẻ ĐÃ CÓ xác suất/độ bất định.
    def get_all_predictions(self):
        # Trả về dự đoán từ TẤT CẢ các mô hình ban đầu
        return self.all_y_pred

    def get_all_probabilities(self):
        # Trả về xác suất chỉ từ các mô hình CÓ xác suất
        return self.all_y_proba

    def get_all_uncertainty(self):
        # Trả về độ bất định chỉ từ các mô hình CÓ độ bất định
        return self.all_uncertainty

    def get_true_labels(self):
        return self.y_true

    def get_proba_model_indices(self):
        # Trả về index gốc của các mô hình có xác suất
        return self._proba_model_indices

    def get_uncertainty_model_indices(self):
        # Trả về index gốc của các mô hình có độ bất định
        return self._uncertainty_model_indices


# --- Cách sử dụng (ví dụ đơn giản trong cùng file) ---
# (Giữ nguyên như trước để test lớp EnsembleModel với dummy data)
if __name__ == "__main__":
    print("\n--- Test lớp EnsembleModel ---")

    # Tạo dữ liệu dummy
    X_dummy = np.random.rand(100, 10)
    y_dummy = np.random.choice(['class_A', 'class_B'], size=100)

    X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = train_test_split(X_dummy, y_dummy, test_size=0.5, random_state=42)

    class_names_dummy = ['class_A', 'class_B']

    # Tạo các mô hình đơn lẻ dummy
    # Mô hình 1: Decision Tree (có predict_proba)
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import entropy # Cần cho dummy SingleModel

    # Dummy SingleModel (cần predict_proba và calculate_uncertainty)
    class DummySingleModelWithProba:
         def __init__(self, name, model, X_train, X_test, y_train, y_test, class_names):
             self.name = name
             self.model = model
             self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
             self.class_names = class_names
             self.y_pred = None
             self.y_proba = None
             self.uncertainty = None # np.ndarray

         def run_pipeline(self):
             print(f"DummySingleModelWithProba '{self.name}' running pipeline...")
             self.model.fit(self.X_train, self.y_train)
             self.y_pred = self.model.predict(self.X_test)
             if hasattr(self.model, 'predict_proba'):
                 self.y_proba = self.model.predict_proba(self.X_test)
                 # Simulate uncertainty (entropy)
                 if self.y_proba is not None:
                      epsilon = 1e-9
                      probabilities = self.y_proba + epsilon
                      probabilities /= probabilities.sum(axis=1, keepdims=True)
                      self.uncertainty = entropy(probabilities, base=2, axis=1)
             print(f"DummySingleModelWithProba '{self.name}' pipeline done.")

         def get_name(self): return self.name
         def get_predictions(self): return self.y_pred
         def get_probabilities(self): return self.y_proba
         def get_uncertainty(self): return self.uncertainty
         def get_true_labels(self): return self.y_test
         def get_model(self): return self.model
         @property
         def X_test(self): return self._X_test # Provide X_test attribute

         # Set dummy X_test, X_train etc. in init for consistency
         def __init__(self, name, model, X_train, X_test, y_train, y_test, class_names):
             self.name = name
             self.model = model
             self.X_train, self._X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test # Use different name for internal storage if property exists
             self.class_names = class_names
             self.y_pred = None
             self.y_proba = None
             self.uncertainty = None


    try:
         # Sử dụng lớp SingleModel thực tế nếu có
         from SingleModel import SingleModel as ActualSingleModel
         print("Using ActualSingleModel from SingleModel.py")
         SingleModelClass = ActualSingleModel
    except ImportError:
         print("SingleModel.py not found, using DummySingleModelWithProba")
         SingleModelClass = DummySingleModelWithProba


    model_dt = DecisionTreeClassifier(random_state=42)
    single_dt = SingleModelClass(
        name="Dummy Decision Tree",
        model=model_dt,
        X_train=X_train_dummy, X_test=X_test_dummy, y_train=y_train_dummy, y_test=y_test_dummy,
        class_names=class_names_dummy
    )
    single_dt.run_pipeline()


    # Mô hình 2: SVC (có probability=True)
    def create_dummy_svc_pipeline_for_ensemble_test():
         return make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale', probability=True))

    model_svc = create_dummy_svc_pipeline_for_ensemble_test()
    single_svc = SingleModelClass(
        name="Dummy SVC (RBF)",
        model=model_svc,
        X_train=X_train_dummy, X_test=X_test_dummy, y_train=y_train_dummy, y_test=y_test_dummy,
        class_names=class_names_dummy
    )
    single_svc.run_pipeline()


    # Mô hình 3: Gaussian Naive Bayes (có predict_proba)
    from sklearn.naive_bayes import GaussianNB
    model_gnb = GaussianNB()
    single_gnb = SingleModelClass(
         name="Dummy Gaussian NB",
         model=model_gnb,
         X_train=X_train_dummy, X_test=X_test_dummy, y_train=y_train_dummy, y_test=y_test_dummy,
         class_names=class_names_dummy
    )
    single_gnb.run_pipeline()


    # Tạo danh sách các mô hình đơn lẻ đã chạy pipeline
    list_of_single_models = [single_dt, single_svc, single_gnb]

    # Khởi tạo EnsembleModel
    ensemble_analyzer = EnsembleModel(
        models=list_of_single_models,
        class_names=class_names_dummy,
        y_true=y_test_dummy # Cần nhãn thật của tập test để đánh giá ensemble
    )

    # Chạy và đánh giá các chiến lược ensemble
    print("\n--- Đánh giá các chiến lược Ensemble ---")

    # UL
    y_pred_ul = ensemble_analyzer.predict_ul()
    if y_pred_ul is not None:
        ensemble_analyzer.evaluate_ensemble(y_pred_ul, strategy_name="Uncertainty Lowest (UL)")

    # UT (cần chọn ngưỡng, ví dụ ngưỡng 0.5)
    UT_THRESHOLD = 0.5 # <-- Thay đổi ngưỡng để thử nghiệm
    y_pred_ut = ensemble_analyzer.predict_ut(threshold=UT_THRESHOLD)
    if y_pred_ut is not None:
         ensemble_analyzer.evaluate_ensemble(y_pred_ut, strategy_name=f"Uncertainty Threshold (UT, threshold={UT_THRESHOLD})")

    # UW
    y_pred_uw = ensemble_analyzer.predict_uw()
    if y_pred_uw is not None:
         ensemble_analyzer.evaluate_ensemble(y_pred_uw, strategy_name="Uncertainty Weighted (UW)")

    # CW
    y_pred_cw = ensemble_analyzer.predict_cw()
    if y_pred_cw is not None:
         ensemble_analyzer.evaluate_ensemble(y_pred_cw, strategy_name="Confidence Weighted (CW)")

    # Thêm đánh giá 2 chiến lược đơn giản: Mean và Max Probability Voting
    y_pred_mean_proba = ensemble_analyzer.predict_mean_proba_voting()
    if y_pred_mean_proba is not None:
         ensemble_analyzer.evaluate_ensemble(y_pred_mean_proba, strategy_name="Mean Probability Voting")

    y_pred_max_proba = ensemble_analyzer.predict_max_proba_voting()
    if y_pred_max_proba is not None:
         ensemble_analyzer.evaluate_ensemble(y_pred_max_proba, strategy_name="Max Probability Voting")


    print("\n--- Test EnsembleModel hoàn thành ---")