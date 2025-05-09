# Tên file: SVCSpecific.py

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Sửa: Hàm chấp nhận tham số class_weight, probability và **kwargs
def create_specific_svc_pipeline(class_weight="balanced", probability=True, **kwargs):
    """
    Tạo và trả về một pipeline của scikit-learn bao gồm StandardScaler
    và một mô hình SVC được cấu hình sẵn hoặc tùy chỉnh.

    Args:
        class_weight (str hoặc dict, optional): Thiết lập class_weight cho SVC.
                                                "balanced" tự động điều chỉnh trọng số dựa trên tần suất lớp.
                                                None để không dùng trọng số.
                                                Một dict {class_label: weight} để gán trọng số thủ công.
                                                Mặc định là "balanced".
        probability (bool, optional): Bật/tắt tính năng ước tính xác suất.
                                      Cần thiết (=True) cho các chiến lược ensemble
                                      dựa trên xác suất và tính độ bất định.
                                      Lưu ý: Bật probability có thể làm chậm quá trình huấn luyện và dự đoán.
                                      Mặc định là True.
        **kwargs: Các tham số bổ sung sẽ được truyền trực tiếp tới constructor của SVC.
                  Ví dụ: kernel='linear', C=10.0, gamma='auto', decision_function_shape='ovr'.
                  Các tham số trùng tên với class_weight, probability, kernel, C, gamma
                  đã định nghĩa rõ ràng ở trên sẽ bị ghi đè bởi các tham số trong **kwargs
                  nếu chúng được cung cấp trong lệnh gọi hàm.

    Returns:
        sklearn.pipeline.Pipeline: Đối tượng pipeline SVC đã được cấu hình.
    """
    print("Đang tạo pipeline cho mô hình SVC...")

    # Định nghĩa tham số mặc định cho SVC
    # Bắt đầu với các tham số mặc định ban đầu (trước khi thêm flexibility)
    svc_params = {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        # class_weight và probability sẽ được thêm/ghi đè sau
    }

    # Cập nhật hoặc thêm các tham số được truyền qua **kwargs
    # Các tham số trong kwargs sẽ ghi đè các giá trị mặc định ban đầu
    svc_params.update(kwargs)

    # Cập nhật các tham số class_weight và probability từ đối số hàm.
    # Các đối số này (class_weight="balanced", probability=True)
    # sẽ ghi đè bất kỳ giá trị trùng tên nào có thể có trong kwargs ban đầu.
    # Điều này đảm bảo rằng nếu người dùng gọi `create_specific_svc_pipeline(probability=False, **{'probability': True})`,
    # thì `probability=False` sẽ được ưu tiên.
    # Cách an toàn hơn là thêm chúng SAU khi update từ kwargs,
    # hoặc kiểm tra xem chúng có trong kwargs không trước khi thêm.
    # Cách đơn giản nhất là thêm chúng vào dictionary sau khi update từ kwargs,
    # điều này đảm bảo đối số hàm luôn ưu tiên hơn kwargs.
    svc_params['probability'] = probability
    svc_params['class_weight'] = class_weight


    # Tạo mô hình SVC với các tham số đã tổng hợp
    # sklearn sẽ kiểm tra tính hợp lệ của các tham số (ví dụ: probability=True chỉ hoạt động với một số kernel)
    # và báo lỗi nếu có xung đột.
    print(f"  Tham số SVC cuối cùng: {svc_params}")
    try:
        svm_model = SVC(**svc_params)
    except TypeError as e:
        print(f"Lỗi khi tạo mô hình SVC với tham số: {e}. Kiểm tra lại các tham số truyền vào.")
        # Ném lỗi lại để quy trình dừng lại
        raise

    # Tạo pipeline
    pipeline = make_pipeline(
        StandardScaler(),
        svm_model # Thêm mô hình SVC đã cấu hình vào pipeline
    )
    print("Pipeline SVC đã được tạo.")
    return pipeline

# --- Cách sử dụng (ví dụ đơn giản trong cùng file) ---
if __name__ == "__main__":
    print("--- Test tạo pipeline SVC cụ thể (đã sửa) ---")

    # Test mặc định (class_weight='balanced', probability=True)
    print("\nTest với tham số mặc định:")
    svc_model_default = create_specific_svc_pipeline()
    print("Pipeline mặc định:", svc_model_default)
    print("Tham số SVC trong pipeline mặc định:", svc_model_default.named_steps['svc'].get_params())

    # Test tắt class_weight (class_weight=None)
    print("\nTest tắt class_weight:")
    svc_model_no_cw = create_specific_svc_pipeline(class_weight=None)
    print("Pipeline (no class_weight):", svc_model_no_cw)
    print("Tham số SVC trong pipeline (no class_weight):", svc_model_no_cw.named_steps['svc'].get_params())


    # Test tắt probability (probability=False)
    print("\nTest tắt probability:")
    svc_model_no_proba = create_specific_svc_pipeline(probability=False)
    print("Pipeline (no probability):", svc_model_no_proba)
    print("Tham số SVC trong pipeline (no probability):", svc_model_no_proba.named_steps['svc'].get_params())


    # Test ghi đè tham số khác (kernel, C)
    print("\nTest ghi đè tham số (kernel='linear', C=10):")
    # gamma='auto' thường cần cho kernel='linear' nếu không dùng 'scale'
    # **kwargs sẽ bao gồm {'kernel': 'linear', 'C': 10.0, 'gamma': 'auto'}
    svc_model_custom = create_specific_svc_pipeline(kernel='linear', C=10.0, gamma='auto')
    print("Pipeline tùy chỉnh:", svc_model_custom)
    print("Tham số SVC trong pipeline tùy chỉnh:", svc_model_custom.named_steps['svc'].get_params())

    # Test ghi đè cả class_weight và probability thông qua **kwargs (sẽ bị đối số hàm ghi đè)
    print("\nTest ghi đè class_weight='dict', probability=False qua kwargs (sẽ bị ghi đè):")
    svc_model_override_kwargs = create_specific_svc_pipeline(
        class_weight={'classA': 1, 'classB': 10}, probability=False, # Các đối số hàm này được ưu tiên
        **{'class_weight': 'balanced', 'probability': True, 'C': 0.5} # Các giá trị trong kwargs này cho class_weight, probability sẽ bị ghi đè
        )
    print("Pipeline ghi đè qua kwargs:", svc_model_override_kwargs)
    print("Tham số SVC trong pipeline ghi đè qua kwargs:", svc_model_override_kwargs.named_steps['svc'].get_params())
    # Output sẽ cho thấy class_weight là dict và probability là False như định nghĩa trong đối số hàm, C=0.5 từ kwargs.


    # Test ghi đè tham số mặc định ban đầu thông qua kwargs
    print("\nTest ghi đè tham số mặc định ban đầu (kernel='poly', degree=3):")
    # class_weight và probability sẽ dùng mặc định 'balanced' và True
    svc_model_override_default = create_specific_svc_pipeline(kernel='poly', degree=3, C=0.1)
    print("Pipeline ghi đè mặc định ban đầu:", svc_model_override_default)
    print("Tham số SVC trong pipeline ghi đè mặc định ban đầu:", svc_model_override_default.named_steps['svc'].get_params())


    # Bạn sẽ cần dữ liệu thực tế để huấn luyện và kiểm tra pipeline này.
    # Đây chỉ là để kiểm tra xem hàm tạo pipeline có hoạt động và nhận tham số đúng không.