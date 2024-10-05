from ultralytics import YOLO
import cv2

# Tải mô hình đã huấn luyện
model = YOLO('D:/BTL_TGMT/runs/detect/train2/weights/best.pt')  # Đường dẫn đến mô hình đã huấn luyện

# Đường dẫn đến ảnh để kiểm tra
image_path = r'D:/BTL_TGMT/AB.jpg'  # Thay đổi đường dẫn đến tệp ảnh cụ thể

# Đọc ảnh
img = cv2.imread(image_path)

# Kiểm tra nếu ảnh được đọc thành công
if img is not None:

    img_resized = cv2.resize(img, (500, 500))
    # Dự đoán
    results = model(img_resized)

    # Vẽ hộp bao quanh và nhãn
    annotated_frame = results[0].plot()

    # Hiển thị ảnh
    cv2.imshow("Du doan", annotated_frame)
    cv2.waitKey(0)
else:
    print("Khong the doc anh.")

# Đóng cửa sổ hiển thị
cv2.destroyAllWindows()