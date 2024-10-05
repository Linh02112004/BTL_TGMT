from ultralytics import YOLO
import cv2

# Tải mô hình đã huấn luyện
model = YOLO('D:/BTL_TGMT/runs/detect/train2/weights/best.pt')  # Đường dẫn đến mô hình đã huấn luyện

# Khởi tạo webcam
cap = cv2.VideoCapture(0)  # 0 là ID của webcam

# Kiểm tra nếu webcam mở thành công
if not cap.isOpened():
    print("Khong the mo webcam.")
    exit()

while True:
    # Đọc khung hình từ webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Khong the doc khung hinh.")
        break

    # Lật khung hình theo chiều ngang
    frame = cv2.flip(frame, 1)  # 1 là flip code để lật ngang (flip horizontally)

    # Dự đoán
    results = model(frame)

    # Vẽ hộp bao quanh và nhãn
    annotated_frame = results[0].plot()

    # Hiển thị ảnh
    cv2.imshow("Du doan", annotated_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam và đóng tất cả cửa sổ
cap.release()
cv2.destroyAllWindows()
