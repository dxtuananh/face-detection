import cv2

# Đường dẫn đến file haarcascade
haarcascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# Tạo đối tượng nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier(haarcascade_path)
# Khởi tạo camera
cap = cv2.VideoCapture(1)  # Sử dụng camera mặc định, nếu có nhiều camera => thay đổi index

while True:
    # Đọc frame từ camera
    ret, frame = cap.read()
    if not ret:
        break
    # Chuyển đổi frame sang ảnh xám (Gray)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Vẽ hình chữ nhật quanh các khuôn mặt được phát hiện
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Hiển thị frame với khuôn mặt được phát hiện
    cv2.imshow('Face Detection', frame)

    # Thoát khỏi vòng lặp nếu nhấn phím 'e'
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

# Giải phóng camera và đóng cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()
