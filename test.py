import cv2
import pytesseract
from ultralytics import YOLO
import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox
import numpy as np
import torch
import re
import time  # Thêm thư viện để tính thời gian

# Đường dẫn Tesseract OCR (nếu sử dụng Windows, đặt đường dẫn này)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Kiểm tra nếu có GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Tải mô hình YOLO với GPU nếu có
try:
    model = YOLO('yolov8_license_plate_model.pt').to(device)
except Exception as e:
    print("Lỗi tải mô hình YOLO:", str(e))

# Hàm nhận dạng biển số bằng Tesseract OCR
def recognize_license_plate(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(binary, config='--psm 6')  # '6' là chế độ OCR nhiều dòng
        return filter_license_plate_text(text.strip())
    except Exception as e:
        print("Lỗi nhận dạng biển số:", str(e))
        return ""


# Hàm lọc các ký tự hợp lệ (chỉ giữ số, chữ, dấu "-" và dấu ".")
def filter_license_plate_text(text):
    return re.sub(r'[^a-zA-Z0-9.-]', '', text)

# Hàm định dạng biển số với dấu "-" giữa hai dòng
def format_license_plate_text(plate_text):
    lines = plate_text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    if len(lines) == 2:
        return f"{lines[0]}-{lines[1]}"  # Biển số 2 dòng có dấu "-"
    elif len(lines) == 1:
        return lines[0]  # Biển số 1 dòng
    else:
        return plate_text

# Hàm phát hiện biển số và xử lý ảnh
def detect_license_plate(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Không thể đọc ảnh từ đường dẫn.")

        results = model(image)
        if not results[0].boxes:
            print(f"Không phát hiện biển số nào trong ảnh {image_path}.")
            return image

        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
            cropped_plate = image[y1:y2, x1:x2]

        # Nhận diện biển số
            plate_text = recognize_license_plate(cropped_plate)
            formatted_plate_text = format_license_plate_text(plate_text)

           # Vẽ hình chữ nhật và chữ biển số lên ảnh
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Khung màu xanh lá
            cv2.putText(image, formatted_plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 7)  # Viền chữ màu đen
            cv2.putText(image, formatted_plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)  # Chữ màu vàng sáng

            print(f"Biển số nhận dạng: {formatted_plate_text}")

        return image
    except Exception as e:
        print("Lỗi phát hiện biển số:", str(e))
        return None

# Hàm để xử lý ảnh từ tệp
def select_image():
    image_path = filedialog.askopenfilename(title="Chọn Ảnh", filetypes=[("Image Files", "*.jpg;*.png")])
    if not image_path:
        messagebox.showwarning("Thông báo", "Bạn chưa chọn ảnh nào!")
        return

    print(f"Đang xử lý ảnh: {image_path}")
    processed_image = detect_license_plate(image_path)
    if processed_image is not None:
        show_resized_image(processed_image)

# Hàm để xử lý tất cả ảnh trong thư mục
def select_directory():
    folder_path = filedialog.askdirectory(title="Chọn Thư Mục")
    if not folder_path:
        messagebox.showwarning("Thông báo", "Bạn chưa chọn thư mục nào!")
        return

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.png')):  # Chỉ lấy các tệp ảnh
            image_path = os.path.join(folder_path, filename)
            print(f"Đang xử lý ảnh: {image_path}")

            # Nhận diện và xử lý từng ảnh
            processed_image = detect_license_plate(image_path)
            if processed_image is not None:
                if show_resized_image(processed_image):  # Dừng nếu nhấn 'q'
                    print("Đã thoát nhận diện.")
                    return

    messagebox.showinfo("Thông báo", "Hoàn thành xử lý tất cả ảnh trong thư mục.")



# Hàm để xử lý video
def select_video():
    # Mở hộp thoại chọn video
    video_path = filedialog.askopenfilename(title="Chọn Video", filetypes=[("Video Files", "*.mp4;*.avi")])
    if not video_path:
        messagebox.showwarning("Thông báo", "Bạn chưa chọn video nào!")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không thể mở video.")
        return

    frame_skip = 5  # Nhảy qua một số khung hình để tăng tốc độ
    frame_count = 0
    prev_time = time.time()  # Thời gian bắt đầu

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Tiền xử lý khung hình để cải thiện nhận diện
        frame_enhanced = enhance_frame(frame)

        # Tính toán FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Nhận diện biển số
        results = model(frame_enhanced)  # Dự đoán với mô hình YOLOv8
        if results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Kiểm tra kích thước vùng phát hiện
                if (x2 - x1) < 50 or (y2 - y1) < 20:  # Bỏ qua vùng quá nhỏ
                    continue

                # Cắt ảnh biển số và phóng to để tăng độ chính xác OCR
                plate_image = frame[y1:y2, x1:x2]
                plate_image_resized = cv2.resize(plate_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

                # Nhận dạng biển số từ ảnh cắt
                plate_text = recognize_license_plate(plate_image_resized)
                formatted_plate_text = format_license_plate_text(plate_text)

                # Vẽ hình chữ nhật quanh biển số
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Vẽ văn bản biển số lên video
                cv2.putText(frame, formatted_plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 7)  # Viền chữ màu đen
                cv2.putText(frame, formatted_plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)  # Chữ màu vàng sáng

                print(f"Biển số nhận dạng: {formatted_plate_text}")

                # Hiển thị ảnh biển số cắt riêng biệt
                cv2.imshow("Detected License Plate", plate_image_resized)  # Hiển thị ảnh biển số cắt

        # Hiển thị FPS trên khung hình
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Hiển thị video có các biển số nhận diện
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
            break

    cap.release()
    cv2.destroyAllWindows()


# Hàm tăng cường khung hình
def enhance_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)  # Cân bằng histogram
    enhanced = cv2.GaussianBlur(equalized, (3, 3), 0)  # Làm mờ nhẹ để giảm nhiễu
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)  # Trả về ảnh màu sau khi xử lý

# Hàm định dạng biển số
def format_license_plate_text(text):
    allowed_characters = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.")  # Các ký tự được phép
    return ''.join(char for char in text if char.upper() in allowed_characters)


# Hàm hiển thị ảnh giữ tỷ lệ
def show_resized_image(image):
    height, width = image.shape[:2]
    max_width = 800
    max_height = 600
    ratio = min(max_width / width, max_height / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    resized_image = cv2.resize(image, (new_width, new_height))

    window_name = "Detected License Plate"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, new_width, new_height)
    cv2.imshow(window_name, resized_image)

    # Chờ phím bấm 'q' để thoát
    key = cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    if key == ord('q'):  # Nhấn 'q' để thoát chương trình nhận diện
        cv2.destroyAllWindows()
        return True  # Trả về True để báo thoát
    return False  # Trả về False để tiếp tục


# Hàm hiển thị từng ảnh tuần tự
def show_images_one_by_one(images):
    for i, image in enumerate(images):
        height, width = image.shape[:2]
        max_width = 800
        max_height = 600
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        resized_image = cv2.resize(image, (new_width, new_height))

        window_name = f"Detected License Plate {i+1}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, new_width, new_height)
        cv2.imshow(window_name, resized_image)

        # Chờ phím bấm 'q' để thoát hoặc bất kỳ phím khác để chuyển ảnh tiếp theo
        key = cv2.waitKey(0)
        cv2.destroyWindow(window_name)
        if key == ord('q'):  # Nhấn 'q' để thoát
            cv2.destroyAllWindows()
            break



# Hàm để thoát chương trình
def exit_program(event=None):
    print("Thoát chương trình.")
    cv2.destroyAllWindows()
    root.quit()

# Giao diện chính
root = tk.Tk()
root.title("Nhận diện biển báo giao thông bằng YOLOv8")
root.geometry("1200x700")

# Tiêu đề chính
lbl = tk.Label(root, text="Nhận diện biển báo giao thông bằng YOLOv8", fg="blue", font=("Arial", 16, "bold"))
lbl.pack(pady=10)

# Khung chức năng
frame = tk.Frame(root)
frame.pack(pady=10)

btn_select_image = tk.Button(frame, text="Chọn Ảnh", command=select_image, font=("Arial", 14), bg="lightblue")
btn_select_image.pack(side="left", padx=10)

btn_select_directory = tk.Button(frame, text="Chọn Thư Mục", command=select_directory, font=("Arial", 14), bg="lightgreen")
btn_select_directory.pack(side="left", padx=10)

btn_select_video = tk.Button(frame, text="Chọn Video", command=select_video, font=("Arial", 14), bg="lightyellow")
btn_select_video.pack(side="left", padx=10)

btn_exit = tk.Button(frame, text="Thoát", command=exit_program, font=("Arial", 14), bg="salmon")
btn_exit.pack(side="left", padx=10)

# Khung hiển thị ảnh gốc
frame_original = tk.Frame(root)
frame_original.pack(side=tk.LEFT, padx=10, pady=20)
label_original_title = tk.Label(frame_original, text="Ảnh Gốc", font=("Arial", 12, "bold"))
label_original_title.pack()
media_label_original = tk.Label(frame_original, bg="gray", text="Ảnh Gốc", compound="center", width=80, height=25)
media_label_original.pack()

# Khung hiển thị kết quả nhận diện
frame_detected = tk.Frame(root)
frame_detected.pack(side=tk.RIGHT, padx=10, pady=20)
label_detected_title = tk.Label(frame_detected, text="Kết Quả Nhận Diện", font=("Arial", 12, "bold"))
label_detected_title.pack()
media_label_detected = tk.Label(frame_detected, bg="gray", text="Kết Quả Nhận Diện", compound="center", width=80, height=25)
media_label_detected.pack()

# Gắn sự kiện phím 'q' để thoát chương trình
root.bind('<q>', exit_program)

# Chạy giao diện
root.mainloop()
