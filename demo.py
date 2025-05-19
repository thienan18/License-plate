from ultralytics import YOLO
import pytesseract
import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import re
import time
import torch

# Đường dẫn Tesseract OCR (nếu sử dụng Windows, đặt đường dẫn này)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Kiểm tra nếu có GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Tải mô hình YOLO với GPU nếu có
try:
    model = YOLO('yolov8_license_plate_model.pt').to(device)
except Exception as e:
    print("Lỗi tải mô hình YOLO:", str(e))

# Đường dẫn thư mục lưu kết quả
output_dir = 'C:/Users/Del/Desktop/BTL_thigiacmaytinh/output_images'
output_image_dir = os.path.join(output_dir, 'images')  # Thư mục lưu ảnh
output_video_dir = os.path.join(output_dir, 'videos')  # Thư mục lưu video
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_video_dir, exist_ok=True)


# Hàm nhận diện biển số
def recognize_license_plate(plate_image):
    text = pytesseract.image_to_string(plate_image, config='--psm 8')  # Tesseract OCR cho biển số
    return text.strip()

# Hàm chuẩn hóa định dạng biển số
def format_license_plate_text(plate_text):
    formatted_text = plate_text.replace(" ", "").upper()  # Xóa dấu cách và chuyển thành chữ hoa
    return formatted_text

# Xử lý ảnh
def process_image(image_path):
    clear_labels()
    print(f"Đang xử lý ảnh: {image_path}")
    display_image(image_path, media_label_original)
    
    # Xử lý tên file
    base_name = os.path.basename(image_path)
    expected_name = re.sub(r'\s*\(\d+\)$', '', os.path.splitext(base_name)[0]).strip()  # Xử lý tên ảnh
    print(f"Tên biển số dự kiến: {expected_name}")
    
    # Thực hiện nhận diện biển số với YOLO
    results = model.predict(source=image_path)  # Dự đoán biển số từ ảnh
    annotated_image = results[0].plot()  # Vẽ box và thông tin lên ảnh
    
    # Lấy lớp nhận diện (biển số)
    detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
    detected_name = detected_classes[0] if detected_classes else "Không xác định được biển số"
    
    # Xử lý biển số xe từ các box
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Tọa độ của box
        cropped_plate = annotated_image[y1:y2, x1:x2]  # Cắt vùng biển số
        
        # Nhận diện biển số từ vùng cắt bằng Tesseract OCR
        plate_text = recognize_license_plate(cropped_plate)
        formatted_plate_text = format_license_plate_text(plate_text)

        # Vẽ hình chữ nhật và chữ biển số lên ảnh
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Khung màu xanh lá
        cv2.putText(annotated_image, formatted_plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)  # Viền chữ màu đen
        cv2.putText(annotated_image, formatted_plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)  # Chữ màu vàng sáng

        print(f"Biển số nhận diện: {formatted_plate_text}")
    
    # Lưu kết quả
    output_path = os.path.join(output_image_dir, f"detected_{base_name}")
    cv2.imwrite(output_path, annotated_image)  # Lưu ảnh đã được nhận diện
    print(f"Kết quả nhận diện được lưu tại: {output_path}")
    
    display_image(output_path, media_label_detected)  # Hiển thị ảnh đã nhận diện lên giao diện
    
    # Cập nhật tiêu đề
    label_original_title.configure(text=f"Ảnh Gốc: {expected_name}")
    label_detected_title.configure(text=f"Kết Quả: {detected_name}")

# Xử lý ảnh từ thư mục
def detect_on_image():
    clear_labels()
    filename = filedialog.askopenfilename(
        title="Chọn ảnh",
        filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*"))
    )
    if not filename:
        return
    print(f"Đã chọn ảnh: {filename}")
    display_image(filename, media_label_original)
    
    # Xử lý tên file
    base_name = os.path.basename(filename)
    expected_name = re.sub(r'\s*\(\d+\)$', '', os.path.splitext(base_name)[0]).strip()
    print(f"Tên biển báo dự kiến (sau xử lý): {expected_name}")
    
    # Thực hiện nhận diện biển số với YOLO
    results = model.predict(source=filename)  # Dự đoán biển số từ ảnh
    annotated_image = results[0].plot()  # Vẽ box và thông tin lên ảnh
    
    # Lấy lớp nhận diện (biển số)
    detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
    detected_name = detected_classes[0] if detected_classes else "Không xác định được biển số"
    
    # Xử lý biển số xe từ các box và nhận diện ký tự từ Tesseract
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Tọa độ của box
        cropped_plate = annotated_image[y1:y2, x1:x2]  # Cắt vùng biển số
        
        # Nhận diện biển số từ vùng cắt bằng Tesseract OCR
        plate_text = recognize_license_plate(cropped_plate)
        formatted_plate_text = format_license_plate_text(plate_text)

        # Vẽ hình chữ nhật và chữ biển số lên ảnh
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Khung màu xanh lá
        cv2.putText(annotated_image, formatted_plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)  # Viền chữ màu đen
        cv2.putText(annotated_image, formatted_plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)  # Chữ màu vàng sáng

        print(f"Biển số nhận diện: {formatted_plate_text}")
    
    # Lưu kết quả
    output_path = os.path.join(output_image_dir, f"detected_{base_name}")
    cv2.imwrite(output_path, annotated_image)  # Lưu ảnh đã được nhận diện
    print(f"Kết quả nhận diện được lưu tại: {output_path}")
    
    display_image(output_path, media_label_detected)  # Hiển thị ảnh đã nhận diện lên giao diện
    
    # Cập nhật tiêu đề
    label_original_title.configure(text=f"Ảnh Gốc: {expected_name}")
    label_detected_title.configure(text=f"Kết Quả: {detected_name}")

# Hiển thị ảnh lên giao diện Tkinter
def display_image(image_path, label):
    try:
        image = Image.open(image_path)
        image = image.resize((600, 400), Image.LANCZOS)
        icon = ImageTk.PhotoImage(image)
        label.configure(image=icon, text='')  # Cập nhật ảnh lên label
        label.image = icon
    except Exception as e:
        print(f"Lỗi khi hiển thị ảnh: {e}")

# Hàm làm sạch các label hiển thị
def clear_labels():
    media_label_original.configure(image='', text="Ảnh Gốc")
    media_label_original.image = None
    media_label_detected.configure(image='', text="Ảnh Nhận Diện")
    media_label_detected.image = None

# Thiết lập giao diện Tkinter
root = tk.Tk()
root.title("Nhận diện biển số xe ô tô bằng YOLOv8 và Tesseract OCR")

# Các Label và Button cho giao diện
lbl = tk.Label(root, text="Nhận diện biển số xe ô tô bằng YOLOv8", fg="blue", font=("Arial", 16, "bold"))
lbl.pack(pady=10)

frame = tk.Frame(root)
frame.pack(pady=10)

btn_image = tk.Button(frame, text="Chọn Ảnh", command=detect_on_image, height=2, width=15, bg="lightblue")
btn_image.pack(side=tk.LEFT, padx=10)

# Các frame cho ảnh gốc và ảnh đã nhận diện
frame_original = tk.Frame(root)
frame_original.pack(side=tk.LEFT, padx=10, pady=20)
label_original_title = tk.Label(frame_original, text="Ảnh Gốc", font=("Arial", 12, "bold"))
label_original_title.pack()
media_label_original = tk.Label(frame_original, bg="gray", text="Ảnh Gốc", compound="center", width=600, height=400)
media_label_original.pack()

frame_detected = tk.Frame(root)
frame_detected.pack(side=tk.RIGHT, padx=10, pady=20)
label_detected_title = tk.Label(frame_detected, text="Kết Quả Nhận Diện", font=("Arial", 12, "bold"))
label_detected_title.pack()
media_label_detected = tk.Label(frame_detected, bg="gray", text="Ảnh Nhận Diện", compound="center", width=600, height=400)
media_label_detected.pack()

root.mainloop()
