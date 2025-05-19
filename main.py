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
    model = YOLO('C:/Users/Del/Desktop/BTL_thigiacmaytinh/training_results_8n/license_plate_model/weights/best.pt').to(device)
except Exception as e:
    print("Lỗi tải mô hình YOLO:", str(e))

# Đường dẫn thư mục lưu kết quả
output_dir = 'C:/Users/Del/Desktop/BTL_thigiacmaytinh/output_images'
output_image_dir = os.path.join(output_dir, 'images')  # Thư mục lưu ảnh
output_video_dir = os.path.join(output_dir, 'videos')  # Thư mục lưu video
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_video_dir, exist_ok=True)

# Biến toàn cục để lưu trạng thái thư mục và danh sách ảnh
image_files = []
current_image_index = 0

# Xóa nội dung các label hiển thị
def clear_labels():
    media_label_original.configure(image='', text="Ảnh/Video Gốc")
    media_label_original.image = None
    media_label_detected.configure(image='', text="Ảnh/Video Nhận Diện")
    media_label_detected.image = None


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

def process_image(image_path):
    clear_labels()
    print(f"Đang xử lý ảnh: {image_path}")
    display_image(image_path, media_label_original)
    
    # Xử lý tên file
    base_name = os.path.basename(image_path)
    expected_name = re.sub(r'\s*\(\d+\)$', '', os.path.splitext(base_name)[0]).strip()
    print(f"Tên biển báo dự kiến: {expected_name}")
    
    # Thực hiện nhận diện bằng YOLOv8
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
        return

    # Dự đoán từ YOLOv8
    results = model.predict(source=image_path)
    
    detected_plate_text = ""  # Biến để lưu biển số phát hiện được

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Lấy tọa độ bounding box
        
        # Cắt ảnh biển số
        cropped_plate = image[y1:y2, x1:x2]
        
        # Nhận diện biển số bằng Tesseract
        plate_text = recognize_license_plate(cropped_plate)
        formatted_plate_text = format_license_plate_text(plate_text)

        # Cập nhật biển số phát hiện
        if formatted_plate_text:
            detected_plate_text = formatted_plate_text

        # Vẽ hình chữ nhật và chữ biển số lên ảnh
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Khung màu xanh lá
        cv2.putText(image, formatted_plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 7)  # Viền chữ màu đen
        cv2.putText(image, formatted_plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)  # Chữ màu vàng sáng

        print(f"Biển số nhận dạng: {formatted_plate_text}")

    global total_detection_time
    start_time = time.time()  # Ghi lại thời gian bắt đầu
    # --- Thực hiện nhận dạng ảnh ở đây ---
    time.sleep(1)  # Giả lập xử lý nhận dạng ảnh (thay thế bằng mã thực tế)
    # ---
    end_time = time.time()  # Ghi lại thời gian kết thúc
    detection_time = end_time - start_time
    total_detection_time += detection_time  # Cộng dồn thời gian xử lý ảnh
    print(f"Đã xử lý ảnh: {image_path}, Thời gian: {detection_time:.2f}s")


    # Lưu kết quả
    output_path = os.path.join(output_image_dir, f"detected_{base_name}")
    cv2.imwrite(output_path, image)
    print(f"Kết quả nhận diện được lưu tại: {output_path}")
    
    display_image(output_path, media_label_detected)
    
    # Cập nhật tiêu đề
    label_original_title.configure(text=f"Ảnh Gốc: {expected_name}")
    label_detected_title.configure(text=f"Kết Quả: {detected_plate_text if detected_plate_text else 'Không phát hiện biển số'}")


# Hàm để xử lý ảnh 
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
    
    # Đọc ảnh từ file
    image = cv2.imread(filename)
    if image is None:
        print(f"Không thể đọc ảnh từ đường dẫn: {filename}")
        return

    # Thực hiện nhận diện với YOLOv8
    results = model.predict(source=filename)
    
    detected_plate_text = ""  # Biến lưu biển số phát hiện
    
    # Duyệt qua các kết quả nhận diện
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Lấy tọa độ bounding box
        
        # Cắt ảnh biển số
        cropped_plate = image[y1:y2, x1:x2]
        
        # Nhận diện biển số bằng Tesseract
        plate_text = recognize_license_plate(cropped_plate)
        formatted_plate_text = format_license_plate_text(plate_text)

        # Cập nhật biển số phát hiện
        if formatted_plate_text:
            detected_plate_text = formatted_plate_text

        # Vẽ hình chữ nhật và biển số lên ảnh
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Khung màu xanh lá
        cv2.putText(image, formatted_plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 7)  # Viền chữ màu đen
        cv2.putText(image, formatted_plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)  # Chữ màu vàng sáng

        print(f"Biển số nhận dạng: {formatted_plate_text}")

    # Kiểm tra khớp
    if detected_plate_text.lower() == expected_name.lower():
        result_message = "Biển số nhận diện trùng với tên file! Nhận diện đúng!"
    else:
        result_message = "Biển số nhận diện không trùng với tên file! Nhận diện sai!"
    
    print(f"Kết quả nhận diện: {detected_plate_text}")
    messagebox.showinfo("Nhận diện", result_message)
    
    # Lưu kết quả
    output_path = os.path.join(output_image_dir, f"detected_{base_name}")
    cv2.imwrite(output_path, image)
    print(f"Kết quả nhận diện được lưu tại: {output_path}")
    
    display_image(output_path, media_label_detected)
    
    # Cập nhật tiêu đề
    label_original_title.configure(text=f"Ảnh Gốc: {expected_name}")
    label_detected_title.configure(text=f"Kết Quả: {detected_plate_text if detected_plate_text else 'Không phát hiện biển số'}")

# Hàm để xử lý video
def detect_on_video(video_path, skip_frames=2, min_area=1000):
    clear_labels()  # Hàm để xóa nhãn trước khi nhận diện
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        messagebox.showerror("Lỗi", f"Không thể mở video: {video_path}")
        return
    
    # Lấy thông tin về video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_video_path = os.path.join(output_video_dir, f"detected_{os.path.basename(video_path)}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()

    def update_frame():
        nonlocal frame_count, start_time
        ret, frame = cap.read()
        if not ret:
            cap.release()
            out.release()
            messagebox.showinfo("Thông báo", f"Kết quả nhận diện video đã được lưu tại: {output_video_path}")
            return
        
        frame_count += 1

        # Nhảy qua một số khung hình để tăng tốc độ
        if frame_count % skip_frames != 0:
            media_label_detected.after(10, update_frame)
            return

        # Nhận diện biển số xe bằng YOLOv8
        results = model.predict(source=frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Lấy danh sách bounding box
        annotated_frame = frame.copy()  # Khung hình gốc chưa có đánh dấu
        # Xử lý từng biển số được nhận diện
        for box in boxes:
            x1, y1, x2, y2 = box  # Chỉ lấy tọa độ của bounding box (4 giá trị)
            width_box = x2 - x1
            height_box = y2 - y1
            area = width_box * height_box
            # Bỏ qua những vùng có diện tích quá nhỏ
            if area < min_area:
                continue
            license_plate = frame[int(y1):int(y2), int(x1):int(x2)]
            if license_plate.size > 0:
                # Sử dụng hàm nhận diện biển số
                plate_text = recognize_license_plate(license_plate)
                if plate_text:
                    # Vẽ bounding box và văn bản biển số lên video
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # Vẽ viền chữ (màu đen) trước
                    cv2.putText(annotated_frame, plate_text, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 7, cv2.LINE_AA)
                    
                    # Vẽ chữ biển số (màu vàng sáng)
                    cv2.putText(annotated_frame, plate_text, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3, cv2.LINE_AA)
                    
                    print(f"Biển số nhận diện: {plate_text}")

        # Cập nhật FPS và hiển thị
        editable_frame = annotated_frame.copy()
        current_time = time.time()
        elapsed_time = current_time - start_time
        current_fps = frame_count / elapsed_time
        fps_text = f"FPS: {current_fps:.2f}"
        cv2.putText(editable_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Ghi video đã xử lý vào file đầu ra
        out.write(editable_frame)

        # Hiển thị ảnh gốc
        frame_rgb_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image_original = Image.fromarray(frame_rgb_original)
        frame_image_resized_original = frame_image_original.resize((600, 400), Image.LANCZOS)
        frame_tk_original = ImageTk.PhotoImage(frame_image_resized_original)
        media_label_original.configure(image=frame_tk_original, text='')
        media_label_original.image = frame_tk_original

        # Hiển thị ảnh đã xử lý với bounding box và ký tự nhận diện
        frame_rgb_detected = cv2.cvtColor(editable_frame, cv2.COLOR_BGR2RGB)
        frame_image_detected = Image.fromarray(frame_rgb_detected)
        frame_image_resized_detected = frame_image_detected.resize((600, 400), Image.LANCZOS)
        frame_tk_detected = ImageTk.PhotoImage(frame_image_resized_detected)
        media_label_detected.configure(image=frame_tk_detected, text='')
        media_label_detected.image = frame_tk_detected

        # Lặp lại cập nhật khung hình
        media_label_detected.after(10, update_frame)

    update_frame()

# Hiển thị ảnh lên giao diện Tkinter
def display_image(image_path, label):
    try:
        image = Image.open(image_path)
        image = image.resize((600, 400), Image.LANCZOS)
        icon = ImageTk.PhotoImage(image)
        label.configure(image=icon, text='')
        label.image = icon
    except Exception as e:
        print(f"Lỗi khi hiển thị ảnh: {e}")

# Chọn thư mục chứa ảnh
# def select_folder():
#     """
#     Hàm chọn thư mục chứa ảnh và tính tổng thời gian nhận dạng.
#     """
#     global image_files, current_image_index, total_detection_time
#     folder_path = filedialog.askdirectory(title="Chọn thư mục chứa ảnh")
#     if not folder_path:
#         return

#     # Lấy danh sách các ảnh trong thư mục
#     image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#     if not image_files:
#         messagebox.showinfo("Thông báo", "Thư mục không chứa ảnh!")
#         return

#     # Đặt lại chỉ số và tổng thời gian
#     current_image_index = 0
#     total_detection_time = 0

#     # Xử lý từng ảnh trong thư mục
#     for image_path in image_files:
#         process_image(image_path)

#     # Hiển thị tổng thời gian xử lý
#     messagebox.showinfo("Thông báo", f"Xử lý hoàn tất! Tổng thời gian nhận dạng: {total_detection_time:.2f} giây")

def select_folder():
    global image_files, current_image_index
    folder_path = filedialog.askdirectory(title="Chọn thư mục chứa ảnh")
    if not folder_path:
        return
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        messagebox.showinfo("Thông báo", "Thư mục không chứa ảnh!")
        return
    current_image_index = 0
    process_image(image_files[current_image_index]) 

# Chuyển sang ảnh tiếp theo
def next_image():
    global current_image_index
    if not image_files:
        messagebox.showinfo("Thông báo", "Chưa có ảnh nào được chọn!")
        return
    current_image_index += 1
    if current_image_index >= len(image_files):
        messagebox.showinfo("Thông báo", "Bạn đã đến ảnh cuối cùng!")
        current_image_index = len(image_files) - 1
        return
    process_image(image_files[current_image_index])

# Giao diện chính
root = tk.Tk()
root.title("Nhận dạng biển số xe ô tô")
lbl = tk.Label(root, text="Nhận dạng biển số xe ô tô bằng YOLOv8 và TesseractOCR", fg="blue", font=("Arial", 16, "bold"))
lbl.pack(pady=10)

frame = tk.Frame(root)
frame.pack(pady=10)

# Các nút chức năng
btn_image = tk.Button(frame, text="Chọn Ảnh", command=detect_on_image, height=2, width=15, bg="lightblue")
btn_image.pack(side=tk.LEFT, padx=10)

btn_folder = tk.Button(frame, text="Chọn Thư Mục", command=select_folder, height=2, width=15, bg="lightgreen")
btn_folder.pack(side=tk.LEFT, padx=10)

btn_video = tk.Button(frame, text="Chọn Video", command=lambda: detect_on_video(filedialog.askopenfilename(
    title="Chọn video",
    filetypes=(("Video files", "*.mp4;*.avi;*.mkv"), ("All files", "*.*"))
)), height=2, width=15, bg="lightyellow")
btn_video.pack(side=tk.LEFT, padx=10)

frame_controls = tk.Frame(root)
frame_controls.pack(pady=10)

# Nút "Next" để chuyển ảnh
btn_next = tk.Button(frame_controls, text="Ảnh Tiếp Theo", command=next_image, height=2, width=15, bg="#D8B7DD")
btn_next.pack(side=tk.LEFT, padx=10)

# Nút Thoát
btn_exit = tk.Button(frame_controls, text="Thoát", command=root.quit, bg="salmon", height=2, width=15)
btn_exit.pack(side=tk.LEFT, padx=10)

# Các khung để hiển thị ảnh gốc và ảnh nhận diện
frame_original = tk.Frame(root)
frame_original.pack(side=tk.LEFT, padx=10, pady=20)
label_original_title = tk.Label(frame_original, text="Ảnh/Video Gốc", font=("Arial", 12, "bold"))
label_original_title.pack()
media_label_original = tk.Label(frame_original, bg="gray", text="Ảnh/Video Gốc", compound="center", width=600, height=400)
media_label_original.pack()

frame_detected = tk.Frame(root)
frame_detected.pack(side=tk.RIGHT, padx=10, pady=20)
label_detected_title = tk.Label(frame_detected, text="Kết Quả Nhận Diện", font=("Arial", 12, "bold"))
label_detected_title.pack()
media_label_detected = tk.Label(frame_detected, bg="gray", text="Ảnh Nhận Diện", compound="center", width=600, height=400)
media_label_detected.pack()

root.mainloop()
