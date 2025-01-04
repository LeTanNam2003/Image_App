from PIL import Image, ImageTk, ImageEnhance, ImageOps
from tkinter import filedialog,Canvas
from scipy.ndimage import gaussian_filter as gf
from scipy.ndimage import median_filter as scipy_median_filter
from scipy.ndimage import uniform_filter
import numpy as np
import customtkinter as ctk
import cv2
from scipy import fftpack
import huffmanEncode
from bitstream import BitStream
import os
from skimage.util import random_noise
from rembg import remove 
from pyxelate import Pyx
import io
import tkinter.messagebox as messagebox
import tkinter as tk


# ghi đè vào thư viện để trả về tọa độ
class CTkCanvas(tk.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aa_circle_canvas_ids = set()
    def coords(self, tag_or_id, *args):
        if type(tag_or_id) == str and "ctk_aa_circle_font_element" in self.gettags(tag_or_id):
            coords_id = self.find_withtag(tag_or_id)[0]  # take the lowest id for the given tag
            super().coords(coords_id, *args[:2])

            if len(args) == 3:
                super().itemconfigure(coords_id, font=("CustomTkinter_shapes_font", -int(args[2]) * 2), text=self._get_char_from_radius(args[2]))

        elif type(tag_or_id) == int and tag_or_id in self._aa_circle_canvas_ids:
            super().coords(tag_or_id, *args[:2])

            if len(args) == 3:
                super().itemconfigure(tag_or_id, font=("CustomTkinter_shapes_font", -args[2] * 2), text=self._get_char_from_radius(args[2]))

        else:
            super().coords(tag_or_id, *args)
        return super().coords(tag_or_id)

def open_image(event=None):
    filepath = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
    if filepath:
        load_image(filepath)
    hide_all_menus()
    

def get_file_size_from_image(image):
    """
    Tính dung lượng file ảnh từ đối tượng PIL Image.
    :param image: Đối tượng ảnh PIL Image.
    :return: Dung lượng file ảnh (KB).
    """
    # Chuyển đổi ảnh sang chế độ L (grayscale) hoặc RGB nếu ảnh có chế độ 'F'
    if image.mode == 'F':  # Nếu ảnh có chế độ floating point
        image = image.convert('L')  # Chuyển sang grayscale (hoặc 'RGB' nếu cần)

    # Xác định định dạng ảnh hoặc mặc định là PNG
    image_format = image.format if image.format else "PNG"
    
    # Lưu ảnh vào buffer để tính dung lượng
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    
    # Lấy kích thước file từ buffer
    size_in_bytes = buffer.tell()  # Số byte
    size_in_kb = size_in_bytes / 1024  # Chuyển sang KB
    return size_in_kb

def save_image(event=None):
   hide_all_menus()
   global img_edited  # Sử dụng ảnh đã chỉnh sửa để lưu
   if img_edited:
        filepath = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG", "*.png"),
                                                           ("JPEG", "*.jpg"),
                                                           ("All Files", "*.*")])
        if filepath:
            img_edited.save(filepath)  # Lưu ảnh đã chỉnh sửa
def hide_all_menus():
    global is_file_menu_open, is_help_menu_open
    file_menu_frame.place_forget()
    help_menu_frame.place_forget()
    is_file_menu_open = False
    is_help_menu_open = False
def load_image(filepath):
    
    global img, img_edited, preview_img, preview_display
    hide_all_menus()
    clear_tool_space()
    
    img = Image.open(filepath)
    img_edited = None  # Reset lại ảnh chỉnh sửa
    reset_image()
    # Hiển thị ảnh chính
    display_image(img)

    # Cập nhật ảnh xem trước
    preview_img = img.resize((200, 200))
    preview_display = ImageTk.PhotoImage(preview_img)
    preview_label.configure(image=preview_display)

    file_size=get_file_size_from_image(img)
    info_label.configure(
        text=f"IMAGE INFO:\n\nColor Space: {img.mode}\n\nImage Size: {img.width}x{img.height}\n\nFile Size: {file_size:.2f} KB")  

def clear_tool_space():
    if config_label:
        config_label.pack_forget()
    if adj_config:
        adj_config.pack_forget()
    canvas.delete(rect_id)
    hide_basic_scale()
    hide_filter_set()
    hide_cut_grid()
    hide_edge_set()
    hide_resize_grid()
    hide_quality_Option()
    clear_noise_space()
    clear_corners()

def reset_image():
    global img,img_edited, brightness_scale,saturation_scale,sharpness_scale,contrast_scale,rect_id,noisy_img   
    if config_label:
        config_label.pack_forget()
    if adj_config:
        adj_config.pack_forget()
    if img_edited:
        img_edited=None

    noisy_img=None
    # Đặt lại giá trị của các thanh trượt về 0
    brightness_scale.set(0.0)
    saturation_scale.set(0.0)
    sharpness_scale.set(0.0)
    contrast_scale.set(0.0)
    bin_scale.set(0)
    noise_slider.set(0.0)
    edge_dropdown.set("None") 
    filter_dropdown.set("None")
    edge_dropdown.set("None") 
    noise_dropdown.set("None")
    canvas.delete(rect_id)
    clear_noise_space()    
    hide_basic_scale()
    hide_filter_set()
    hide_edge_set()
    hide_cut_grid()
    hide_resize_grid()
    hide_quality_Option()
    clear_corners()
    file_size=get_file_size_from_image(img)
    info_label.configure(
        text=f"IMAGE INFO:\n\nColor Space: {img.mode}\n\nImage Size: {img.width}x{img.height}\n\nFile Size: {file_size:.2f} KB"
    )
    canvas.bind("<ButtonPress-1>", start_drag)  # Nhấn giữ chuột trái
    canvas.bind("<B1-Motion>", on_drag)         # Kéo chuột
    canvas.bind("<ButtonRelease-1>", stop_drag)  # Thả chuột trái
    canvas.bind("<MouseWheel>", on_mouse_wheel) 
    display_image(img)

def hide_edge_set():
    if(edge_dropdown):
        edge_dropdown.pack_forget()

def hide_basic_scale():
    # Kiểm tra từng widget trước khi ẩn
    if brightness_scale:
        brightness_scale.pack_forget()
    if brightness_label:
        brightness_label.pack_forget()
    if contrast_label:
        contrast_label.pack_forget()
    if contrast_scale:
        contrast_scale.pack_forget()
    if saturation_scale:
        saturation_scale.pack_forget()
    if saturation_label:
        saturation_label.pack_forget()
    if sharpness_label:
        sharpness_label.pack_forget()
    if sharpness_scale:
        sharpness_scale.pack_forget()
    if bin_scale:
        bin_scale.pack_forget()
    if bin_label:
        bin_label.pack_forget()


def adjust_brightness(value):
    global img, img_edited
    if img is None:
        return  # Không có ảnh để chỉnh sửa
    if img_edited is None:
        img_edited = img.copy()
    tmp_img=img_edited.copy()

    # Điều chỉnh độ sáng
    enhancer = ImageEnhance.Brightness(tmp_img)
    tmp_img = enhancer.enhance(1 + float(value))  # Cập nhật img_edited
    # Hiển thị ảnh sau khi điều chỉnh
    display_image(tmp_img)
    #img_edited = img.copy()

def adjust_saturation(value):
    global img, img_edited
    if img is None:
        return  # Không có ảnh để chỉnh sửa

    if img_edited is None:
        img_edited = img.copy()
    tmp_img=img_edited.copy()

    # Điều chỉnh độ bão hòa
    enhancer = ImageEnhance.Color(tmp_img)
    tmp_img = enhancer.enhance(1 + float(value))  # Cập nhật img_edited

    # Hiển thị ảnh sau khi điều chỉnh
    display_image(tmp_img)
    #img_edited = img.copy()

def adjust_contrast(value):
    global img, img_edited
    if img is None:
        return  # Không có ảnh để chỉnh sửa

    if img_edited is None:
        img_edited = img.copy()
    tmp_img=img_edited.copy()

    # Điều chỉnh độ tương phản
    enhancer = ImageEnhance.Contrast(tmp_img)
    tmp_img = enhancer.enhance(1 + float(value))  # Cập nhật img_edited

    # Hiển thị ảnh sau khi điều chỉnh
    display_image(tmp_img)


def adjust_sharpness(value):
    global img, img_edited
    if img is None:
        return  # Không có ảnh để chỉnh sửa

    if img_edited is None:
        img_edited = img.copy()
    tmp_img=img_edited.copy()
    # Điều chỉnh độ nét
    enhancer = ImageEnhance.Sharpness(tmp_img)
    tmp_img = enhancer.enhance(1 + float(value))  # Cập nhật img_edited

    # Hiển thị ảnh sau khi điều chỉnh
    display_image(tmp_img)
    #img_edited = img.copy()
def adjust_zoom(value):
    global img, img_edited
    if img is None:
        return  # Không có ảnh để chỉnh sửa
    
    img_edited_new = img.copy() # Original: img.edited = img.copy()

    # Tính toán tỷ lệ zoom
    scale_factor = float(value)
    
    # Resize ảnh theo tỷ lệ zoom với phương pháp LANCZOS (chất lượng cao)
    new_width = int(img_edited_new.width * scale_factor)
    new_height = int(img_edited_new.height * scale_factor)
    resized_img = img_edited_new.resize((new_width, new_height), Image.Resampling.LANCZOS)  # Dùng LANCZOS để giữ chất lượng ảnh

    # Cập nhật img_edited để giữ kích thước đã zoom
    img_edited_new = resized_img
    
    # Hiển thị ảnh đã thay đổi kích thước
    display_image_zoom(img_edited_new)



def display_image_zoom(image):
    global img_display, canvas_width, canvas_height
    img_display = ImageTk.PhotoImage(image)

    # Xóa ảnh cũ trên canvas
    canvas.delete("all")

    canvas_width =  canvas.winfo_width()
    canvas_height = canvas.winfo_height()
   

    # Lấy kích thước ảnh
    img_width = image.width
    img_height = image.height

    # Tính toán vị trí để căn giữa
    x_offset = (canvas_width - img_width) // 2
    y_offset = (canvas_height - img_height) // 2

    # Hiển thị ảnh tại vị trí căn giữa
    canvas.create_image(x_offset, y_offset, anchor='nw', image=img_display)    

def display_image(image):
    global img_display, canvas_width, canvas_height,displayed_img_coords,x_offset,y_offset,resized_image
        # Lấy kích thước hiện tại của canvas
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    
    # In ra kích thước của Canvas
    print(f"Canvas Width: {canvas_width}, Canvas Height: {canvas_height}")
    #info_label.configure( text=f"IMAGE INFO:\n\n\nColor Space: {img.mode}\n\nImage Size: {img.width}x{img.height}")
    # Lấy kích thước gốc của ảnh
    img_width = image.width
    img_height = image.height
    
    # Tính toán tỷ lệ scale
    scale_x = canvas_width / img_width
    scale_y = canvas_height / img_height
    scale = min(scale_x, scale_y)

    # Tính kích thước ảnh mới
    new_width = int(img_width * scale*0.7)
    new_height = int(img_height * scale*0.7)
       # Resize ảnh để vừa với canvas
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Chuyển đổi sang định dạng hiển thị trên Tkinter
    img_display = ImageTk.PhotoImage(resized_image)

    # Xóa ảnh cũ trên canvas
    canvas.delete("all")

    # Tính toán vị trí để căn giữa
    x_offset = (canvas_width - new_width) // 2
    y_offset = (canvas_height - new_height) // 2

    # Hiển thị ảnh tại vị trí căn giữa
    canvas.create_image(x_offset, y_offset, anchor='nw', image=img_display)
    displayed_img_coords = (x_offset, y_offset, new_width, new_height)

def display_resize(image):
    global img_display,x_offset,y_offset
    # Lấy kích thước gốc của ảnh
    img_width = image.width
    img_height = image.height

    # Cập nhật thông tin ảnh
    file_size=get_file_size_from_image(image)
    info_label.configure(
        text=f"IMAGE INFO:\n\nColor Space: {image.mode}\n\nImage Size: {image.width}x{image.height}\n\nFile Size: {file_size:.2f} KB"
    )

    # Chuyển đổi ảnh sang định dạng hiển thị trên Tkinter
    img_display = ImageTk.PhotoImage(image)

    # Xóa ảnh cũ trên canvas
    canvas.delete("all")

    # Tính toán vị trí để căn giữa canvas
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    x_offset = (canvas_width - img_width) // 2
    y_offset = (canvas_height - img_height) // 2

    # Hiển thị ảnh gốc trên canvas
    canvas.create_image(x_offset, y_offset, anchor='nw', image=img_display)

def on_resize(event):
    global img,img_edited
    
    display_image(img)
    img_edited=None

def show_basic_scale():
   clear_tool_space()
   adj_config.pack(fill="x", pady=20)
   config_label.pack(side="top", padx=10, pady=10)
   brightness_control()
   saturation_control()
   sharpness_control()
   contrast_control()


def zoom_control():   
    zoom_label.pack(side="top", padx=10, pady=5)
    zoom_scale.pack(side="top", padx=10, pady=5)

def brightness_control():
    
    
    brightness_label.pack(side="top", padx=10, pady=5)
    brightness_scale.pack(side="top", padx=10, pady=5)


def saturation_control():
    saturation_label.pack(side="top", padx=10, pady=5)
    saturation_scale.pack(side="top", padx=10, pady=5)

def sharpness_control():
    sharpness_label.pack(side="top", padx=10, pady=5)
    sharpness_scale.pack(side="top", padx=10, pady=5)

def contrast_control():
    contrast_label.pack(side="top", padx=10, pady=5)
    contrast_scale.pack(side="top", padx=10, pady=(5,10))

def adjust_threshold(value):
    threshold = int(value)  # Chuyển giá trị từ chuỗi sang số nguyên
    gray2bin(threshold) 

def bin_scale_control():
    clear_tool_space()
    config_label.configure(text="GRAY TO BIN")
    adj_config.pack(fill="x", pady=20)
    config_label.pack(side="top", padx=10, pady=10)
    bin_label.pack(side="top", padx=10, pady=5)    
    bin_scale.pack(side="top", padx=10, pady=(5,10))

def rotate_image():
    
    global img, img_edited  # Sử dụng biến toàn cục để cập nhật ảnh gốc
    if img_edited is  None:
        img_edited=img.copy()
    img_edited = img_edited.rotate(90, expand=True)  # Xoay ảnh và cập nhật lại biến img
    display_image(img_edited)  # Hiển thị ảnh đã xoay
    
    
def rgb2gray():
    clear_tool_space()
    global img, img_edited
    if img is None:
        return  # Không có ảnh để chuyển đổi
    if img_edited is None:
        img_edited=img.copy()
    # Chuyển đổi hình ảnh PIL thành np array
    img_np = np.array(img_edited)

    # Trích xuất các kênh màu R, G, B
    r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]

    # Chuyển đổi sang ảnh grayscale
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)
    
    img_edited = Image.fromarray(gray)
    file_size=get_file_size_from_image(img_edited)
    info_label.configure(
        text=f"IMAGE INFO:\n\nColor Space: {img_edited.mode}\n\nImage Size: {img_edited.width}x{img_edited.height}\n\nFile Size: {file_size:.2f} KB"
    )
    # Hiển thị ảnh grayscale
    display_image(img_edited)
    
def gray2bin(threshold):
    global img, img_edited
    if img is None:
        return  # Không có ảnh để chuyển đổi
    
    # Chuyển đổi hình ảnh PIL thành np array
    img_np = np.array(img)

    # Kiểm tra xem ảnh có phải grayscale hay không
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:  # Ảnh RGB
        # Chuyển đổi sang ảnh grayscale (dùng công thức chuẩn)
        gray = (0.2989 * img_np[:, :, 0] + 0.5870 * img_np[:, :, 1] + 0.1140 * img_np[:, :, 2]).astype(np.uint8)
    else:
        # Nếu ảnh đã là grayscale, giữ nguyên
        gray = img_np

    # Áp dụng ngưỡng để chuyển sang ảnh nhị phân
    bin_img = np.where(gray > threshold, 255, 0).astype(np.uint8)

    # Chuyển lại sang ảnh PIL từ mảng np
    img_edited = Image.fromarray(bin_img)

    # Hiển thị ảnh nhị phân
    display_image(img_edited)

   

def apply_sobel(img):
    img_final = np.zeros((len(img), len(img[0])))
    gx = np.zeros((len(img), len(img[0])))
    gy = np.zeros((len(img), len(img[0])))

    for i in range(1, len(img) - 1):  # Duyệt qua chiều cao
        for j in range(1, len(img[0]) - 1):  # Duyệt qua chiều rộng
            gx[i][j] = (img[i - 1][j - 1] + 2*img[i][j - 1] + img[i + 1][j - 1]) - \
                       (img[i - 1][j + 1] + 2*img[i][j + 1] + img[i + 1][j + 1])
            gy[i][j] = (img[i - 1][j - 1] + 2*img[i - 1][j] + img[i - 1][j + 1]) - \
                       (img[i + 1][j - 1] + 2*img[i + 1][j] + img[i + 1][j + 1])
            img_final[i][j] = min(255, np.sqrt(gx[i][j]**2 + gy[i][j]**2))
    
    return img_final

def apply_robert(img):
    img_final = np.zeros((len(img), len(img[0])))  # Khởi tạo ảnh kết quả
    gx = np.zeros((len(img), len(img[0])))  # Gradient theo chiều x
    gy = np.zeros((len(img), len(img[0])))  # Gradient theo chiều y

    # Bộ lọc Robert cho chiều x và y
    for i in range(0, len(img) - 1):  # Không bắt đầu từ pixel đầu tiên
        for j in range(0, len(img[0]) - 1):  # Không bắt đầu từ pixel đầu tiên
            gx[i][j] = img[i][j] - img[i+1][j+1]  # Kernel Gx của Robert
            gy[i][j] = img[i][j+1] - img[i+1][j]  # Kernel Gy của Robert
            img_final[i][j] = min(255, int(np.sqrt(gx[i][j]**2 + gy[i][j]**2)))  # Độ lớn gradient

    return img_final

def apply_prewitt(img, low_threshold=50, high_threshold=150):
    # Khởi tạo ảnh kết quả và gradient theo chiều x, y
    img_final = np.zeros((len(img), len(img[0])))  # Ảnh kết quả (nhị phân)
    gx = np.zeros((len(img), len(img[0])))  # Gradient theo chiều x
    gy = np.zeros((len(img), len(img[0])))  # Gradient theo chiều y

    # Bộ lọc Prewitt cho chiều x và y
    for i in range(1, len(img) - 1):
        for j in range(1, len(img[0]) - 1):
            gx[i][j] = img[i-1][j-1] + img[i][j-1] + img[i+1][j-1] - (img[i-1][j+1] + img[i][j+1] + img[i+1][j+1])
            gy[i][j] = img[i-1][j-1] + img[i-1][j] + img[i-1][j+1] - (img[i+1][j-1] + img[i+1][j] + img[i+1][j+1])
            
            # Độ lớn gradient
            gradient_magnitude = np.sqrt(gx[i][j]**2 + gy[i][j]**2)

            # Áp dụng ngưỡng để tạo ảnh nhị phân
            if gradient_magnitude > high_threshold:
                img_final[i][j] = 255  # Cạnh mạnh
            elif gradient_magnitude > low_threshold:
                img_final[i][j] = 128  # Cạnh yếu
            else:
                img_final[i][j] = 0  # Không phải cạnh

    return img_final


def apply_canny(img, low_threshold=100, high_threshold=200, kernel_size=5, sigma=1.4):

    # Áp dụng thuật toán Canny
    edges = cv2.Canny(img.astype(np.uint8), low_threshold, high_threshold)

    return edges
def custom_canny(image, low_threshold=10, high_threshold= 40, sigma=1.2):  # Increased sigma value
    # Kiểm tra và chuyển ảnh thành np array nếu chưa
    #image = np.array(image.convert("L"), dtype=np.float32)  # Chuyển ảnh sang ảnh xám (grayscale) và định dạng float32

    # Bước 2: Giảm nhiễu bằng bộ lọc Gaussian
    blurred_image = gf(image, sigma=sigma)  # Áp dụng bộ lọc Gaussian để làm mờ ảnh, giảm nhiễu

    # Bước 3: Tính toán Gradient bằng Sobel
    gx = np.zeros_like(image)  # Tạo mảng lưu gradient theo hướng x
    gy = np.zeros_like(image)  # Tạo mảng lưu gradient theo hướng y
    
    # Tính toán gradient theo hướng x và y sử dụng công thức Sobel
    for i in range(1, len(image) - 1):
        for j in range(1, len(image[0]) - 1):
            # Tính gradient theo hướng x (hướng ngang)
            gx[i, j] = (blurred_image[i - 1, j - 1] + 2*blurred_image[i, j - 1] + blurred_image[i + 1, j - 1]) - \
                        (blurred_image[i - 1, j + 1] + 2*blurred_image[i, j + 1] + blurred_image[i + 1, j + 1])
            # Tính gradient theo hướng y (hướng dọc)
            gy[i, j] = (blurred_image[i - 1, j - 1] + 2*blurred_image[i - 1, j] + blurred_image[i - 1, j + 1]) - \
                        (blurred_image[i + 1, j - 1] + 2*blurred_image[i + 1, j] + blurred_image[i + 1, j + 1])

    # Tính độ lớn và hướng của gradient
    magnitude = np.sqrt(gx**2 + gy**2)  # Độ lớn của gradient (magnitude)
    direction = np.arctan2(gy, gx)  # Hướng của gradient (direction)

    # Bước 4: Non-maximum Suppression
    def non_maximum_suppression(magnitude, direction):
        M, N = magnitude.shape  # Lấy kích thước của ảnh
        output = np.zeros((M, N), dtype=np.int32)  # Tạo mảng output để lưu kết quả
        angle = direction * 180. / np.pi  # Chuyển hướng gradient sang độ (degrees)
        angle[angle < 0] += 180  # Chuyển đổi góc từ âm sang dương (0-180 độ)

        # Áp dụng thuật toán Non-maximum Suppression để loại bỏ các điểm không phải là cực đại
        for i in range(1, M-1):
            for j in range(1, N-1):
                q = r = 255  # Giá trị mặc định cho các điểm không phải cực đại
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q, r = magnitude[i, j+1], magnitude[i, j-1]  # Kiểm tra theo hướng ngang
                elif 22.5 <= angle[i, j] < 67.5:
                    q, r = magnitude[i+1, j-1], magnitude[i-1, j+1]  # Kiểm tra theo hướng chéo
                elif 67.5 <= angle[i, j] < 112.5:
                    q, r = magnitude[i+1, j], magnitude[i-1, j]  # Kiểm tra theo hướng dọc
                elif 112.5 <= angle[i, j] < 157.5:
                    q, r = magnitude[i-1, j-1], magnitude[i+1, j+1]  # Kiểm tra theo hướng chéo ngược lại

                # Nếu độ lớn tại điểm ảnh lớn hơn hai điểm lân cận, giữ lại điểm ảnh đó
                output[i, j] = magnitude[i, j] if (magnitude[i, j] >= q and magnitude[i, j] >= r) else 0
        return output

    nms_image = non_maximum_suppression(magnitude, direction)  # Áp dụng Non-maximum Suppression

    # Bước 5: Ngưỡng kép (Double Threshold)
    def double_threshold(image, low_threshold, high_threshold):
        M, N = image.shape  # Lấy kích thước của ảnh
        res = np.zeros((M, N), dtype=np.int32)  # Tạo mảng kết quả
        strong, weak = 255, 50  # Định nghĩa ngưỡng mạnh và yếu

        # Tìm các điểm ảnh có giá trị lớn hơn ngưỡng cao (strong) và ngưỡng thấp (weak)
        strong_i, strong_j = np.where(image >= high_threshold)
        weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))

        # Gán các giá trị mạnh và yếu vào mảng kết quả
        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return res

    thresholded_image = double_threshold(nms_image, low_threshold, high_threshold)  # Áp dụng ngưỡng kép

    # Bước 6: Hysteresis (Kết nối các điểm yếu với các điểm mạnh)
    def hysteresis(image, weak, strong=255):
        M, N = image.shape  # Lấy kích thước của ảnh
        for i in range(1, M-1):
            for j in range(1, N-1):
                if image[i, j] == weak:  # Nếu điểm ảnh có giá trị yếu
                    # Kiểm tra các điểm lân cận nếu có điểm mạnh, giữ lại điểm yếu thành điểm mạnh
                    if ((image[i+1, j-1] == strong) or (image[i+1, j] == strong) or (image[i+1, j+1] == strong)
                        or (image[i, j-1] == strong) or (image[i, j+1] == strong)
                        or (image[i-1, j-1] == strong) or (image[i-1, j] == strong) or (image[i-1, j+1] == strong)):
                        image[i, j] = strong  # Chuyển điểm yếu thành điểm mạnh
                    else:
                        image[i, j] = 0  # Nếu không có điểm mạnh lân cận, gán điểm này thành 0
        return image

    final_image = hysteresis(thresholded_image, weak=50)  # Áp dụng Hysteresis để kết nối các điểm cạnh

    return final_image

def apply_edge(event=None):
    global img, img_edited
    if img is None:
        return  # No image to edit
    
    if img_edited is None:
        img_edited = img

    # Lấy giá trị chọn từ ComboBox
    edge_type = edge_dropdown.get()
    print(edge_type)
    # Tạo cửa sổ con để hiển thị thanh tiến trình
    root = ctk.CTkToplevel()  # Sử dụng CTkToplevel thay vì Toplevel
    root.title("Processing Image")

    # Tính toán vị trí để mở cửa sổ con ở giữa màn hình chính
    screen_width = 1920
    screen_height = 1080
    window_width = 300
    window_height = 50
    position_top = int(screen_height / 2 - window_height / 2)
    position_left = int(screen_width / 2 - window_width / 2)
    root.geometry(f'{window_width}x{window_height}+{position_left}+{position_top}')

    # Tạo thanh tiến trình
    progress = ctk.CTkLabel(root, text="Processing........!")
  
    progress.pack(padx=10, pady=10)
    root.deiconify()  # Hiển thị cửa sổ nếu bị ẩn
    root.attributes('-topmost', True)  # Đưa cửa sổ luôn lên trên
    root.focus_force()  # Đặt focus vào cửa sổ
    root.update() 
    

   
    img_array = np.array(img_edited)

    if len(img_array.shape) == 3:  # Nếu là ảnh màu (RGB), chuyển sang ảnh xám (grayscale)
        img_edited = img_edited.convert("L")
    img_edited = np.array(img_edited, dtype=np.int32)

    # Áp dụng các bộ lọc Edge Detection tương ứng với lựa chọn từ ComboBox
    if edge_type == "Sobel":
        img_edited = apply_sobel(img_edited)
    elif edge_type == "Prewitt":
        img_edited = apply_prewitt(img_edited)
    elif edge_type == "Robert":
        img_edited = apply_robert(img_edited)
    elif edge_type == "Canny Custom":
        img_edited = custom_canny(img_edited)
    elif edge_type == "Canny":
        img_edited = apply_canny(img_edited)
    elif edge_type == "None":
        img_edited = np.array(img)  # Nếu không chọn gì thì giữ nguyên ảnh gốc

    img_edited = Image.fromarray(img_edited)

    # Dừng thanh tiến trình và đóng cửa sổ
    progress.forget()        
    root.destroy()

    # Hiển thị ảnh đã xử lý
    file_size=get_file_size_from_image(img_edited)
    info_label.configure(
        text=f"IMAGE INFO:\n\nColor Space: {img_edited.mode}\n\nImage Size: {img_edited.width}x{img_edited.height}\n\nFile Size: {file_size:.2f} KB"
    )
    display_image(img_edited)
def edge_option():
    clear_tool_space()
    config_label.configure(text="EDGE DETECTION")
    adj_config.pack(fill="x", pady=20)
    config_label.pack(side="top", padx=10, pady=10)
    edge_dropdown.pack(side="top", padx=10, pady=(5,10))

#Bảng lượng tử chuẩn
zigzagOrder = np.array([0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42,
                           49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63])

std_luminance_quant_tbl = np.array(
    [16,  11,  10,  16,  24,  40,  51,  61,
     12,  12,  14,  19,  26,  58,  60,  55,
     14,  13,  16,  24,  40,  57,  69,  56,
     14,  17,  22,  29,  51,  87,  80,  62,
     18,  22,  37,  56,  68, 109, 103,  77,
     24,  35,  55,  64,  81, 104, 113,  92,
     49,  64,  78,  87, 103, 121, 120, 101,
     72,  92,  95,  98, 112, 100, 103,  99], dtype=int)
std_luminance_quant_tbl = std_luminance_quant_tbl.reshape([8, 8])

std_chrominance_quant_tbl = np.array(
    [17,  18,  24,  47,  99,  99,  99,  99,
     18,  21,  26,  66,  99,  99,  99,  99,
     24,  26,  56,  99,  99,  99,  99,  99,
     47,  66,  99,  99,  99,  99,  99,  99,
     99,  99,  99,  99,  99,  99,  99,  99,
     99,  99,  99,  99,  99,  99,  99,  99,
     99,  99,  99,  99,  99,  99,  99,  99,
     99,  99,  99,  99,  99,  99,  99,  99], dtype=int)
std_chrominance_quant_tbl = std_chrominance_quant_tbl.reshape([8, 8])

def quant_tbl_quality_scale(quality):
    if (quality <= 0):
        quality = 1
    if (quality > 100):
        quality = 100
    if (quality < 50):
        qualityScale = 5000 / quality
    else:
        qualityScale = 200 - quality * 2
    luminanceQuantTbl = np.array(np.floor(
        (std_luminance_quant_tbl * qualityScale + 50) / 100))
    luminanceQuantTbl[luminanceQuantTbl == 0] = 1
    luminanceQuantTbl[luminanceQuantTbl > 255] = 255
    luminanceQuantTbl = luminanceQuantTbl.reshape([8, 8]).astype(int)

    chrominanceQuantTbl = np.array(np.floor(
        (std_chrominance_quant_tbl * qualityScale + 50) / 100))
    chrominanceQuantTbl[chrominanceQuantTbl == 0] = 1
    chrominanceQuantTbl[chrominanceQuantTbl > 255] = 255
    chrominanceQuantTbl = chrominanceQuantTbl.reshape([8, 8]).astype(int)
    return luminanceQuantTbl, chrominanceQuantTbl

def rgb2ycbcr(im):
    cbcr = np.empty_like(im)
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]
    # Y
    cbcr[:,:,0] = 16 +  65.481/255 * r + 128.553/255 * g + 24.966/255 * b
    # Cb
    cbcr[:,:,1] = 128 - 37.797/255 * r - 74.203/255 * g + 112.0/255 * b
    # Cr
    cbcr[:,:,2] =  128 +  112.0/255 * r - 93.786/255 * g - 18.214/255 * b
    return np.uint8(cbcr)
quality_Compress=None
def Compress():
    
    global img   
    quality =  int(quality_Compress.get())
    root = ctk.CTkToplevel(app,fg_color='#121212')
    root.title("Processing Image")
    # Lấy kích thước màn hình
    screen_width = 1920
    screen_height = 1080
    window_width = 300
    window_height =50
    position_top = int(screen_height / 2 - window_height / 2)
    position_left = int(screen_width / 2 - window_width / 2)
   # Thiết lập vị trí cửa sổ con
    root.geometry(f"{window_width}x{window_height}+{position_left}+{position_top}")  
    progress = ctk.CTkLabel(root, text="Processing........!")  
    try:
        progress.pack(padx=10, pady=10)
        root.deiconify()  # Hiển thị cửa sổ nếu bị ẩn
        root.attributes('-topmost', True)  # Đưa cửa sổ luôn lên trên
        root.focus_force()  # Đặt focus vào cửa sổ
        root.update()       # Cập nhật giao diện
        # Bắt đầu xử lý
        luminanceQuantTbl, chrominanceQuantTbl = quant_tbl_quality_scale(quality)
        imgMatrix = np.array(img)
        heightImg, withImg, _ = imgMatrix.shape

        # Xử lý padding
        withImg_ = (withImg + 7) // 8 * 8
        heightImg_ = (heightImg + 7) // 8 * 8
        newImgMatrix = imgMatrix.copy()

        # Chuyển đổi sang YCbCr
        newImgMatrix = rgb2ycbcr(newImgMatrix)
        Y_matrix = (newImgMatrix[:, :, 0] - 128).astype(np.int8)
        Cb_matrix = (newImgMatrix[:, :, 1] - 128).astype(np.int8)
        Cr_matrix = (newImgMatrix[:, :, 2] - 128).astype(np.int8)

        # Khởi tạo các mảng dữ liệu
        totalBlock = (withImg_ // 8) * (heightImg_ // 8)
        Y_DC, Cb_DC, Cr_DC = np.zeros(totalBlock, dtype=int), np.zeros(totalBlock, dtype=int), np.zeros(totalBlock, dtype=int)
        d_Y_DC, d_Cb_DC, d_Cr_DC = np.zeros(totalBlock, dtype=int), np.zeros(totalBlock, dtype=int), np.zeros(totalBlock, dtype=int)
        sosBitStream = BitStream()
        currentBlock = 0

        # Duyệt các khối 8x8
        for i in range(0, heightImg_, 8):
            for j in range(0, withImg_, 8):
                # Áp dụng DCT
                Y_DCTMatrix = fftpack.dct(fftpack.dct(Y_matrix[i:i + 8, j:j + 8], norm='ortho').T, norm='ortho').T
                Cb_DCTMatrix = fftpack.dct(fftpack.dct(Cb_matrix[i:i + 8, j:j + 8], norm='ortho').T, norm='ortho').T
                Cr_DCTMatrix = fftpack.dct(fftpack.dct(Cr_matrix[i:i + 8, j:j + 8], norm='ortho').T, norm='ortho').T

                # Lượng tử hóa
                Y_QuantMatrix = np.rint(Y_DCTMatrix / luminanceQuantTbl).astype(int)
                Cb_QuantMatrix = np.rint(Cb_DCTMatrix / chrominanceQuantTbl).astype(int)
                Cr_QuantMatrix = np.rint(Cr_DCTMatrix / chrominanceQuantTbl).astype(int)

                # Zigzag scan
                Y_ZZcode = Y_QuantMatrix.reshape(64)[zigzagOrder]
                Cb_ZZcode = Cb_QuantMatrix.reshape(64)[zigzagOrder]
                Cr_ZZcode = Cr_QuantMatrix.reshape(64)[zigzagOrder]
                Y_DC[currentBlock], Cb_DC[currentBlock], Cr_DC[currentBlock] = Y_ZZcode[0], Cb_ZZcode[0], Cr_ZZcode[0]

                # Xử lý sai khác DC
                if (currentBlock == 0):
                    d_Y_DC[currentBlock] = Y_DC[currentBlock]
                    d_Cb_DC[currentBlock] = Cb_DC[currentBlock]
                    d_Cr_DC[currentBlock] = Cr_DC[currentBlock]
                else:
                    d_Y_DC[currentBlock] = Y_DC[currentBlock] - Y_DC[currentBlock-1]
                    d_Cb_DC[currentBlock] = Cb_DC[currentBlock] - Cb_DC[currentBlock-1]
                    d_Cr_DC[currentBlock] = Cr_DC[currentBlock] - Cr_DC[currentBlock-1]

                # Huffman encoding
                sosBitStream.write(huffmanEncode.encodeDCToBoolList(d_Y_DC[currentBlock], 1), bool)
                huffmanEncode.encodeACBlock(sosBitStream, Y_ZZcode[1:], 1)

                sosBitStream.write(huffmanEncode.encodeDCToBoolList(d_Cb_DC[currentBlock], 0), bool)
                huffmanEncode.encodeACBlock(sosBitStream, Cb_ZZcode[1:], 0)

                sosBitStream.write(huffmanEncode.encodeDCToBoolList(d_Cr_DC[currentBlock], 0), bool)
                huffmanEncode.encodeACBlock(sosBitStream, Cr_ZZcode[1:], 0)

                currentBlock += 1

        # Tạo file JPEG
        outputFile = "compressed_image.jpg"
        with open(outputFile, 'wb') as jpegFile:
            jpegFile.write(huffmanEncode.hexToBytes(
            'FFD8FFE000104A46494600010100000100010000'))

            # write Y Quantization Table
            jpegFile.write(huffmanEncode.hexToBytes('FFDB004300'))
            luminanceQuantTbl = luminanceQuantTbl.reshape([64])
            jpegFile.write(bytes(luminanceQuantTbl.tolist()))
        
            # write u/v Quantization Table
            jpegFile.write(huffmanEncode.hexToBytes('FFDB004301'))
            chrominanceQuantTbl = chrominanceQuantTbl.reshape([64])
            jpegFile.write(bytes(chrominanceQuantTbl.tolist()))
        
            # write height and width
            jpegFile.write(huffmanEncode.hexToBytes('FFC0001108'))
            hHex = hex(heightImg)[2:]
            while len(hHex) != 4:
                hHex = '0' + hHex
        
            jpegFile.write(huffmanEncode.hexToBytes(hHex))
        
            wHex = hex(withImg)[2:]
            while len(wHex) != 4:
                wHex = '0' + wHex
        
            jpegFile.write(huffmanEncode.hexToBytes(wHex))
            # write Subsamp
            jpegFile.write(huffmanEncode.hexToBytes('03011100021101031101'))
        
            # write huffman table
            jpegFile.write(huffmanEncode.hexToBytes('FFC401A20000000701010101010000000000000000040503020601000708090A0B0100020203010101010100000000000000010002030405060708090A0B1000020103030204020607030402060273010203110400052112314151061361227181143291A10715B14223C152D1E1331662F0247282F12543345392A2B26373C235442793A3B33617546474C3D2E2082683090A181984944546A4B456D355281AF2E3F3C4D4E4F465758595A5B5C5D5E5F566768696A6B6C6D6E6F637475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F82939495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA110002020102030505040506040803036D0100021103042112314105511361220671819132A1B1F014C1D1E1234215526272F1332434438216925325A263B2C20773D235E2448317549308090A18192636451A2764745537F2A3B3C32829D3E3F38494A4B4C4D4E4F465758595A5B5C5D5E5F5465666768696A6B6C6D6E6F6475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F839495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA'))
        
            # SOS Start of Scan
            # yDC yAC uDC uAC vDC vAC
            sosLength = sosBitStream.__len__()
            filledNum = 8 - sosLength % 8
            if (filledNum != 0):
                sosBitStream.write(np.ones([filledNum]).tolist(), bool)
        
            # FF DA 00 0C 03 01 00 02 11 03 11 00 3F 00
            jpegFile.write(bytes([255, 218, 0, 12, 3, 1, 0, 2, 17, 3, 17, 0, 63, 0]))
        
            # write encoded data
            sosBytes = sosBitStream.read(bytes)
            for i in range(len(sosBytes)):
                jpegFile.write(bytes([sosBytes[i]]))
                if (sosBytes[i] == 255):
                    jpegFile.write(bytes([0]))  # FF to FF 00
        
            # write end symbol
            jpegFile.write(bytes([255, 217]))  # FF D9
            jpegFile.close()

        # Hiển thị ảnh nén
        compressed_img = Image.open(outputFile)
        display_image(compressed_img)

    except Exception as e:
        print(f"Lỗi: {e}")
    finally:
        progress.forget()        
        root.destroy()

def quality_Option():
    reset_image()
    config_label.configure(text="COMPRESS IMAGE")
    adj_config.pack(fill="x", pady=20)
    config_label.pack(side="top", padx=10, pady=10)
    global crop_frame_Compress, quality_Compress   
    # Tạo frame nhập chất lượng
    crop_frame_Compress = ctk.CTkFrame(adj_config,fg_color="#262726")  # Thay đổi từ Frame sang CTkFrame
    crop_frame_Compress.pack(pady=10)

    # Nhãn và ô nhập giá trị chất lượng  
    ctk.CTkLabel(crop_frame_Compress, text="Enter Quality (0-100):").pack(side="top", padx=5, pady=2)
    quality_Compress = ctk.CTkEntry(crop_frame_Compress,fg_color="#1e1e1e",border_color="#1e1e1e")  # Thay đổi từ tk.Entry sang CTkEntry
    quality_Compress.insert(0, "75")  # Giá trị mặc định là 75
    quality_Compress.pack(padx=5, pady=2)

    #Nút Apply
    btn_apply = ctk.CTkButton(crop_frame_Compress, text="Apply", fg_color="#00aa00", hover_color="#005500",width=10, command=Compress)
    btn_apply.pack(padx=5, pady=(5,10))

def hide_quality_Option():
    global crop_frame_Compress, quality_Compress   
    # Xóa tất cả widget trong crop_frame
    if crop_frame_Compress:
       crop_frame_Compress.destroy()  # Xóa frame và các thành phần bên trong
       crop_frame_Compress = None 

def on_mouse_wheel(event):
    global img, img_edited
    if img is None:
        return

    # Điều chỉnh zoom dựa trên cuộn chuột
    delta = 0.1 if event.delta > 0 else -0.1
    current_zoom = zoom_scale.get() + delta
    current_zoom = max(0.1, min(current_zoom, 3.0))  # Giới hạn từ 0.1 đến 3.0
    zoom_scale.set(current_zoom)

    # Cập nhật kích thước ảnh
    scale_factor = current_zoom
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    print(zoom_scale,"    ",new_width,"   ",new_height)
    # Hiển thị ảnh đã zoom
    img_edited = resized_img
    display_image_zoom(resized_img)   


def crop_image():
    global img, img_edited,x_offset,y_offset,resized_image
    if img is None:
        print("Không có ảnh để chỉnh sửa.")
        return  # Không có ảnh để chỉnh sửa    
    if rect_id is None:
        print("Khung crop chưa được tạo.")
        return  # Không có khung crop để lấy tọa độ

    # Lấy tọa độ khung từ canvas
    coords = canvas.coords(rect_id)
    if not coords:
        print("Không tìm thấy tọa độ khung.")
        return  # Không có tọa độ hợp lệ

    # Chuyển đổi tọa độ canvas sang tọa độ gốc của ảnh
    x1, y1, x2, y2 = map(int, coords)
    x1=x1-x_offset
    y1=y1-y_offset
    x2=x2-x_offset
    y2=y2-y_offset
    # Kiểm tra tọa độ hợp lệ
    if x1 >= x2 or y1 >= y2:
        print("Tọa độ không hợp lệ. Xác minh lại khung crop.")
        return
    try:
        # Cắt ảnh theo tọa độ gốc
        cropped_img = resized_image.crop((x1, y1, x2, y2))

        # Cập nhật ảnh đã cắt
        img_edited = cropped_img
        file_size=get_file_size_from_image(img_edited)
        info_label.configure(
        text=f"IMAGE INFO:\n\nColor Space: {img_edited.mode}\n\nImage Size: {img_edited.width}x{img_edited.height}\n\nFile Size: {file_size:.2f} KB"
    )
        # Hiển thị ảnh đã cắt
        display_image(cropped_img)
        print("Ảnh đã được cắt thành công.")
    except Exception as e:
        print(f"Lỗi khi cắt ảnh: {e}")


   
def show_cut_grid():
    global crop_frame_Crop, crop_top_left_x, crop_top_left_y, crop_bottom_right_x, crop_bottom_right_y,rect_id,displayed_img_coords
    clear_tool_space()
    config_label.configure(text="CROP IMAGE")
    adj_config.pack(fill="x", pady=20)
    config_label.pack(side="top", padx=10, pady=10)

    canvas.bind("<ButtonPress-1>", start_action)  # Nhấn giữ chuột trái
    canvas.bind("<B1-Motion>", perform_action)         # Kéo chuột
    canvas.bind("<ButtonRelease-1>", stop_action)  # Thả chuột trái
    canvas.bind("<MouseWheel>", on_mouse_wheel)  # Cuộn chuột để zoom
    
    if rect_id:
        canvas.delete(rect_id)
        rect_id = None
    x_offset, y_offset, img_width, img_height = displayed_img_coords
    # Tạo khung crop nằm trên ảnh
    crop_margin = 20  # Khoảng cách lề của khung crop so với mép ảnh
    rect_x1 = x_offset + crop_margin
    rect_y1 = y_offset + crop_margin
    rect_x2 = x_offset + img_width - crop_margin
    rect_y2 = y_offset + img_height - crop_margin
    # Tạo khung crop trên canvas
    rect_id = canvas.create_rectangle(rect_x1, rect_y1, rect_x2, rect_y2, outline="white", width=4)
    
    crop_frame_Crop = ctk.CTkFrame(adj_config, fg_color="#262726")  # Frame chứa thông tin tọa độ
    crop_frame_Crop.pack(pady=10)

    # Hiển thị tọa độ góc trên trái
    ctk.CTkLabel(crop_frame_Crop, text="Start X:").pack(side="top", padx=5, pady=2)
    crop_top_left_x = ctk.CTkLabel(crop_frame_Crop, text=rect_x1, fg_color="#1e1e1e", width=80,corner_radius=5)
    crop_top_left_x.pack(padx=5, pady=2)

    ctk.CTkLabel(crop_frame_Crop, text="Start Y:").pack(side="top", padx=5, pady=2)
    crop_top_left_y = ctk.CTkLabel(crop_frame_Crop, text=rect_y1, fg_color="#1e1e1e", width=80,corner_radius=5)
    crop_top_left_y.pack(padx=5, pady=2)

    # Hiển thị tọa độ góc dưới phải
    ctk.CTkLabel(crop_frame_Crop, text="End X:").pack(side="top", padx=5, pady=2)
    crop_bottom_right_x = ctk.CTkLabel(crop_frame_Crop, text=rect_x2, fg_color="#1e1e1e",corner_radius=5, width=80)
    crop_bottom_right_x.pack(padx=5, pady=2)

    ctk.CTkLabel(crop_frame_Crop, text="End Y:").pack(side="top", padx=5, pady=2)
    crop_bottom_right_y = ctk.CTkLabel(crop_frame_Crop, text=rect_y2, fg_color="#1e1e1e",corner_radius=5, width=80)
    crop_bottom_right_y.pack(padx=5, pady=2)
    update_corners()
    # Nút Apply
    btn_apply = ctk.CTkButton(crop_frame_Crop, text="Apply", fg_color="#00aa00", hover_color="#005500", width=10, command=crop_image)
    btn_apply.pack(padx=5, pady=(10,5))


def hide_cut_grid():
    global crop_frame_Crop

    # Xóa tất cả widget trong crop_frame
    if crop_frame_Crop:
        crop_frame_Crop.destroy()
        crop_frame_Crop=None

def show_resize_grid():
    clear_tool_space()
    config_label.configure(text="RESIZE IMAGE")
    adj_config.pack(fill="x", pady=20)
    config_label.pack(side="top", padx=10, pady=10)
    # Tạo khung nhập kích thước cho việc thay đổi kích thước
    global crop_frame_Resize, width, height
    
    crop_frame_Resize = ctk.CTkFrame(adj_config,fg_color="#262726")  # Thay đổi từ Frame sang CTkFrame
    crop_frame_Resize.pack(pady=10)

    ctk.CTkLabel(crop_frame_Resize, text="New width:").pack(side="top", padx=5, pady=2)  # Thay đổi từ Label sang CTkLabel
    width = ctk.CTkEntry(crop_frame_Resize, fg_color="#1e1e1e",border_color="#1e1e1e")  # Thay đổi từ Entry sang CTkEntry
    width.insert(0, "0")  # Giá trị mặc định
    width.pack(padx=5, pady=2)

    ctk.CTkLabel(crop_frame_Resize, text="New height:").pack(side="top", padx=5, pady=2)  # Thay đổi từ Label sang CTkLabel
    height = ctk.CTkEntry(crop_frame_Resize, fg_color="#1e1e1e",border_color="#1e1e1e")  # Thay đổi từ Entry sang CTkEntry
    height.insert(0, "0")  # Giá trị mặc định
    height.pack(padx=5, pady=2)
    
    btn_apply = ctk.CTkButton(crop_frame_Resize, text="Apply",fg_color="#00aa00", hover_color="#005500",width=10, command=resize_image)  # Thay đổi từ Button sang CTkButton
    btn_apply.pack(padx=5, pady=5)
def hide_resize_grid():
    global crop_frame_Resize

    # Xóa tất cả widget trong crop_frame
    if crop_frame_Resize:
        crop_frame_Resize.destroy()
        crop_frame_Resize=None
def resize_image():
    global img, img_edited, info_label,width,height
    
    if img is None:
        return  # Không có ảnh để chỉnh sửa

    # Chỉ khởi tạo giá trị chỉnh sửa từ ảnh gốc lần đầu
    if img_edited is None:
        img_edited = img.copy()
        
    new_width = int(width.get())
    new_height = int(height.get())

    # Thay đổi kích thước ảnh
    img_arr = np.array(img_edited)
    resized_img_arr = cv2.resize(img_arr, (new_width, new_height))
    
    resized_img = Image.fromarray(resized_img_arr)

    # Cập nhật thông tin ảnh
    file_size=get_file_size_from_image(resized_img)
    info_label.configure(
        text=f"IMAGE INFO:\n\nColor Space: {resized_img.mode}\n\nImage Size: {resized_img.width}x{resized_img.height}\n\nFile Size: {file_size:.2f} KB"
    )
    img_edited = resized_img
    # Hiển thị ảnh đã thay đổi kích thước
    
    display_resize(resized_img) 

def keep_one_color(lower_hsv, upper_hsv):
    global img, img_edited
    if img is None:
        return
        
    # Convert the image to HSV
    hsv_image = img.convert("HSV")
    hsv_array = np.array(hsv_image)  # Convert to NumPy array for processing
    
    # Create a mask for the desired color range
    lower_h, lower_s, lower_v = lower_hsv
    upper_h, upper_s, upper_v = upper_hsv
    mask = ((hsv_array[..., 0] >= lower_h) & (hsv_array[..., 0] <= upper_h) &  # Hue range
            (hsv_array[..., 1] >= lower_s) & (hsv_array[..., 1] <= upper_s) &  # Saturation range
            (hsv_array[..., 2] >= lower_v) & (hsv_array[..., 2] <= upper_v))  # Value range
    
    # Convert the original image to grayscale
    gray_image = ImageOps.grayscale(img)
    gray_array = np.array(gray_image)
    gray_3channel = np.stack([gray_array] * 3, axis=-1)  # Convert to 3 channels (RGB)
    
    # Create the final image: keep color where the mask is True, else use grayscale
    original_array = np.array(img)
    final_array = np.where(mask[..., None], original_array, gray_3channel)
    
    # Convert back to an image and save the result
    img_edited = Image.fromarray(final_array.astype("uint8"))
    return img_edited

def submit_number():
    global dialog,hsv_values,hue_l,hue_h,sat_l,sat_h,val_l,val_h
    try:
        # Lấy giá trị từ các Entry
        hsv_values['lower_hue'] = int(hue_l.get())
        hsv_values['higher_hue'] = int(hue_h.get())
        hsv_values['lower_sat'] = int(sat_l.get())
        hsv_values['higher_sat'] = int(sat_h.get())
        hsv_values['lower_val'] = int(val_l.get())
        hsv_values['higher_val'] = int(val_h.get())

        # Tạo cửa sổ nhỏ hiển thị thông báo thành công
        success_window = ctk.CTkToplevel(dialog)
        success_window.title("Success")
        success_window.geometry("300x200")
        success_window.deiconify()  # Hiển thị cửa sổ nếu bị ẩn
        success_window.lift()       # Đưa cửa sổ lên trên cùng
        success_window.focus()  # Đặt focus vào cửa sổ con
        # Nội dung thông báo
        ctk.CTkLabel(success_window, text="Input Received Successfully!", font=("Arial", 14)).pack(pady=10)
        ctk.CTkLabel(success_window, text=f"Lower Hue: {hsv_values['lower_hue']}").pack(pady=2)
        ctk.CTkLabel(success_window, text=f"Higher Hue: {hsv_values['higher_hue']}").pack(pady=2)
        ctk.CTkLabel(success_window, text=f"Lower Saturation: {hsv_values['lower_sat']}").pack(pady=2)
        ctk.CTkLabel(success_window, text=f"Higher Saturation: {hsv_values['higher_sat']}").pack(pady=2)
        ctk.CTkLabel(success_window, text=f"Lower Value: {hsv_values['lower_val']}").pack(pady=2)
        ctk.CTkLabel(success_window, text=f"Higher Value: {hsv_values['higher_val']}").pack(pady=2)

        # Nút đóng cửa sổ
        ctk.CTkButton(success_window, text="Close", command=success_window.destroy).pack(pady=10)
        dialog.destroy()
    except ValueError:
        # Tạo cửa sổ lỗi nếu input không hợp lệ
        error_window = ctk.CTkToplevel(dialog)
        error_window.title("Error")
        error_window.geometry("250x100")
        error_window.deiconify()  # Hiển thị cửa sổ nếu bị ẩn
        error_window.lift()       # Đưa cửa sổ lên trên cùng
        error_window.focus()  # Đặt focus vào cửa sổ con
        # Nội dung thông báo lỗi
        ctk.CTkLabel(error_window, text="Failed: Input must be a number!", text_color="red", font=("Arial", 12)).pack(pady=10)
        ctk.CTkButton(error_window, text="Close", command=error_window.destroy).pack(pady=10)
def keep_color_value():
    clear_tool_space()
    global dialog,hsv_values,hue_l,hue_h,sat_l,sat_h,val_l,val_h
        

    dialog = ctk.CTkToplevel(app)  # Thay đổi từ Toplevel sang CTkToplevel
    dialog.title("Input Number")
    dialog.geometry("400x450")
    dialog.resizable(False, False)
    dialog.deiconify()  # Hiển thị cửa sổ nếu bị ẩn
    dialog.lift()       # Đưa cửa sổ lên trên cùng
    dialog.focus()
    dialog.update()
    # Dictionary to save HSV values
    hsv_values = {}

    # Label and Entry
    ctk.CTkLabel(dialog, text="Enter Hue (Lower):", font=("Arial", 12)).pack(padx=5)  # Thay đổi từ Label sang CTkLabel
    hue_l = ctk.CTkEntry(dialog, font=("Arial", 12))  # Thay đổi từ Entry sang CTkEntry
    hue_l.pack(pady=5)

    ctk.CTkLabel(dialog, text="Enter Hue (Higher):", font=("Arial", 12)).pack(padx=5)  # Thay đổi từ Label sang CTkLabel
    hue_h = ctk.CTkEntry(dialog, font=("Arial", 12))  # Thay đổi từ Entry sang CTkEntry
    hue_h.pack(pady=5)

    ctk.CTkLabel(dialog, text="Enter Saturation (Lower):", font=("Arial", 12)).pack(padx=5)  # Thay đổi từ Label sang CTkLabel
    sat_l = ctk.CTkEntry(dialog, font=("Arial", 12))  # Thay đổi từ Entry sang CTkEntry
    sat_l.pack(pady=5)

    ctk.CTkLabel(dialog, text="Enter Saturation (Higher):", font=("Arial", 12)).pack(padx=5)  # Thay đổi từ Label sang CTkLabel
    sat_h = ctk.CTkEntry(dialog, font=("Arial", 12))  # Thay đổi từ Entry sang CTkEntry
    sat_h.pack(pady=5)

    ctk.CTkLabel(dialog, text="Enter Value (Low):", font=("Arial", 12)).pack(padx=5)  # Thay đổi từ Label sang CTkLabel
    val_l = ctk.CTkEntry(dialog, font=("Arial", 12))  # Thay đổi từ Entry sang CTkEntry
    val_l.pack(pady=5)

    ctk.CTkLabel(dialog, text="Enter Value (Higher):", font=("Arial", 12)).pack(padx=5)  # Thay đổi từ Label sang CTkLabel
    val_h = ctk.CTkEntry(dialog, font=("Arial", 12))  # Thay đổi từ Entry sang CTkEntry
    val_h.pack(pady=5)

    # Submit Button
    ctk.CTkButton(dialog, text="Submit",fg_color='#00aa00',hover_color="#005500", command=submit_number, font=("Arial", 12)).pack(pady=10)  # Thay đổi từ Button sang CTkButton

    dialog.transient(app)  # Set dialog as a child of the main window
    dialog.grab_set()  # Prevent interaction with the main window
    dialog.wait_window()

    lower_hsv = ()
    higher_hsv = ()


    if hsv_values:
        lower_hsv = (hsv_values['lower_hue'], hsv_values['lower_sat'], hsv_values['lower_val'])
        higher_hsv = (hsv_values['higher_hue'], hsv_values['higher_sat'], hsv_values['higher_val'])

        # Call your processing function
        print(f"Lower HSV: {lower_hsv}, Higher HSV: {higher_hsv}")

    # Apply function
    result = keep_one_color(lower_hsv, higher_hsv)
    img_edited = result
    display_image(img_edited)

def apply_noise(event):
    global img_edited, img,noisy_img
    if img is None:
        print("No image available to apply noise.")
        return

    noise_type = noise_dropdown.get()
    noise_level = float(noise_slider.get())   # Lấy giá trị từ thanh trượt

    # Chuyển ảnh PIL sang NumPy array và chuẩn hóa
    img_np = np.array(img, dtype=np.float32) / 255.0
    is_gray = (img.mode == 'L')

    # Áp dụng nhiễu dựa trên loại nhiễu và mức độ
    if noise_type == "Gaussian":
        noisy_img = random_noise(img_np, mode='gaussian', var=noise_level)
    elif noise_type == "Poisson":
        noisy_img = random_noise(img_np, mode='poisson')  
    elif noise_type == "Salt & Pepper":
        noisy_img = random_noise(img_np, mode='s&p', amount=noise_level)
    elif noise_type == "Speckle":
        noisy_img = random_noise(img_np, mode='speckle', var=noise_level)
    else:
        noisy_img = img_np  # Không áp dụng nhiễu

    # Chuyển đổi lại sang uint8
    noisy_img = (noisy_img * 255).clip(0, 255).astype(np.uint8)

    # Chuyển đổi về PIL Image và hiển thị
    img_edited = Image.fromarray(noisy_img, mode='L' if is_gray else 'RGB')
    display_image(img_edited)

def toggle_noise_dropdown():
    clear_tool_space()
    config_label.configure(text="ADD NOISE")
    adj_config.pack(fill="x", pady=20)
    config_label.pack(side="top", padx=10, pady=10)
    if not noise_dropdown.winfo_ismapped():  # Kiểm tra nếu Dropdown chưa hiển thị
       
        noise_dropdown.pack(anchor='n', pady=10)  # Hiển thị Dropdown
        noise_label.pack(anchor='n', pady=5)  # Hiển thị chú thích
        noise_slider.pack(anchor='n', pady=10)  # Hiển thị thanh trượt
    else:
        noise_label.pack_forget()  # Ẩn chú thích nếu đang hiển thị
        noise_dropdown.pack_forget()  # Ẩn Dropdown nếu đang hiển thị
        noise_slider.pack_forget()  # Hiển thị thanh trượt
def clear_noise_space():
    noise_label.pack_forget()  # Ẩn chú thích nếu đang hiển thị
    noise_dropdown.pack_forget()  # Ẩn Dropdown nếu đang hiển thị
    noise_slider.pack_forget()  # Hiển thị thanh trượt

def median_filter(data, kernel_size):
    kernel_halfsize = kernel_size // 2

    # Kiểm tra ảnh màu hay ảnh xám
    if len(data.shape) == 3:  # Ảnh màu có 3 kênh
        data_final = np.zeros_like(data)
        for c in range(data.shape[2]):  # Lọc từng kênh
            data_final[:, :, c] = scipy_median_filter(data[:, :, c], size=kernel_size)
    else:  # Ảnh xám
        height, width = data.shape
        data_final = np.zeros_like(data)
        
        for i in range(height):
            for j in range(width):
                temp = []

                # Traverse the kernel window
                for z in range(-kernel_halfsize, kernel_halfsize + 1):
                    for k in range(-kernel_halfsize, kernel_halfsize + 1):
                        ni, nj = i + z, j + k
                        
                        # Boundary handling: use nearest padding
                        if 0 <= ni < height and 0 <= nj < width:
                            temp.append(data[ni, nj])
                        else:
                            temp.append(data[i, j])  # Padding by nearest value
                
                # Sort and find the median
                temp.sort()
                data_final[i, j] = temp[len(temp) // 2]
        
    return data_final

def mean_filter(data, kernel_size):
    kernel_halfsize = kernel_size // 2
    
    if len(data.shape) == 3:  # Ảnh màu có 3 kênh
        data_final = np.zeros_like(data)
        for c in range(data.shape[2]):  # Lọc từng kênh
            data_final[:, :, c] = uniform_filter(data[:, :, c], size=kernel_size)
    else:  # Ảnh xám (ảnh đơn kênh)
        # Padding dữ liệu để xử lý biên dễ dàng hơn
        padded_data = np.pad(data, ((kernel_halfsize, kernel_halfsize), 
                                    (kernel_halfsize, kernel_halfsize)), 
                             mode='reflect')  # Dùng 'reflect' để làm mềm biên

        # Khởi tạo ảnh kết quả
        data_final = np.zeros_like(data)
        
        # Duyệt qua từng pixel trong ảnh
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                # Trích xuất vùng con trong kernel
                temp_mean = padded_data[i:i + kernel_size, j:j + kernel_size]
                
                # Tính giá trị trung bình của vùng con
                data_final[i, j] = np.mean(temp_mean)
    
    return data_final
def gaussian_filter(data, sigma):
 
    if len(data.shape) == 3:  # Ảnh RGB có 3 kênh
        data_final = np.zeros_like(data)
        for c in range(data.shape[2]):  # Lọc từng kênh màu
            data_final[:, :, c] = gf(data[:, :, c], sigma=sigma)
    else:  # Ảnh xám (1 kênh)
        data_final = gf(data, sigma=sigma)
    
    return data_final
def apply_filter(event):
    global img, img_edited,noisy_img
    if img is None:
        return  # No image to edit
    
    if noisy_img is None:
        noisy_img=img.copy()

    noisy_img = np.array(noisy_img)  # Convert the image to np array
    filter_type = filter_dropdown.get()  # Get the selected filter type

    kernel_size = 3  # Size of the filter kernel

    # Apply filter based on the selection
    if filter_type == "Median Filter":
        img_edited = median_filter(noisy_img, kernel_size)
    elif filter_type == "Mean Filter":
        img_edited = mean_filter(noisy_img, kernel_size)
    elif filter_type == "Gaussian Filter":
        img_edited = gaussian_filter(noisy_img, sigma=2)
    elif filter_type == "None":
        img_edited = np.array(img)  # No filter, keep original image

    # Convert the edited image back to a PIL Image
    img_edited = Image.fromarray(img_edited)

    # Display the filtered image
    display_image(img_edited)
    
def show_filter_set():
    clear_tool_space()
    config_label.configure(text="FILTER NOISE")
    adj_config.pack(fill="x", pady=20)
    config_label.pack(side="top", padx=10, pady=10)
    filter_dropdown.pack(anchor='n', pady=10)

def hide_filter_set():
    filter_dropdown.pack_forget()

def remove_bg():
    clear_tool_space()
    global img, img_edited
    if img is None:
        return
    if img_edited is None:
        img_edited = img.copy()
    root = ctk.CTkToplevel(app,fg_color='#121212')
    root.title("Processing Image")
    # Lấy kích thước màn hình
    screen_width = 1920
    screen_height = 1080
    window_width = 300
    window_height =50
    position_top = int(screen_height / 2 - window_height / 2)
    position_left = int(screen_width / 2 - window_width / 2)
   # Thiết lập vị trí cửa sổ con
    root.geometry(f"{window_width}x{window_height}+{position_left}+{position_top}")  
    progress = ctk.CTkLabel(root, text="Processing........!")
    progress.pack(padx=10, pady=10)
    root.deiconify()  # Hiển thị cửa sổ nếu bị ẩn
    root.attributes('-topmost', True)  # Đưa cửa sổ luôn lên trên
    root.focus_force()  # Đặt focus vào cửa sổ
    root.update()    
    img_edited = remove(img_edited)
    progress.forget()
    root.destroy() 
    display_image(img_edited)

# Biến toàn cục để hỗ trợ kéo ảnh
dragging = False
start_x, start_y = 0, 0
canvas_img = None

def start_drag(event):
    """
    Kích hoạt khi người dùng nhấn chuột trái.
    Lưu vị trí ban đầu để kéo ảnh.
    """
    global dragging, start_x, start_y
    dragging = True
    start_x, start_y = event.x, event.y  # Lưu vị trí chuột

def on_drag(event):
    """
    Xử lý khi người dùng kéo chuột.
    Di chuyển ảnh theo vị trí chuột.
    """
    global dragging, start_x, start_y
    if not dragging:
        return

    # Tính toán khoảng cách di chuyển
    dx, dy = event.x - start_x, event.y - start_y

    # Di chuyển ảnh trên canvas
    canvas.move("all", dx, dy)

    # Cập nhật vị trí mới
    start_x, start_y = event.x, event.y

def stop_drag(event):
    """
    Kích hoạt khi người dùng thả chuột trái.
    Kết thúc trạng thái kéo.
    """
    global dragging
    dragging = False


def auto_resize_canvas():
    # Lấy kích thước màn hình
    screen_width = app.winfo_screenwidth()
    screen_height = app.winfo_screenheight()

    # Điều chỉnh kích thước canvas bằng 50% màn hình
    canvas.config(width=screen_width // 2, height=screen_height // 2)
    
def open_info():
    # Hiển thị hộp thoại thông tin
    messagebox.showinfo(
        title="About Developer",
        message="Developed by:\n Nguyễn Ngọc Hưng\n Lê Tấn Nam \n Lê Nhất Huy \n Vũ Nam Sơn \n Nguyễn Hoà Phúc \n Trần Hữu Nghĩa \n Trần Văn Nghĩa \n- Email: bod_fetel@hcmus.edu.vn \n- Version : 1.0.0 \n- Address: 227 Nguyen Van Cu St., Ward 4, District 5, Ho Chi Minh City"
    )

def open_tutorial():
    pdf_path = os.path.join(os.path.dirname(__file__), "Tutorial.pdf")
    
    if os.path.exists(pdf_path):
        os.startfile(pdf_path)  
    else:
        messagebox.showerror("Error", "Not found, please try again.")


def start_action(event):
    global rect_id, current_action, resizing_corner
    if rect_id is None:
        print("Khung chữ nhật chưa được tạo.")
        return

    coords = canvas.coords(rect_id)  # Lấy tọa độ
    if not coords:
        print("Không tìm thấy tọa độ của khung.")
        return

    # Nếu có tọa độ, tiếp tục xử lý
    x1, y1, x2, y2 = coords
    print(f"Toạ độ khung: ({x1}, {y1}) đến ({x2}, {y2})")

    # Kiểm tra nếu điểm bấm gần các góc để bắt đầu resize
    if abs(event.x - x1) < 20 and abs(event.y - y1) < 20:
        current_action = "resize_crop"
        resizing_corner = "top_left"
    elif abs(event.x - x2) < 20 and abs(event.y - y1) < 20:
        current_action = "resize_crop"
        resizing_corner = "top_right"
    elif abs(event.x - x1) < 20 and abs(event.y - y2) < 20:
        current_action = "resize_crop"
        resizing_corner = "bottom_left"
    elif abs(event.x - x2) < 20 and abs(event.y - y2) < 20:
        current_action = "resize_crop"
        resizing_corner = "bottom_right"
    else:
        print("Không nằm gần góc nào, không thể resize.")

def perform_action( event):
        global rect_id,current_action,resizing_corner
        if current_action == "resize_crop":
            resize_crop(event)
     
def stop_action( event):
    global rect_id,current_action,resizing_corner
    current_action = "none"

def resize_crop( event):
    global rect_id, current_action, resizing_corner
    # Thay đổi kích thước khung crop
    if rect_id:
        x1, y1, x2, y2 = canvas.coords(rect_id)
        if resizing_corner == "top_left":
            canvas.coords(rect_id, event.x, event.y, x2, y2)
        elif resizing_corner == "top_right":
            canvas.coords(rect_id, x1, event.y, event.x, y2)
        elif resizing_corner == "bottom_left":
            canvas.coords(rect_id, event.x, y1, x2, event.y)
        elif resizing_corner == "bottom_right":
            canvas.coords(rect_id, x1, y1, event.x, event.y)
        update_corners()
    

def update_corners():
    global rect_id, current_action, resizing_corner, x1, y1, x2, y2
    clear_corners()
    if rect_id:
        x1, y1, x2, y2 = canvas.coords(rect_id)
        crop_top_left_x.configure(text=x1)
        crop_top_left_y.configure(text=y1)
        crop_bottom_right_x.configure(text=x2)
        crop_bottom_right_y.configure(text=y2)
        r = 5  # Bán kính điểm góc
        corner_circles.append(canvas.create_oval(x1 - r, y1 - r, x1 + r, y1 + r, fill="white"))
        corner_circles.append(canvas.create_oval(x2 - r, y1 - r, x2 + r, y1 + r, fill="white"))
        corner_circles.append(canvas.create_oval(x1 - r, y2 - r, x1 + r, y2 + r, fill="white"))
        corner_circles.append(canvas.create_oval(x2 - r, y2 - r, x2 + r, y2 + r, fill="white"))
def clear_corners():
    global corner_circles
    for circle in corner_circles:
        canvas.delete(circle)
    corner_circles = []


 #####################################################################################



app = ctk.CTk()
#app.geometry("1920x1080")  # Đặt kích thước mặc định nếu cần
ctk.set_appearance_mode("dark")

app.iconbitmap('ps.ico')

app.attributes('-topmost', True)  # Đặt topmost để focus ngay
app.attributes('-topmost', False)  # Trả về trạng thái bình thường


img = None
img_edited = None
crop_frame = None
noisy_img=None
crop_top_left_x = None
crop_top_left_y = None
crop_bottom_right_x = None
crop_bottom_right_y = None
crop_frame_Compress=None
crop_frame_Crop=None
crop_frame_Resize=None  
rect_id = None            # ID của khung crop
corner_circles = []       # Các điểm góc của khung crop
edge_circles = []         # Các điểm cạnh của khung crop
current_action = "none"  # Trạng thái hành động hiện tại
resizing_corner = None
start_x = 0
start_y = 0
x1=None
y1=None
x2=None
y2=None

# Menu chính
menu_bar = ctk.CTkFrame(app, height=30, fg_color="#333333")
menu_bar.pack(side="top", fill="x", pady=2)

# Tạo nút File với menu con
file_image = ImageTk.PhotoImage(Image.open("file.png").resize((30, 30), Image.Resampling.LANCZOS))
file_button = ctk.CTkButton(menu_bar, height=30, image=file_image,text="File",text_color="black", fg_color="#49b0e8", width=70, command=lambda: toggle_file_menu())
file_button.pack(side="left", padx=5)

help_button = ctk.CTkButton(menu_bar, height=30, text="Help", fg_color="#333333", width=50, command=lambda: toggle_help_menu())
help_button.pack(side="left", padx=5)

# Tạo menu con cho File và Help
file_menu_frame = ctk.CTkFrame(app, fg_color="#333333",  corner_radius=10)
file_menu_frame.place_forget()  # Ẩn menu khi khởi động
help_menu_frame = ctk.CTkFrame(app, fg_color="#333333", corner_radius=10)
help_menu_frame.place_forget()  # Ẩn menu khi khởi động

# Tạo trạng thái hiển thị menu
is_file_menu_open = False
is_help_menu_open = False

def toggle_file_menu():
    global is_file_menu_open, is_help_menu_open
    if is_file_menu_open:
        file_menu_frame.place_forget()
        is_file_menu_open = False
    else:
        file_menu_frame.place(x=10, y=40)  # Đặt menu con ở vị trí cố định
        help_menu_frame.place_forget()
        file_menu_frame.lift()  # Đưa menu lên trên
        is_file_menu_open = True
        is_help_menu_open = False

def toggle_help_menu():
    global is_file_menu_open, is_help_menu_open
    if is_help_menu_open:
        help_menu_frame.place_forget()
        is_help_menu_open = False
    else:
        help_menu_frame.place(x=70, y=40)  # Đặt menu con ở vị trí cố định
        file_menu_frame.place_forget()
        help_menu_frame.lift()
        is_help_menu_open = True
        is_file_menu_open = False

 

# Nội dung menu File

ctk.CTkButton(file_menu_frame, text="Open", command=open_image, fg_color="#333333", hover_color="#555555").pack()
ctk.CTkButton(file_menu_frame, text="Save", command=save_image, fg_color="#333333", hover_color="#555555").pack()
ctk.CTkButton(file_menu_frame, text="Exit", command=app.quit, fg_color="#333333", hover_color="#555555").pack()

# Nội dung menu Help
ctk.CTkButton(help_menu_frame, text="Tutorial", command=open_tutorial, fg_color="#333333", hover_color="#555555").pack()
ctk.CTkButton(help_menu_frame, text="About", command=open_info, fg_color="#333333", hover_color="#555555").pack()

# Tạo khung chính để chứa preview, canvas và color space info
main_frame = ctk.CTkFrame(app)
main_frame.pack(fill="both", expand=True)

# Tạo khung xem trước ở góc trên bên trái
preview_frame = ctk.CTkFrame(main_frame, fg_color="#2a2b2a")  # Bo góc và đổi màu
preview_frame.pack(side="left", padx=5, pady=5, fill="y")

ctk.CTkLabel(preview_frame, text="ORIGINAL: ", text_color="white",font=("default", 14, "bold")).pack(pady=5)
preview_label = ctk.CTkLabel(preview_frame,text="", width=150, height=150, fg_color="#262726", corner_radius=10)
preview_label.pack()

# Tạo separator ngang bên dưới preview
horizontal_separator = ctk.CTkFrame(preview_frame, height=2, fg_color="#555555")  # Dùng CTkFrame làm separator
horizontal_separator.pack(fill="x", pady=5)

# Tạo khung chứa nút bên dưới preview
button_frame = ctk.CTkFrame(preview_frame, width=50, fg_color="#2a2b2a", corner_radius=10)
button_frame.pack(pady=5, padx=5, fill="x")  # pack button_frame với chiều rộng giới hạn

# Thêm một nút ví dụ vào button_frame
tune_image = ImageTk.PhotoImage(Image.open("tune.png").resize((35, 35), Image.Resampling.LANCZOS))
reset1_image = ImageTk.PhotoImage(Image.open("reset.png").resize((35, 35), Image.Resampling.LANCZOS))
rotate1_image = ImageTk.PhotoImage(Image.open("Rotate.png").resize((35, 35), Image.Resampling.LANCZOS))
rgb_image = ImageTk.PhotoImage(Image.open("rgb.png").resize((35, 35), Image.Resampling.LANCZOS))
bin_image = ImageTk.PhotoImage(Image.open("binary.png").resize((35, 35), Image.Resampling.LANCZOS))
edge_image = ImageTk.PhotoImage(Image.open("edge.png").resize((35, 35), Image.Resampling.LANCZOS))
comp_image = ImageTk.PhotoImage(Image.open("comp.png").resize((35, 35), Image.Resampling.LANCZOS))
cut_image = ImageTk.PhotoImage(Image.open("cut.png").resize((35, 35), Image.Resampling.LANCZOS))
re_image = ImageTk.PhotoImage(Image.open("resize.png").resize((35, 35), Image.Resampling.LANCZOS))
ex_image = ImageTk.PhotoImage(Image.open("ex.png").resize((35, 35), Image.Resampling.LANCZOS))
noise_image = ImageTk.PhotoImage(Image.open("noise.png").resize((35, 35), Image.Resampling.LANCZOS))
remv_image = ImageTk.PhotoImage(Image.open("remv.png").resize((35, 35), Image.Resampling.LANCZOS))
filter_image = ImageTk.PhotoImage(Image.open("filter.png").resize((35, 35), Image.Resampling.LANCZOS))
# Tạo các nút
btn_reset = ctk.CTkButton(button_frame,image=reset1_image, text="RESET",fg_color="#990000",font=("default", 14, "bold"), command=reset_image,hover_color="#770000")
btn_reset.pack( pady=(2,5))
btn_tune = ctk.CTkButton(button_frame,  image=tune_image,text="Adjust               ",fg_color="#262726",command=show_basic_scale, hover_color="#555555")
btn_tune.pack( pady=2)
btn_rotate_90 = ctk.CTkButton(button_frame,image=rotate1_image, text="Rotate 90°        ", fg_color="#262726", hover_color="#555555", command=rotate_image)
btn_rotate_90.pack(pady=2)
btn_rgb2gray = ctk.CTkButton(button_frame,image=rgb_image, text="RGB To Gray   ", fg_color="#262726", hover_color="#555555", command=rgb2gray)
btn_rgb2gray.pack(pady=2)
btn_gray2bin = ctk.CTkButton(button_frame,image=bin_image, text="Gray To Bin      ", fg_color="#262726", hover_color="#555555", command=bin_scale_control)
btn_gray2bin.pack(pady=2)
btn_edge_detection = ctk.CTkButton(button_frame,image=edge_image, text="Edge Detection", fg_color="#262726", hover_color="#555555", command=edge_option)
btn_edge_detection.pack(pady=2)
btn_compress = ctk.CTkButton(button_frame,image=comp_image, text="Compress         ", fg_color="#262726", hover_color="#555555", command=quality_Option)
btn_compress.pack(pady=2)
btn_crop = ctk.CTkButton(button_frame,image=cut_image, text="Crop Image     ", fg_color="#262726", hover_color="#555555", command=show_cut_grid)
btn_crop.pack(pady=2)
btn_resize = ctk.CTkButton(button_frame,image=re_image, text="Resize Image   ", fg_color="#262726", hover_color="#555555", command=show_resize_grid)
btn_resize.pack(pady=2)
btn_extract_color = ctk.CTkButton(button_frame,image=ex_image, text="Extract Color    ", fg_color="#262726", hover_color="#555555", command=keep_color_value)
btn_extract_color.pack(pady=2)
btn_add_noise = ctk.CTkButton(button_frame,image=noise_image, text="Add Noise         ", fg_color="#262726", hover_color="#555555", command=toggle_noise_dropdown)
btn_add_noise.pack(pady=2)
btn_filter = ctk.CTkButton(button_frame,image=filter_image, text="Filter                  ", fg_color="#262726", hover_color="#555555", command=show_filter_set)
btn_filter.pack(pady=2)
btn_remove_bg = ctk.CTkButton(button_frame,image=remv_image, text="Remove BG      ", fg_color="#262726", hover_color="#555555", command=remove_bg)
btn_remove_bg.pack(pady=2)

canvas = CTkCanvas(main_frame, width=1355, height=996, bg="#262726", highlightbackground="#262726")
canvas.pack(side='left', padx=5, pady=5, fill='both', expand=True)
canvas.bind("<Configure>", on_resize)    
info_frame = ctk.CTkFrame(main_frame, fg_color="#2a2b2a", corner_radius=10)
info_frame.pack(side='right', padx=5, pady=5, fill='y')
adj_config=ctk.CTkFrame(info_frame,fg_color="#262726")
info_label = ctk.CTkLabel(
    info_frame,
    text="IMAGE INFO:\n\n\nColor Space: N/A\n\nImage Size: N/A\n\nFile Size: N/A",
    text_color="white",
    fg_color="#262726",
    corner_radius=10,
    width=200,
    height=150,
    justify="center"
)
info_label.pack(anchor='n', pady=10)
edge_options = ["None", "Sobel", "Prewitt", "Robert", "Canny", "Canny Custom"]
# Tạo ComboBox cho các lựa chọn edge detection và liên kết hàm apply_edge
edge_dropdown = ctk.CTkComboBox(adj_config, values=edge_options, state="readonly", command=apply_edge)

noise_options = ["None", "Gaussian", "Salt & Pepper", "Poisson", "Speckle"]
# Tạo ComboBox cho các lựa chọn edge detection và liên kết hàm apply_edge
noise_dropdown = ctk.CTkComboBox(adj_config, values=noise_options, state="readonly", command=apply_noise)
noise_dropdown.set("None")
noise_slider = ctk.CTkSlider(adj_config, from_=0.00, to=0.1, number_of_steps=10,command=apply_noise)
noise_label = ctk.CTkLabel(adj_config, text="Noise Level")

filter_options = ["None", "Median Filter", "Mean Filter","Gaussian Filter"]
# Tạo ComboBox cho các lựa chọn edge detection và liên kết hàm apply_edge
filter_dropdown = ctk.CTkComboBox(adj_config, values=filter_options, state="readonly", command=apply_filter)
filter_dropdown.set("None") 
zoom_label = ctk.CTkLabel(info_frame, text="Zoom")
zoom_label.pack(side="top", padx=10, pady=2)
zoom_scale = ctk.CTkSlider(info_frame,from_=0.1,to=3.0,number_of_steps=50,command=adjust_zoom)
zoom_scale.set(1.0)
zoom_scale.pack(side="top", padx=10, pady=(2,20))

horizontal_separator1 = ctk.CTkFrame(info_frame, height=2, fg_color="#555555")  # Dùng CTkFrame làm separator
horizontal_separator1.pack(fill="x", pady=10)




brightness_scale = ctk.CTkSlider(adj_config,from_=-1,to=1.0,number_of_steps=10,command=adjust_brightness)  # Giá trị mặc định là 0.0
saturation_scale = ctk.CTkSlider(adj_config,from_=-1,to=1.0,number_of_steps=10,command=adjust_saturation)  # Giá trị mặc định là 0.0
sharpness_scale = ctk.CTkSlider(adj_config,from_=-1,to=1.0,number_of_steps=10,command=adjust_sharpness)  # Giá trị mặc định là 0.0
contrast_scale = ctk.CTkSlider(adj_config,from_=-1,to=1.0,number_of_steps=10,command=adjust_contrast) # Giá trị mặc định là 0.0
bin_scale = ctk.CTkSlider(adj_config,from_=0.0,to=250.0,number_of_steps=25,command=adjust_threshold ) # Giá trị mặc định là 0.0

config_label=ctk.CTkLabel(adj_config,font=("default", 14, "bold"), text="ADJUSTMENT")
brightness_label = ctk.CTkLabel(adj_config, text="BrightNess")
saturation_label = ctk.CTkLabel(adj_config, text="Saturation")
sharpness_label = ctk.CTkLabel(adj_config, text="SharpNess")
contrast_label = ctk.CTkLabel(adj_config, text="Contrast")
bin_label = ctk.CTkLabel(adj_config, text="Threshold")

app.bind("<Control-o>", open_image)
app.bind("<Control-s>", save_image)
# Đặt giá trị mặc định là "None"
edge_dropdown.set("None")
canvas.bind("<ButtonPress-1>", start_drag)  # Nhấn giữ chuột trái
canvas.bind("<B1-Motion>", on_drag)         # Kéo chuột
canvas.bind("<ButtonRelease-1>", stop_drag)  # Thả chuột trái
canvas.bind("<MouseWheel>", on_mouse_wheel) 
    

# zoom_control()
app.update_idletasks()  # Đảm bảo layout đã được vẽ
auto_resize_canvas()


app.mainloop()
