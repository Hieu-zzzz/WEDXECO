import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

data = [
    # (gio,  xe_moi_phut, thoi_tiet, toc_do_tb, su_kien, lan_duong, ngay_nghi, nhan)
    (7,  20, 0, 35, 0, 3, 0, 2),
    (8,  15, 1, 30, 1, 3, 0, 2),
    (12,  5, 0, 50, 0, 4, 0, 0),
    (14,  8, 0, 48, 0, 4, 0, 0),
    (18, 12, 1, 32, 0, 3, 0, 1),
    (19, 14, 0, 28, 0, 3, 0, 2),
    (22,  3, 0, 55, 0, 4, 0, 0),
    (9,  10, 1, 36, 1, 3, 0, 1),
    (17, 11, 0, 38, 0, 3, 0, 1),
    (6,   7, 0, 52, 0, 4, 0, 0),
    (21, 13, 1, 30, 0, 3, 0, 2),
]

# Cột dữ liệu và tên hiển thị
columns = [
    "gio", "xe_moi_phut", "thoi_tiet", "toc_do_tb", "su_kien", "lan_duong", "ngay_nghi", "nhan"
]

df = pd.DataFrame(data, columns=columns)

# Từ điển hiển thị
ten_nhan = {0: "Thông thoáng", 1: "Trung bình", 2: "Kẹt xe"}
ten_thoi_tiet = {0: "Nắng", 1: "Mưa"}
ten_su_kien = {0: "Không", 1: "Có"}
ten_ngay_nghi = {0: "Ngày thường", 1: "Ngày nghỉ"}

# Tạo ma trận đặc trưng và nhãn
X = df[[
    "gio", "xe_moi_phut", "thoi_tiet", "toc_do_tb", "su_kien", "lan_duong", "ngay_nghi"
]].values
y = df["nhan"].values

# Pipeline: Chuẩn hóa + Perceptron
model = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("clf", Perceptron(max_iter=1000, random_state=42, tol=1e-3))
])
model.fit(X, y)

# Đánh giá ban đầu
pred_train = model.predict(X)
print(f"Độ chính xác (training): {accuracy_score(y, pred_train):.2f}\n")
print(classification_report(
    y, pred_train,
    target_names=[ten_nhan[0], ten_nhan[1], ten_nhan[2]],
    digits=3
))

# Ma trận nhầm lẫn
cm = confusion_matrix(y, pred_train, labels=[0,1,2])
fig = plt.figure()
plt.imshow(cm, interpolation='nearest')
plt.title("Ma trận nhầm lẫn (training)")
plt.xticks([0,1,2], [ten_nhan[0], ten_nhan[1], ten_nhan[2]])
plt.yticks([0,1,2], [ten_nhan[0], ten_nhan[1], ten_nhan[2]])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.show()

# Widget dự đoán
w_gio = widgets.IntSlider(description="Giờ", min=0, max=23, step=1, value=8)
w_xe  = widgets.IntSlider(description="Xe/phút", min=0, max=50, step=1, value=12)
w_thoi_tiet = widgets.Dropdown(description="Thời tiết", options=[("Nắng", 0), ("Mưa", 1)], value=0)
w_toc_do = widgets.IntSlider(description="Tốc độ TB", min=0, max=80, step=1, value=35)
w_su_kien = widgets.Dropdown(description="Sự kiện", options=[("Không",0),("Có",1)], value=0)
w_lan = widgets.IntSlider(description="Làn đường", min=1, max=6, step=1, value=3)
w_nghi = widgets.Dropdown(description="Ngày nghỉ", options=[("Ngày thường",0),("Ngày nghỉ",1)], value=0)

btn_predict = widgets.Button(description="Dự đoán", button_style="primary")
lbl_ket_qua = widgets.HTML("<b>Kết quả:</b> —")
out = widgets.Output()

def on_predict_clicked(_):
    with out:
        clear_output()
        x = np.array([[
            w_gio.value, w_xe.value, w_thoi_tiet.value,
            w_toc_do.value, w_su_kien.value, w_lan.value, w_nghi.value
        ]])
        yhat = int(model.predict(x)[0])
        lbl_ket_qua.value = f"<b>Kết quả:</b> {yhat} – <b>{ten_nhan.get(yhat, '?')}</b>"
        print(
            f"Đầu vào -> giờ={w_gio.value}, xe/phút={w_xe.value}, thời tiết={ten_thoi_tiet[w_thoi_tiet.value]}, "
            f"tốc độ TB={w_toc_do.value}, sự kiện={ten_su_kien[w_su_kien.value]}, làn đường={w_lan.value}, "
            f"{ten_ngay_nghi[w_nghi.value]}"
        )
        print(f"Nhãn dự đoán: {yhat} ({ten_nhan.get(yhat,'?')})")

btn_predict.on_click(on_predict_clicked)

ui = widgets.VBox([
    widgets.HTML("<h3>Phân loại tình trạng giao thông</h3>"),
    widgets.HBox([w_gio, w_xe, w_thoi_tiet]),
    widgets.HBox([w_toc_do, w_su_kien, w_lan, w_nghi]),
    widgets.HBox([btn_predict, lbl_ket_qua]),
    out
])

display(ui)

# Widget thêm mẫu
w_gio_new = widgets.IntSlider(description="Giờ", min=0, max=23, step=1, value=7)
w_xe_new  = widgets.IntSlider(description="Xe/phút", min=0, max=50, step=1, value=10)
w_thoi_tiet_new = widgets.Dropdown(description="Thời tiết", options=[("Nắng",0), ("Mưa",1)], value=0)
w_toc_do_new = widgets.IntSlider(description="Tốc độ TB", min=0, max=80, step=1, value=30)
w_su_kien_new = widgets.Dropdown(description="Sự kiện", options=[("Không",0),("Có",1)], value=0)
w_lan_new = widgets.IntSlider(description="Làn đường", min=1, max=6, step=1, value=3)
w_nghi_new = widgets.Dropdown(description="Ngày nghỉ", options=[("Ngày thường",0),("Ngày nghỉ",1)], value=0)
w_nhan_new = widgets.Dropdown(description="Nhãn đúng", options=[(ten_nhan[0],0),(ten_nhan[1],1),(ten_nhan[2],2)], value=0)
btn_add = widgets.Button(description="Thêm mẫu + Huấn luyện lại")
out2 = widgets.Output()

def on_add_clicked(_):
    global df, model, X, y
    new_row = {
        "gio": w_gio_new.value,
        "xe_moi_phut": w_xe_new.value,
        "thoi_tiet": w_thoi_tiet_new.value,
        "toc_do_tb": w_toc_do_new.value,
        "su_kien": w_su_kien_new.value,
        "lan_duong": w_lan_new.value,
        "ngay_nghi": w_nghi_new.value,
        "nhan": w_nhan_new.value
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    X = df[[
        "gio", "xe_moi_phut", "thoi_tiet", "toc_do_tb", "su_kien", "lan_duong", "ngay_nghi"
    ]].values
    y = df["nhan"].values
    model.fit(X, y)
    with out2:
        clear_output()
        print("Đã thêm:", new_row)
        print(f"Kích thước dữ liệu hiện tại: {len(df)} dòng")
        print(f"Độ chính xác mới (training): {accuracy_score(y, model.predict(X)):.2f}")
        display(df.tail(10))

btn_add.on_click(on_add_clicked)

display(widgets.VBox([
    widgets.HTML("<h4>Thêm mẫu & Huấn luyện lại (tùy chọn)</h4>"),
    widgets.HBox([w_gio_new, w_xe_new, w_thoi_tiet_new]),
    widgets.HBox([w_toc_do_new, w_su_kien_new, w_lan_new, w_nghi_new]),
    widgets.HBox([w_nhan_new, btn_add]),
    out2
]))
