import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import cv2,os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

def assure_path_exists(path): # Tạo các file nếu không tìm thấy
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

##################################################################################

def tick(): # hiển thị giờ theo thời gian thực
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200,tick) # 200 mili giây = 0.2 giây

###################################################################################


def check_haarcascadefile():    # Kiểm tra file haarcascade có tồn tại không
    exists = os.path.isfile("haarcascade_frontalface_alt.xml")
    if exists:
        pass
    else:
        mess._show(title='Some file missing', message='Please check haar file')
        window.destroy()


def clearID(): # Xóa ID
    txt.delete(0, 'end')
    res = ""
    message1.configure(text=res)


def clearName():   # Xóa tên
    txt2.delete(0, 'end')
    res = ""
    message1.configure(text=res)


def check_duplicate_id(new_id):
    # Kiểm tra tệp thông tin sinh viên có tồn tại không
    if os.path.isfile("StudentDetails/StudentDetails.csv"):
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
        # Kiểm tra sự trùng lặp dựa trên ID và NAME
        duplicate = ((df['ID'] == new_id)).any()
        return duplicate
    return False
#######################################################################################


def TakeImages():   #Chụp ảnh lấy dữ liệu
    check_haarcascadefile()
    columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")

    serial = 0
    exists = os.path.isfile("StudentDetails\StudentDetails.csv")
    if exists:
        with open("StudentDetails\StudentDetails.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                serial = serial + 1
        serial = (serial // 2)
        #csvFile1.close()
    else:
        with open("StudentDetails\StudentDetails.csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(columns)
            serial = 1
        #csvFile1.close()
    Id = (txt.get())
    name = (txt2.get().title())


    if (Id.isdigit() and len(Id) > 0) and len(name) > 0 and name.strip() != "" and all(c.isalpha() or c.isspace() for c in name):  # kiểm tra trường nhập hợp lệ
        cam = cv2.VideoCapture(0)  # Mở camera
        harcascadePath = "haarcascade_frontalface_alt.xml"
        detector = cv2.CascadeClassifier(harcascadePath)  # Tạo đối tượng phát hiện khuôn mặt
        sampleNum = 0  # Biến đếm số ảnh chụp
        numPhoto = 100 # Giới hạn ảnh chụp
        while (True):
            ret, img = cam.read() # Đọc khung hình từ camera
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # chuyển ảnh về thang xám để dễ phát hiện khuôn mặt
            faces = detector.detectMultiScale(gray, 1.3, 5) # Phát hiện khuôn mặt trong thang ảnh xám
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum = sampleNum + 1
                # Lưu ảnh đã chụp
                cv2.imwrite("TrainingImage\ " + name + "." + str(serial) + "." + Id + '.' + str(sampleNum) + ".jpg",
                            gray[y:y + h, x:x + w])
                cv2.imshow('Taking photos', img)

            if cv2.waitKey(100) & 0xFF == ord('q'):  # frame chụp ảnh mở trong 100 miliseconds hoăc ấn phím 'q' để thoát
                break
            elif sampleNum > numPhoto: # thoát nếu chụp nhiều hơn numPhoto ảnh
                break
        cam.release()
        cv2.destroyAllWindows() # đóng các cửa sổ hiển thị
        res = "Hoàn tất chụp ảnh"
        row = [serial, '', Id, '', name]
        with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message1.configure(text=res)
    else:
        if len(Id) == 0 and len(name) == 0:
            res = "Vui lòng nhập thông tin"
        elif not Id.isdigit() or len(Id) == 0:
            res = "ID phải là số và không để trống"
        elif len(name) == 0 or name.strip() == "":
            res = "Tên không được bỏ trống"
        elif not any(c.isalpha() for c in name):
            res = "Tên sai ký tự"
        elif not all(c.isalpha() or c.isspace() for c in name):
            res = "Tên chỉ được chứa ký tự chữ và dấu cách"
        else:
            res = "Tên không hợp lệ"
        mess._show(title='Lỗi nhập thông tin', message=res)
        #message1.configure(text=res)
        #mess._show(title='Lỗi nhập thông tin', message=res)
########################################################################################


def TrainImages(): # Huấn liệu dữ liệu ảnh
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")
    recognizer = cv2.face_LBPHFaceRecognizer.create() # Tạo đối tượng nhận diện khuôn mặt sử dụng LBPH
    harcascadePath = "haarcascade_frontalface_alt.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, ID = getImagesAndLabels("TrainingImage")
    try:
        recognizer.train(faces, np.array(ID))
    except:
        mess._show(title='Danh sách trống', message='Vui lòng thêm dữ liệu trước!!!')
        return
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Lưu dữ liệu thành công"
    message1.configure(text=res)
    #message.configure(text='Tổng số sinh viên : ' + str(len(set(ID)) + 1))
    #message.configure(text='Tổng số sinh viên : ' + str(ID[0] + 1))

############################################################################################

def getImagesAndLabels(path):   #Lấy dữ liệu ảnh và nhãn
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)] # Tạo danh sách các đường dẫn ảnh
    faces = [] #Danh sách lưu trữ khuôn mặt
    Ids = [] #Danh sách lưu trữ ID
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L') # mở ảnh và chuyển sang thang xám
        imageNp = np.array(pilImage, 'uint8') # chuyển ảnh về mảng numpy
        ID = int(os.path.split(imagePath)[-1].split(".")[1]) # lấy id ảnh
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids

###########################################################################################

def TrackImages(): # Điểm danh
    check_haarcascadefile()
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")
    for k in tv.get_children():
        tv.delete(k)
    msg = ''
    i = 0
    j = 0
    recognizer = cv2.face.LBPHFaceRecognizer_create() # Tạo đối tượng nhận diện khuôn mặt
    exists3 = os.path.isfile("TrainingImageLabel\Trainner.yml")
    if exists3:
        recognizer.read("TrainingImageLabel\Trainner.yml")
    else:
        mess._show(title='Data Missing', message=' Lưu dữ liệu trước!!')
        return
    harcascadePath = "haarcascade_frontalface_alt.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', '', 'Name', '', 'Date', '', 'Time']
    exists1 = os.path.isfile("StudentDetails\StudentDetails.csv")
    if exists1:
        df = pd.read_csv("StudentDetails\StudentDetails.csv")
    else:
        mess._show(title='Thông tin sinh viên trống', message='Vui lòng kiểm tra thông tin sinh viên!')
        cam.release()
        cv2.destroyAllWindows()
        window.destroy()
    while True:
        ret, im = cam.read() # Đọc khung hình từ camera
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if (conf < 50):  # Nếu độ tin cậy vuợt ngưỡng, ghi thông tin vào lịch sử
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['SERIAL NO.'] == serial]['NAME'].values
                ID = df.loc[df['SERIAL NO.'] == serial]['ID'].values
                ID = str(ID)
                ID = ID[1:-1]
                bb = str(aa)
                bb = bb[2:-2]
                attendance = [str(ID), '', bb, '', str(date), '', str(timeStamp)]

            else:
                Id = 'Unknown'
                bb = str(Id)
            cv2.putText(im, str(bb), (x, y + h), font, 1, (255, 255, 255), 2)
        cv2.imshow('Attendancing', im)
        if (cv2.waitKey(1) == ord('q')):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    exists = os.path.isfile("Attendance\Attendance_" + date + ".csv")
    if exists:
        with open("Attendance\Attendance_" + date + ".csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(attendance) # Ghi thông tin lịch sử điểm danh vào tệp
        #csvFile1.close()
    else:
        with open("Attendance\Attendance_" + date + ".csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(col_names)
            writer.writerow(attendance)
        #csvFile1.close()

    with open("Attendance\Attendance_" + date + ".csv", 'r') as csvFile1: # Ghi lịch sử lên màn hình
        reader1 = csv.reader(csvFile1)
        for lines in reader1:
            i = i + 1
            if (i > 1): # bỏ qua dòng đầu vì ghi tên thuộc tính
                if (i % 2 != 0): # Ghi ID và các giá trị
                    id_values = str(lines[0]) + '   '
                    tv.insert('', 0, text=id_values, values=(str(lines[2]), str(lines[4]), str(lines[6])))
    #csvFile1.close()
    cam.release()
    cv2.destroyAllWindows()

######################################## USED STUFFS ############################################
    
global key
key = ''

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
day,month,year=date.split("-")


######################################## GUI FRONT-END ###########################################

window = tk.Tk()
window.geometry("1280x720")
window.resizable(True,False)
window.title("PHẦN MỀM ĐIỂM DANH SINH VIÊN")
window.configure(background='#ffffff')

frame2 = tk.Frame(window, bg="#62D9FA")
frame2.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)

frame1 = tk.Frame(window, bg="#62D9FA")
frame1.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.80)

message3 = tk.Label(window, text="ĐIỂM DANH SINH VIÊN" ,fg="Red",bg="#318BF7" ,width=55 ,height=1,font=('comic', 29, ' bold '))
message3.place(x=9, y=10)

frame3 = tk.Frame(window, bg="#c4c6ce")
frame3.place(relx=0.52, rely=0.09, relwidth=0.09, relheight=0.07)

frame4 = tk.Frame(window, bg="#c4c6ce")
frame4.place(relx=0.36, rely=0.09, relwidth=0.16, relheight=0.07)

datef = tk.Label(frame4, text ="       "+day+"/"+month+"/"+year+"    ", fg="red",bg="#ffffff" ,width=55 ,height=1,font=('comic', 22, ' bold '))
datef.pack(fill='both',expand=1)

clock = tk.Label(frame3,fg="red",bg="#ffffff" ,width=55 ,height=1,font=('comic', 22, ' bold '))
clock.pack(fill='both',expand=1)
tick()

head2 = tk.Label(frame1, text="                       Thêm Dữ Liệu Mới                       ", fg="black",bg="#2a07f0" ,font=('comic', 17, ' bold ') )
head2.grid(row=0,column=0)

head1 = tk.Label(frame2, text="                       Điểm Danh Lớp Học                       ", fg="black",bg="#2a07f0" ,font=('comic', 17, ' bold ') )
head1.place(x=0,y=0)

lbl = tk.Label(frame1, text="Nhập ID",width=20  ,height=1  ,fg="black"  ,bg="#62D9FA" ,font=('comic', 17, ' bold ') )
lbl.place(x=80, y=55)

txt = tk.Entry(frame1,width=32 ,fg="black",font=('comic', 15, ' bold '))
txt.place(x=30, y=88)

lbl2 = tk.Label(frame1, text="Nhập họ và tên",width=20  ,fg="black"  ,bg="#62D9FA" ,font=('comic', 17, ' bold '))
lbl2.place(x=80, y=140)

txt2 = tk.Entry(frame1,width=32 ,fg="black",font=('comic', 15, ' bold ')  )
txt2.place(x=30, y=173)

message1 = tk.Label(frame1, text="" ,bg="#62D9FA" ,fg="black"  ,width=39 ,height=1, activebackground = "#3ffc00" ,font=('comic', 15, ' bold '))
message1.place(x=7, y=230)

message = tk.Label(frame1, text="" ,bg="#62D9FA" ,fg="black"  ,width=39,height=1, activebackground = "#3ffc00" ,font=('comic', 16, ' bold '))
message.place(x=7, y=450)

lbl3 = tk.Label(frame2, text="Lịch sử điểm danh",width=20  ,fg="black"  ,bg="#62D9FA"  ,height=1 ,font=('comic', 17, ' bold '))
lbl3.place(x=100, y=115)

##################### MENUBAR #################################

menubar = tk.Menu(window,relief='ridge')


################## TREEVIEW ATTENDANCE TABLE ####################

# Hiển thị lịch sử điểm danh
tv= ttk.Treeview(frame2,height =13,columns = ('name','date','time')) # 82, 130, 133, 133
tv.column('#0',width=130,  anchor='center')
tv.column('name',width=130,  anchor='center')
tv.column('date',width=100,  anchor='center')
tv.column('time',width=100,  anchor='center')
tv.grid(row=2,column=0,padx=(0,0),pady=(150,0),columnspan=4)
tv.heading('#0',text ='ID')
tv.heading('name',text ='NAME')
tv.heading('date',text ='DATE')
tv.heading('time',text ='TIME')

###################### SCROLLBAR ################################

scroll=ttk.Scrollbar(frame2,orient='vertical',command=tv.yview)
scroll.grid(row=2,column=4,padx=(0,100),pady=(150,0),sticky='ns')
tv.configure(yscrollcommand=scroll.set)

###################### BUTTONS ##################################

clearButton = tk.Button(frame1, text="Xóa", command=clearID  ,fg="black"  ,bg="#ff7221"  ,width=11 ,activebackground = "white" ,font=('comic', 11, ' bold '))
clearButton.place(x=335, y=86)
clearButton2 = tk.Button(frame1, text="Xóa", command=clearName  ,fg="black"  ,bg="#ff7221"  ,width=11 , activebackground = "white" ,font=('comic', 11, ' bold '))
clearButton2.place(x=335, y=172)
takeImg = tk.Button(frame1, text="Chụp ảnh", command=TakeImages  ,fg="black"  ,bg="#3ffc00"  ,width=34  ,height=1, activebackground = "white" ,font=('comic', 15, ' bold '))
takeImg.place(x=30, y=300)
trainImg = tk.Button(frame1, text="Lưu dữ liệu", command=TrainImages ,fg="black"  ,bg="#3ffc00"  ,width=34  ,height=1, activebackground = "white" ,font=('comic', 15, ' bold '))
trainImg.place(x=30, y=380)
trackImg = tk.Button(frame2, text="Điểm danh", command=TrackImages  ,fg="black"  ,bg="#3ffc00"  ,width=35  ,height=1, activebackground = "white" ,font=('comic', 15, ' bold '))
trackImg.place(x=30,y=50)
quitWindow = tk.Button(frame2, text="Thoát", command=window.destroy  ,fg="black"  ,bg="#eb4600"  ,width=35 ,height=1, activebackground = "white" ,font=('comic', 15, ' bold '))
quitWindow.place(x=30, y=450)



window.configure(menu=menubar)
window.mainloop()


