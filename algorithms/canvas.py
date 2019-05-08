import numpy as np
from tkinter import *
from PIL import Image, ImageDraw
from feed_forward import feed_forward

ITERATION = 0

def binary(x):
    if x<125:
        return 1
    return 0

def guess():
    with open('../csvFiles/thetas.csv') as thetas_file:
        thetas_lines = thetas_file.readlines()
        theta1 = np.fromstring(thetas_lines[ITERATION*4 + 0], dtype=float, sep=',').reshape(100,785)
        theta2 = np.fromstring(thetas_lines[ITERATION*4 + 1], dtype=float, sep=',').reshape(50,101)
        theta3 = np.fromstring(thetas_lines[ITERATION*4 + 2], dtype=float, sep=',').reshape(10,51)
        theta4 = np.fromstring(thetas_lines[ITERATION*4 + 3], dtype=float, sep=',').reshape(10,11)
        thetas = np.array([theta1, theta2, theta3, theta4])
        content = Image.open('../testimg/image.bmp')
        img = tobinary(np.array(content)[:,:,0].reshape(-1,1))
        A = feed_forward(img,thetas)
        print(A[4])

def save():
    filename = "../testimg/image.bmp"
    image1.save(filename)

def draw(event):
    x, y = event.x, event.y
    if canvas.old_coords:
        x1, y1 = canvas.old_coords
        canvas.create_oval(x, y, x1, y1, fill="black",width=10)
        pil_draw.line([int (x/10), int(y/10), int (x1/10), int(y1/10)], (0,0,0))
    canvas.old_coords = x, y

def release(event):
    canvas.old_coords = None

tobinary = np.vectorize(binary)
root = Tk()
myflag = True
canvas = Canvas(root, width=280, height=280)
canvas.pack()
canvas.old_coords = None
image1 = Image.new("RGB", (28, 28), (255,255,255))
pil_draw = ImageDraw.Draw(image1)

root.bind('<B1-Motion>', draw)
root.bind('<ButtonRelease-1>', release)

button1=Button(text="save",command=save)
button1.pack()
button2=Button(text="guess",command=guess)
button2.pack()

root.mainloop()