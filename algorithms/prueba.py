import os

path = '../images/'

files = []
# r=root, d=directories, f = files
for f in os.listdir(path):
    print(f)

print(len(os.listdir(path)))
