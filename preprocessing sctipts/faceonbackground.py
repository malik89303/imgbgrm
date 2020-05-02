from PIL import Image

def createmask(filename1,filename2):
 img = Image.open(filename1)
 path=filename2
 mask = Image.open(path)
 (width, height) = mask.size
 resized = img.resize((width, height))
 resized.paste(mask, (0, 0), mask)
 pathout="e"+filename1+".jpg"
 resized.save(pathout)
