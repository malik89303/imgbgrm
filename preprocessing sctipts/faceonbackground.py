from PIL import Image

for x in range(6,8):
 img = Image.open('backgrounds/'+str(x)+'.jpg')
 path="img1.png"
 mask = Image.open(path)
 (width, height) = mask.size
 resized = img.resize((width, height))
 resized.paste(mask, (0, 0), mask)
 pathout="imagee"+str(x)+".jpg"
 resized.save(pathout)
