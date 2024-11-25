import cv2

image_path = 'cat.jpeg'
mask_path = 'dog-7128749_640.jpg'


image = cv2.imread(image_path)
mask = cv2.imread(mask_path)

if image is None:
    print(f"Не вдалося завантажити зображення з шляху: {image_path}")
    exit()

if mask is None:
    print(f"Не вдалося завантажити маску з шляху: {mask_path}")
    exit()

cat_face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')

cat_faces = cat_face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in cat_faces:
    mask_resized = cv2.resize(mask, (w, h))
    image[y:y+h, x:x+w] = mask_resized
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

cv2.imshow("Cat with Mask", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
