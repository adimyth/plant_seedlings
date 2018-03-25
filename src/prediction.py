import os
test_dir = os.path.join("data", "test")
test_imgs = []
test_labels = []
# import os
print("Reading Testing Images...")
files = glob(os.path.join(test_dir, "*.png"))
for file in files:
    img = cv2.imread(file)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    test_imgs.append(img)
test_imgs = np.asarray(test_imgs)


testingImgs = []
for img in test_imgs:
    # Use gaussian blur
    blurImg = cv2.GaussianBlur(img, (5, 5), 0)   
    # Convert to HSV image
    hsvImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)  
    # Create mask (parameters - green color range)
    lower_green = (25, 40, 50)
    upper_green = (75, 255, 255)
    mask = cv2.inRange(hsvImg, lower_green, upper_green)  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Create bool mask
    bMask = mask > 0  
    # Apply the mask
    clear = np.zeros_like(img, np.uint8)  # Create empty image
    clear[bMask] = img[bMask]  # Apply boolean mask to the origin image
    testingImgs.append(clear)  # Append image without background

testingImgs = np.asarray(testingImgs)
testingImgs = testingImgs / 255
  
from keras.models import load_model
# model = load_model("models/trained_model5.h5")
testing_image_numbers = []
for file in files:
  testing_image_numbers.append(file.split('/')[-1])


predictions = model.predict(testingImgs)
# predyclasses = np.argmax()
predictions = label_encodings[np.argmax(predictions,  axis=1)]
print(predictions)

# dictionary = {0:''}  
import pandas as pd
predictions = {'file': testing_image_numbers, 'species': predictions}
predictions = pd.DataFrame(predictions)
predictions.to_csv("predictions.csv", index=False)