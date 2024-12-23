import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

eigenfaces = np.load("basis.npy")
with open("vars.pkl", "rb") as f:
    M, scaler = pickle.load(f)

def decompose(img, number_of_basis = 50):
    original_shape = img.shape
    img = img.reshape(-1)
    img -= M
    img /= scaler
    coeffecients = np.array([])
    for ef in eigenfaces[:number_of_basis]:
        coeffecients = np.append(coeffecients, np.dot(img, ef))
        
    comp = np.zeros(img.shape)
    for i in range(number_of_basis):
        comp += coeffecients[i] * eigenfaces[i]
        
    comp *= scaler
    comp += M

    return comp.reshape(original_shape), coeffecients

registered_path = [
    "samples/11.jpg",
    "samples/21.jpeg",
    "samples/31.jpeg",
    "samples/41.jpeg",
    "samples/51.jpeg",
    "samples/61.jpeg",
    "samples/71.jpeg",
    "samples/81.jpeg",
    "samples/91.jpeg",
    "samples/101.jpeg",
    "samples/111.jpeg"
]
test_path = [
    "samples/12.jpg",
    "samples/22.jpeg",
    "samples/32.jpeg",
    "samples/42.jpeg",
    "samples/52.jpeg",
    "samples/62.jpeg",
    "samples/72.jpeg",
    "samples/82.jpeg",
    "samples/92.jpeg",
    "samples/102.jpeg",
    "samples/112.jpeg"
]
registered_images = []
for path in registered_path:
    img = cv2.imread(path)
    img = cv2.resize(img, (47, 62))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    registered_images.append(img)

registered_images = np.array(registered_images)
registered_images = registered_images.astype(np.float64)
registered_images /= 255/2
registered_images -= 1

test_images = []
for path in test_path:
    img = cv2.imread(path)
    img = cv2.resize(img, (47, 62))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    test_images.append(img)
    
test_images = np.array(test_images)
test_images = test_images.astype(np.float64)
test_images /= 255/2
test_images -= 1

# 1
number_of_basis = [1, 5, 10, 50, 100, 300, 500, 1000, 2000]
coef = np.array([])
for n in number_of_basis:
    comp, coef = decompose(registered_images[0], n)
    plt.imsave(f"results/1_{n}.jpg", comp, cmap="gray")
    
print(coef)

# 2
registered_coef = []
for img in registered_images:
    _, coef = decompose(img, 200)
    registered_coef.append(coef)
    
for idx, ti in enumerate(test_images):
    max_idx = 0
    prev = -1
    _, coef = decompose(ti, 200)
    for i in range(registered_images.shape[0]):
        cos_sim = np.dot(coef, registered_coef[i])
        cos_sim /= (np.linalg.norm(coef) * np.linalg.norm(registered_coef[i]))
        if cos_sim > prev:
            max_idx = i + 1
            prev = cos_sim
            
        print(cos_sim)
        
    print(idx + 1, max_idx)
    print()