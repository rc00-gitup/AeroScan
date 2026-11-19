import cv2
import glob
import os

# 1. Récupération des deux premières photos du dataset
fichiers = sorted(list(set(glob.glob("./photos_drone/*.*"))))
fichiers = [f for f in fichiers if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if len(fichiers) < 2:
    print("❌ Il faut au moins 2 photos dans ./photos_drone pour ce test.")
    exit()

img1_path, img2_path = fichiers[0], fichiers[1]
print(f"🔍 Test de correspondance entre :\n - Photo A : {os.path.basename(img1_path)}\n - Photo B : {os.path.basename(img2_path)}")

# 2. Chargement et détection des points SIFT
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# 1. On augmente le nombre de points détectés (5000 au lieu de 1000)
sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.03)
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 2. On utilise le ratio standard pour drone en milieu naturel (0.80 au lieu de 0.70)
matcher = cv2.BFMatcher()
matches = matcher.knnMatch(des1, des2, k=2)
bons_points = [m for m, n in matches if m.distance < 0.80 * n.distance]

print(f"✅ {len(bons_points)} points de repère communs identifiés dans la zone de recouvrement !")

# 4. Dessin des lignes de correspondance entre les deux photos
img_debug = cv2.drawMatches(img1, kp1, img2, kp2, bons_points[:80], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

os.makedirs("./static", exist_ok=True)
cv2.imwrite("./static/test_correspondance_qa.jpg", img_debug)
print("🖼️ Image de preuve sauvegardée dans : ./static/test_correspondance_qa.jpg")