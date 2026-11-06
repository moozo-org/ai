import cv2


recognizer = cv2.FaceRecognizerSF.create('models/face_recognition_sface_2021dec.onnx', '')

def getImageFeatures(imgPath):
    img = cv2.imread(imgPath)
    if img is None:
        exit("invalid img path")

    h, w = img.shape[:2]

    detector = cv2.FaceDetectorYN.create('models/face_detection_yunet_2023mar.onnx', '', (w, h))
    faces = detector.detect(img)[1]
    if len(faces) <= 0:
        exit("no faces detected")

    alignedsFaces = []
    for i in range(len(faces)):
        alignedsFaces.append(recognizer.alignCrop(img, faces[i]))
    if len(alignedsFaces) <= 0:
        exit("no aligned faces")

    feats = []
    for embedding in alignedsFaces:
        feats.append(recognizer.feature(embedding))
    if len(feats) <= 0:
        exit('no features')

    return feats

tony1 = 'samples/gallery/tony-blair/02.jpg'
tony2 = 'samples/gallery/tony-blair/03.jpg'
squeezie = 'samples/gallery/squeezie/01.jpg'
greg = 'greg.png'

tony1feats = getImageFeatures(tony1)
tony2feats = getImageFeatures(tony2)
squeeziefeats = getImageFeatures(squeezie)
gregfeats = getImageFeatures(greg)
# print(recognizer.feature(tony1Emb))
# recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE)
print(f'tony1, tony2: {recognizer.match(squeeziefeats[0], gregfeats[0], cv2.FaceRecognizerSF_FR_COSINE)}')