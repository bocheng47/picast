import cv2
weights_path = "emotions-recognition/emotions-recognition-retail-0003.bin"
config_path = "emotions-recognition/emotions-recognition-retail-0003.xml"
net = cv2.dnn.readNet(weights_path, config=config_path, framework="DLDT")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
img_path = "angry.jpg"
ref_image = cv2.imread(img_path)
blob = cv2.dnn.blobFromImages(ref_image, 1., (128, 128), True, crop=True)
net.setInput(blob)
res = net.forward()
print(res[0])