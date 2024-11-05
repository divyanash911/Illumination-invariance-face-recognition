from deepface import DeepFace

models = ["VGG-Face", "Facenet", "OpenFace", "DeepID", "Dlib", "ArcFace"]

# verification = DeepFace.verify("img1.jpg", "img2.jpg", model_name = models[1])

recognition = DeepFace.find(img_path = "/home/divyansh/dip/proj/data_new/subject01.centerlight_frame_1.jpg", db_path = "/home/divyansh/dip/proj/data_new", model_name = models[1])

# for i in recognition[0].identity:
#     print(i)

for model in models:
    print(model)
    recognition = DeepFace.find(img_path = "/home/divyansh/dip/proj/data_new/subject01.centerlight_frame_1.jpg", db_path = "/home/divyansh/dip/proj/data_new", model_name = model)
    for i in recognition[0].identity:
        print(i)
    print("\n\n")