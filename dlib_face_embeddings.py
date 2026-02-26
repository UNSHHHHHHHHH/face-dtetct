'''
Create face embeddings for all the faces in the dataset/train directory
'''

import pickle
try:
    from imutils import paths
except ImportError:
    import os
    def _list_images(base_path):
        """Recursively list image files in base_path."""
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff','.heic')
        for root, _, files in os.walk(base_path):
            for f in files:
                if f.lower().endswith(extensions):
                    yield os.path.join(root, f)
    paths = type('paths', (), {'list_images': _list_images})
import cv2
import face_recognition
import os
from parameters import DLIB_FACE_ENCODING_PATH, DATASET_PATH

def create_face_embeddings():
    '''
    This function creates face encodings for all the faces in the dataset/train directory
    '''
    imagePaths = list(paths.list_images(DATASET_PATH))
    print(f"[INFO] Found {len(imagePaths)} images in {DATASET_PATH}")

    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}: {}".format(i + 1, len(imagePaths), imagePath))
        name = imagePath.split(os.path.sep)[-2]
        print(f"[INFO] Person: {name}")

        # load the input image
        image = cv2.imread(imagePath)
        if image is None:
            print(f"[WARN] Could not read image: {imagePath}, skipping.")
            continue

        # convert it from BGR (OpenCV ordering) to dlib ordering (RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect face encodings; skip frames where no face is detected
        face_encs = face_recognition.face_encodings(
            image,
            num_jitters=10,  # Higher jitters = more accurate encoding
            model='large'    # 'large' or 'small'
        )
        if len(face_encs) == 0:
            print(f"[WARN] No face detected in: {imagePath}, skipping.")
            continue

        knownEncodings.append(face_encs[0])
        knownNames.append(name)

    print(f"[INFO] Encoded {len(knownEncodings)} face(s) from {len(imagePaths)} image(s).")

    # dump the facial encodings + names to disk
    print("[INFO] Serializing encodings...")
    os.makedirs(os.path.dirname(DLIB_FACE_ENCODING_PATH), exist_ok=True)
    data = {"encodings": knownEncodings, "names": knownNames}
    with open(DLIB_FACE_ENCODING_PATH, "wb") as f:
        f.write(pickle.dumps(data))
    print(f"[INFO] Encodings saved to {DLIB_FACE_ENCODING_PATH}")

if __name__ == '__main__':
    create_face_embeddings()
