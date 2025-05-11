from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans  

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image_array(img_array, block_size=4):

    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    
   
    height, width = img_array.shape[0], img_array.shape[1]
    new_height = ((height + block_size - 1) // block_size) * block_size
    new_width = ((width + block_size - 1) // block_size) * block_size
    
    if new_height != height or new_width != width:
        new_img = np.zeros((new_height, new_width, img_array.shape[2]), dtype=img_array.dtype)
        new_img[:height, :width, :] = img_array
        img_array = new_img
    
    return img_array

def vq_compress(image_path, block_size=4, codebook_size=128, use_minibatch=True):
    image = Image.open(image_path)
  
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    img_array = process_image_array(img_array, block_size)
    
    vectors = []
    for i in range(0, img_array.shape[0], block_size):
        for j in range(0, img_array.shape[1], block_size):
            block = img_array[i:i+block_size, j:j+block_size, :]
            if block.shape[:2] == (block_size, block_size):
                vectors.append(block.flatten())
    
    vectors = np.array(vectors)
    
    if use_minibatch and len(vectors) > 10000:
        kmeans = MiniBatchKMeans(n_clusters=codebook_size, random_state=0, batch_size=1000)
    else:
        kmeans = KMeans(n_clusters=codebook_size, random_state=0)
    
    kmeans.fit(vectors)

    reconstructed = np.zeros_like(img_array)
    idx = 0
    for i in range(0, img_array.shape[0], block_size):
        for j in range(0, img_array.shape[1], block_size):
            if i+block_size > img_array.shape[0] or j+block_size > img_array.shape[1]:
                continue
            
            center = kmeans.cluster_centers_[kmeans.labels_[idx]]
            expected_size = block_size * block_size * img_array.shape[2]
            
            if len(center) == expected_size:
                reconstructed[i:i+block_size, j:j+block_size, :] = \
                    center.reshape(block_size, block_size, img_array.shape[2])
            else:

                min_size = min(len(center), expected_size)
                temp = np.zeros(expected_size)
                temp[:min_size] = center[:min_size]
                reconstructed[i:i+block_size, j:j+block_size, :] = \
                    temp.reshape(block_size, block_size, img_array.shape[2])
            
            idx += 1
    
    return Image.fromarray(reconstructed.astype('uint8')), kmeans

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_' + file.filename)
        file.save(original_path)
        
        try:

            compressed_image, _ = vq_compress(original_path)
            

            compressed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'compressed_' + file.filename)
            compressed_image.save(compressed_path)
            

            original_size = os.path.getsize(original_path)
            compressed_size = os.path.getsize(compressed_path)
            compression_ratio = compressed_size / original_size
            
            return render_template('result.html',
                                 original=original_path,
                                 compressed=compressed_path,
                                 ratio=f"{compression_ratio:.2%}")
        
        except Exception as e:
            return f"error while processing img : {str(e)}"
    
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)