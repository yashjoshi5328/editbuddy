from flask import Flask,render_template,request,flash,send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
#path to upload files to
UPLOAD_FOLDER = 'uploads'

#extension that are allowed
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','webp'}

app = Flask(__name__)

app.config['SECRET_KEY']= '4fc1a6e869052df54967658cdd0bcb2e'

#updating upload folder in app configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#variables
old=""
new=""

#return true if file is using allowed extension else false
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#generating a path string for image show
def generate_path_string(image_name):
    image= os.path.join('images', image_name)
    return os.path.join('static', image)

def update_images():
    global old 
    global new
    
    os.remove(f"static/images/{old}")
    old=new

def update_next():
    global old 
    global new
    new=new.split('.')[0][::-1]+'.'+new.split('.')[1]

#homepage
@app.route("/")
def Home():
    print("start")
    return render_template("index.html")


#editpage
@app.route("/edit", methods=["GET","POST"])
def edit():
    if request.method=="POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return "error page"
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return "error page"
        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            #print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            original_image=cv2.imread(UPLOAD_FOLDER+"/"+filename)
            update_image=original_image
            cv2.imwrite(f"static/images/{filename}",update_image)
            global old
            global new
            old=filename[:]
            new=filename.split('.')[0]+"1."+filename.split('.')[1]
            #print(g.next)
            cv2.imwrite(f"static/images/{new}",update_image)
            print(generate_path_string(filename))
            return render_template('layout.html',image=generate_path_string(filename))
        
    return ''''''

@app.route("/download")
def download():
    global new
    return send_from_directory(directory='static/images',path=new,as_attachment=True)

@app.route("/undo")
def undo():
    global old
    global new
    os.remove(f"static/images/{new}")
    image=cv2.imread(f"static/images/{old}")
    cv2.imwrite(f"static/images/{new}",image)
    return render_template('layout.html',image=generate_path_string(new))


@app.route("/crop_4_3")
def crop_4_3():
    global old
    global new
    
    update_images() 
    try:
        image=cv2.imread(generate_path_string(new))

        if image is not None:
            height, width = image.shape[:2]
            # Define the frame's aspect ratio (e.g., 4:3)
            frame_ratio = 4 / 3  # You can change this to your desired ratio
            # Calculate the center coordinates and the new dimensions for cropping
            if width / height > frame_ratio:
                new_width = int(height * frame_ratio)
                new_height = height
                start_x = int((width - new_width) / 2)
                start_y = 0
            else:
                new_width = width
                new_height = int(width / frame_ratio)
                start_x = 0
                start_y = int((height - new_height) / 2)

            # Perform the crop
            cropped_image = image[start_y:start_y + new_height, start_x:start_x + new_width]

            # Save the cropped image
            update_next()
            cv2.imwrite(generate_path_string(new), cropped_image)
        else:
            print("Image could not be loaded. Check the file path.")
        
        return render_template("layout.html",image=generate_path_string(new))
    except Exception as e:
        print(f"An error occurred: {e}")

@app.route("/crop_3_4")
def crop_3_4():
    return crop_4_3()

@app.route("/crop_1_1")
def crop_1_1():
    global old
    global new
    
    update_images() 
    try:
        image=cv2.imread(generate_path_string(new))
            # Get the height and width of the image
        if image is not None:
            height, width, _ = image.shape

            # Find the minimum dimension (either height or width)
            min_dim = min(height, width)

            # Calculate the starting point for the crop
            start_height = (height - min_dim) // 2
            start_width = (width - min_dim) // 2

            # Perform the crop
            cropped_image = image[start_height:start_height + min_dim, start_width:start_width + min_dim]

            # Save the cropped image
            update_next()
            cv2.imwrite(generate_path_string(new), cropped_image)
            
        else:
            print("Image could not be loaded. Check the file path.")
        
        return render_template("layout.html",image=generate_path_string(new))
    except Exception as e:
        print(f"An error occurred: {e}")
   
# Filters
@app.route("/alien")
def alien():
    global old
    global new
    update_images() 
    image=cv2.imread(generate_path_string(new))

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Increase the hue channel to give a blue/cold tone effect
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + 30) % 180  # Increase hue by 30 (adjust based on preference)

    # Convert back to BGR color space
    arctic_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

    update_next()
    cv2.imwrite(generate_path_string(new), arctic_image)
    return render_template("layout.html",image=generate_path_string(new))

@app.route("/bnw")
def bnw():
    global old
    global new
    
    update_images() 
    image=cv2.imread(generate_path_string(new))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    update_next()
    cv2.imwrite(generate_path_string(new), gray_image)
    return render_template("layout.html",image=generate_path_string(new))

@app.route("/predator")
def predator():
    global old
    global new
    
    update_images() 
    image=cv2.imread(generate_path_string(new))

    # Apply a blue tint (cold tone)
    blue_tint = np.array([100, 0, 0], dtype=np.uint8)  # Adjust the values based on your preference
    arctic_image = np.clip(image + blue_tint, 0, 255).astype(np.uint8)

    update_next()
    cv2.imwrite(generate_path_string(new), arctic_image)
    return render_template("layout.html",image=generate_path_string(new))

@app.route("/motion_blur")
def motion_blur():
    global old
    global new
    
    update_images() 
    image=cv2.imread(generate_path_string(new))
    size = 50
    kernel = np.zeros((size,size))
    kernel[int((size-1)/2),:] = np.ones(size)
    kernel = kernel/size

    output  = cv2.filter2D(image, -1, kernel)
    update_next()
    cv2.imwrite(generate_path_string(new), output)
    return render_template("layout.html",image=generate_path_string(new))

@app.route("/water_color")
def water_color():
    #---------------PHASE 1----------------

    global old
    global new
    
    update_images() 
    image=cv2.imread(generate_path_string(new))

    image_cleared = cv2.medianBlur(image, 3)
    image_cleared = cv2.medianBlur(image_cleared, 3)
    image_cleared = cv2.medianBlur(image_cleared, 3)
    image_filtered = cv2.bilateralFilter(image_cleared, 3, 10, 5)
    for i in range(2):
        image_filtered = cv2.bilateralFilter(image_filtered, 3, 20, 10)

    for i in range(3):
        image_filtered = cv2.bilateralFilter(image_filtered, 5, 30, 10)
        gaussian_mask= cv2.GaussianBlur(image_filtered, (7,7), 2)
    image_sharp = cv2.addWeighted(image_filtered, 1.5, gaussian_mask, -0.5, 0)
    image_sharp = cv2.addWeighted(image_sharp, 1.4, gaussian_mask, -0.2, 10)
    # displaying images
    update_next()
    cv2.imwrite(generate_path_string(new), image_sharp)
    return render_template("layout.html",image=generate_path_string(new))


@app.route("/sketch")
def sketch():

    global old
    global new
    
    update_images() 
    image=cv2.imread(generate_path_string(new))

     # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inverted_gray = 255 - gray

    # Apply Gaussian blur to the inverted grayscale image
    blurred = cv2.GaussianBlur(inverted_gray, (21, 21), 0)

    # Invert the blurred image to create a sketch-like effect
    inverted_blurred = 255 - blurred

    # Blend the grayscale image with the inverted blurred image using the "Color Dodge" blending mode
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    # displaying images
    update_next()
    cv2.imwrite(generate_path_string(new), sketch)
    return render_template("layout.html",image=generate_path_string(new))



@app.route("/cartoon_yourself")
def cartoon_yourself():
    global old
    global new
    
    update_images()           
    
    img=cv2.imread(generate_path_string(new))
     # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter for edge-preserving smoothing
    smoothed = cv2.bilateralFilter(img, 10, 30, 30)

    # Find edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    # Combine the smoothed image with edges to create a cartoony effect
    cartoon = cv2.bitwise_and(smoothed, smoothed, mask=edges)

    update_next()
    cv2.imwrite(generate_path_string(new), cartoon)
    return render_template("layout.html",image=generate_path_string(new))

@app.route('/brightness',methods=["GET","POST"])
def brightness():
    global old
    global new         
    
    if request.method=="POST":
        update_images()  
        image=cv2.imread(generate_path_string(new))
        value = request.form.get('value')
        value=int(value)
        print(value)
        for brightness_level in range(1, value+1):
            # Increase brightness using cv2.addWeighted
            brightened_image = cv2.addWeighted(image, 1 + brightness_level / 10.0, np.zeros(image.shape, image.dtype), 0, 0)

        # Display the original and brightened images
        update_next()
        cv2.imwrite(generate_path_string(new),brightened_image)
        return render_template('adjustment.html',image=generate_path_string(new))
    
    return render_template('adjustment.html',image=generate_path_string(new))

@app.route('/contrast',methods=["GET","POST"])
def contrast():
    global old
    global new         
    
    if request.method=="POST":
        update_images()  
        image=cv2.imread(generate_path_string(new))
        value = request.form.get('value')
        value=int(value)
        alpha = 1.5 # Contrast control (1.0-3.0)
        beta = 0 # Brightness control (0-100)
        for _ in range(1,value+1):
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        update_next()
        cv2.imwrite(generate_path_string(new),image)
        return render_template('adjustment.html',image=generate_path_string(new))
    
    return render_template('adjustment.html',image=generate_path_string(new))

@app.route('/sharpness',methods=["GET","POST"])
def sharpness():
    global old
    global new         
    
    if request.method=="POST":
        update_images()  
        image=cv2.imread(generate_path_string(new))
        value = request.form.get('value')
        value=int(value)
        updated_image=image
        sharpeness_filter = np.array([[0, -1, 0], 
                                    [-1, 5, -1], 
                                    [0, -1, 0]])
        for _ in range(0,value,1):
            updated_image=cv2.filter2D(updated_image,-1,sharpeness_filter)
        update_next()
        cv2.imwrite(generate_path_string(new),updated_image)
        return render_template('adjustment.html',image=generate_path_string(new))
    
    return render_template('adjustment.html',image=generate_path_string(new))

@app.route("/noise_reduction")
def noise_reduction():

    global old
    global new
    
    update_images() 
    image=cv2.imread(generate_path_string(new))

    output= cv2.medianBlur(image, 5)

    # displaying images
    update_next()
    cv2.imwrite(generate_path_string(new), output)
    return render_template("layout.html",image=generate_path_string(new))

@app.route('/smoothening',methods=["GET","POST"])
def smoothening():
    global old
    global new         
    
    if request.method=="POST":
        update_images()  
        image=cv2.imread(generate_path_string(new))
        value = request.form.get('value')
        value=int(value)
        if(value%2==0):
            value+=1
        updated_image=image
        #for i in range(0,value,1):
        updated_image=cv2.GaussianBlur(updated_image,ksize=(value,value),sigmaX=0,sigmaY=0)
        update_next()
        cv2.imwrite(generate_path_string(new),updated_image)
        return render_template('adjustment.html',image=generate_path_string(new))
    
    return render_template('adjustment.html',image=generate_path_string(new))

@app.route('/about')
def about():
    return render_template('about.html')
if __name__ == '__main__':
    app.run(debug=True)