from flask import Flask, render_template, request,url_for,redirect
import os, cv2, numpy as np
from PIL import Image
from ultralytics import YOLO
app=Flask(__name__)




@app.route('/')
def main():
    return render_template('main.html')

@app.route('/home')
def home():
    return render_template('main.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/technology')
def technology():
    return render_template('technology.html')
    
@app.route('/testing',methods=['GET','POST'])
def testing():
     if request.method=='POST':
        files=request.files.getlist('myfile')
        file_extension=[]
        for f in files:
            basepath=os.path.dirname(__file__)
            filepath=os.path.join(basepath,'uploads',f.filename)
            f.save(filepath)
            file_extension.append(f.filename.rsplit('.',1)[1].lower())

        img_directory=os.path.join(basepath,'uploads')
        new_size = (640, 480)

        for filename in os.listdir(img_directory):
            # Check if the file is an image
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):  # Add more image extensions if needed
                image_path = os.path.join(img_directory, filename)
                try:
                    # Open the image
                    with Image.open(image_path) as img:
                        # Resize the image
                        img_resized = img.resize(new_size)
                        # Save the resized image, overwriting the original
                        img_resized.save(image_path)
                        #print(f"Resized {filename} successfully.")
                except Exception as e:
                    print(f"Failed to resize {filename}: {e}")

        output_folder = 'rbccountingimages'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        image_files = os.listdir(img_directory)

# Iterate over each image file
        for image_file in image_files:
    # Read the image
            image_path = os.path.join(img_directory, image_file)
            I = cv2.imread(image_path)
    # Convert the image to grayscale
            Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization
            I_equalized = cv2.equalizeHist(Igray)
    # Apply Gaussian filter
            sigma = 1.5  # Standard deviation of the Gaussian filter
            I_filtered = cv2.GaussianBlur(I_equalized, (0, 0), sigma)
    # Sharpen the image
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            I_sharpened = cv2.filter2D(I_filtered, -1, kernel)
    # Save the grayscale image
            sharpened_image_path = os.path.join(output_folder, f'{image_file}')
            cv2.imwrite(sharpened_image_path, I_sharpened )
        

        output_folder1 = 'pltcountingimages'
        if not os.path.exists(output_folder1):
            os.makedirs(output_folder1)

# Iterate over each image file
        for image_file1 in image_files:
    # Read the image
            image_path = os.path.join(img_directory, image_file1)
            I = cv2.imread(image_path)
    # Convert the image to grayscale
            Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    # Save the grayscale image
            grayscale_image_path = os.path.join(output_folder1, f'{image_file1}')
            cv2.imwrite(grayscale_image_path, Igray )
        
        output_folder2 = 'wbccountingimages'
        if not os.path.exists(output_folder2):
            os.makedirs(output_folder2)
        
        for image_file2 in image_files:
    # Read the image
            image_path = os.path.join(img_directory, image_file2)
            I = cv2.imread(image_path)
    # Split the image into its channels
            B, G, R = cv2.split(I)
    # Equalize the histogram of each channel
            B_eq = cv2.equalizeHist(B)
            G_eq = cv2.equalizeHist(G)
            R_eq = cv2.equalizeHist(R)
    # Merge the equalized channels back into a single image
            I_eq = cv2.merge((B_eq, G_eq, R_eq))
    # Apply Gaussian filter to remove background noise
            sigma = 1.5  # Standard deviation of the Gaussian filter
            I_filtered = cv2.GaussianBlur(I_eq, (0, 0), sigma)
    # Sharpen the image
            I_sharpened = cv2.filter2D(I_filtered, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    # Save the sharpened image
            sharpened_image_path = os.path.join(output_folder2, f'{image_file2}')
            cv2.imwrite(sharpened_image_path, I_sharpened)
        

        #Testing Models
        rbctestingmodel=YOLO('D:/VSC/integrated/venv/models/rbc_model.pt')
        plttestingmodel=YOLO('D:/VSC/integrated/venv/models/plt_model.pt')
        wbctestingmodel=YOLO('D:/VSC/integrated/venv/models/wbc_model.pt')

        #Testing Stage

        rbctestresult=rbctestingmodel("D:/VSC/integrated/rbccountingimages")
        plttestresult=plttestingmodel("D:/VSC/integrated/pltcountingimages")
        wbctestresult=wbctestingmodel("D:/VSC/integrated/wbccountingimages")

        global platelets
        global rbc
        global wbc
        platelets=0
        rbc=0
        wbc=0
        
        for result in rbctestresult:
                label=result.boxes.cls
                labelarray=label.tolist()
                for value in labelarray:
                        if value== 1:
                                rbc+=1

        for result in plttestresult:
                label=result.boxes.cls
                labelarray=label.tolist()
                for value in labelarray:
                        if value == 0:
                                platelets+=1

        for result in wbctestresult:
                label=result.boxes.cls
                labelarray=label.tolist()
                for value in labelarray:
                        if value == 2:
                                wbc+=1

        '''print('\nPlatelets:', platelets)
        print('RBC:', rbc)
        print('WBC:', wbc)'''
       
        return redirect(url_for('results'))
        


        return 'jeo sangi'

     return render_template('testing.html')

@app.route('/results')
def results():
    return render_template('result.html',rbc=rbc,wbc=wbc,platelets=platelets)


    
if __name__=='__main__':
    app.run(debug=True)
