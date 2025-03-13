CSS Animation Generator with AI-Predicted Motion

Description  
This project uses computer vision and deep learning techniques to analyze a sequence of images and predict object motion. Based on the predicted displacements, it generates a smooth CSS animation that replicates the object's movement in a web browser.

Features  
- Motion Calculation: Uses SIFT and BFMatcher to calculate displacement between frames.  
- Motion Prediction: Predicts the next displacement using an LSTM model.  
- CSS Animation Generation: Creates CSS animations with Bezier curves based on predicted movements.  
- HTML Export: Automatically generates an HTML file to visualize the animation in a browser.  

Requirements  
Ensure you have the necessary dependencies installed:  
pip install numpy opencv-python tensorflow scikit-learn

Usage
Place your images as frame1.png, frame2.png, frame3.png, and frame4.png in the working directory.
Run the script:
python main.py
Open the generated optimized_animation_test.html in your browser to view the animation.
How It Works
1. Image Processing: Extracts key points and descriptors from consecutive frames using SIFT.
2. Motion Prediction: Trains an LSTM model to predict future object displacement.
3. CSS Animation: Generates keyframes for smooth animation using Bezier curves.
4. Evaluation: Computes the Mean Squared Error (MSE) between the predicted and actual motion.
Example Output
1. Predicted displacement for the next frame.
2. CSS animation with smooth transitions.
3. An HTML file for easy visualization.
Customization
1. Modify epochs and batch_size in the LSTM model for better accuracy.
2. Adjust the animation speed by changing the animation-duration property in the generated CSS.
License
This project is open-source and available under the MIT License.

Let me know if you want further adjustments!
