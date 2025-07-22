# Working Gradio Pneumonia App - Minimal Version
from open_inference.openapi.client import OpenInferenceClient, InferenceRequest
import httpx
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import os

# Configuration - your working values
BASE_URL = 'https://ml-2dad9e26-62f.go01-dem.ylcu-atmi.cloudera.site/namespaces/serving-default/endpoints/pneomonia-classifier'
MODEL_NAME = 'rtcn-dmnq-p6oh-4e8f'

def get_token():
    """Get token from /tmp/jwt file"""
    with open("/tmp/jwt", 'r') as f:
        token_data = json.load(f)
        return token_data["access_token"]

def predict_pneumonia(image):
    """Main prediction function"""
    if image is None:
        return "❌ No image uploaded", "Please upload a chest X-ray image"
    
    try:
        # Get token and setup client
        token = get_token()
        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
        httpx_client = httpx.Client(headers=headers)
        client = OpenInferenceClient(base_url=BASE_URL, httpx_client=httpx_client)
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])
        ])
        
        if image.mode != 'L':
            image = image.convert('L')
        
        tensor_image = transform(image)
        tensor_image = tensor_image.unsqueeze(0)
        numpy_image = tensor_image.numpy()
        
        # Run inference
        inference_request = InferenceRequest(
            inputs=[{
                "name": "input",
                "shape": list(numpy_image.shape),
                "datatype": "FP32",
                "data": numpy_image.flatten().tolist()
            }],
        )
        
        response = client.model_infer(MODEL_NAME, request=inference_request)
        response_dict = json.loads(response.json())
        output_data = response_dict['outputs'][0]['data']
        
        # Process results
        predictions = np.array(output_data).reshape(1, -1)
        probabilities = torch.softmax(torch.tensor(predictions), dim=1).numpy()[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        class_names = ['NORMAL', 'PNEUMONIA']
        predicted_label = class_names[predicted_class]
        
        result = f"🔍 **Prediction: {predicted_label}**\n📊 **Confidence: {confidence:.2%}**"
        details = f"""
**Classification:** {predicted_label}
**Confidence Score:** {confidence:.4f}
**Normal Probability:** {probabilities[0]:.4f} ({probabilities[0]*100:.2f}%)
**Pneumonia Probability:** {probabilities[1]:.4f} ({probabilities[1]*100:.2f}%)
"""
        
        httpx_client.close()
        return result, details
        
    except Exception as e:
        return f"❌ Error: {str(e)}", f"Failed to process image: {str(e)}"

# Create Gradio interface
interface = gr.Interface(
    fn=predict_pneumonia,
    inputs=gr.Image(label="Upload Chest X-Ray Image", type="pil"),
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Textbox(label="Detailed Results", lines=5)
    ],
    title="🫁 Pneumonia Classification",
    description="Upload a chest X-ray image for AI-powered pneumonia detection.",
)

# Launch the app
if __name__ == "__main__":
    # Get the port from the environment variable, default to 8080 if not set
    #port = int(os.environ.get("CDSW_APP_PORT", 8080))
    
    # Launch the app on port 8080, which is typically free.
    interface.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        debug=True,
        quiet=True  # Add this line
    )
# if __name__ == "__main__":
#     interface.launch(
#         server_name="0.0.0.0",  # Listen on all interfaces
#         server_port=int(os.environ.get('CDSW_APP_PORT', 8080)),  # Use CDSW port or default
#         share=False,
#         debug=True
#     )