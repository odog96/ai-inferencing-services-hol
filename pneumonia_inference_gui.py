from open_inference.openapi.client import OpenInferenceClient, InferenceRequest
import httpx
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import pandas as pd
import time
from typing import Optional, Dict, Any, Tuple
import io
import base64
import os


class PneumoniaInferenceClient:
    """Client for pneumonia classification inference using Open Inference protocol"""
    
    def __init__(self, base_url: str, model_name: str, token: str):
        self.base_url = base_url
        self.model_name = model_name
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        self.httpx_client = httpx.Client(headers=self.headers)
        self.client = OpenInferenceClient(base_url=base_url, httpx_client=self.httpx_client)
        
        # Initialize image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])  # Grayscale normalization
        ])
        
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess uploaded image for inference"""
        try:
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            
            # Apply transforms
            tensor_image = self.transform(image)
            
            # Add batch dimension and convert to numpy
            tensor_image = tensor_image.unsqueeze(0)  # Shape: (1, 1, 224, 224)
            numpy_image = tensor_image.numpy()
            
            return numpy_image
            
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")
    
    def check_server_status(self) -> Tuple[bool, str]:
        """Check if the server is ready and get model metadata"""
        try:
            # Check server readiness
            self.client.check_server_readiness()
            status_msg = "✓ Server is ready"
            
            # Get model metadata
            metadata = self.client.read_model_metadata(self.model_name)
            metadata_dict = json.loads(metadata.json())
            
            model_info = f"""
Model Information:
- Name: {metadata_dict.get('name', 'Unknown')}
- Platform: {metadata_dict.get('platform', 'Unknown')}
- Inputs: {len(metadata_dict.get('inputs', []))}
- Outputs: {len(metadata_dict.get('outputs', []))}
"""
            
            return True, status_msg + model_info
            
        except Exception as e:
            return False, f"Error checking server status: {str(e)}"
    
    def run_inference(self, image: Image.Image) -> Dict[str, Any]:
        """Run inference on a single chest X-ray image"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Create inference request
            inference_request = InferenceRequest(
                inputs=[{
                    "name": "input",  # Adjust based on your model's input name
                    "shape": list(processed_image.shape),
                    "datatype": "FP32",
                    "data": processed_image.flatten().tolist()
                }],
            )
            
            # Run inference
            start_time = time.time()
            response = self.client.model_infer(self.model_name, request=inference_request)
            inference_time = time.time() - start_time
            
            # Extract predictions from response
            response_dict = json.loads(response.json())
            output_data = response_dict['outputs'][0]['data']
            
            # Process predictions
            predictions = np.array(output_data).reshape(1, -1)  # Shape: (1, 2)
            probabilities = torch.softmax(torch.tensor(predictions), dim=1).numpy()[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
            
            # Map predictions to class names
            class_names = ['NORMAL', 'PNEUMONIA']
            predicted_label = class_names[predicted_class]
            
            return {
                'prediction': predicted_label,
                'confidence': float(confidence),
                'probabilities': {
                    'NORMAL': float(probabilities[0]),
                    'PNEUMONIA': float(probabilities[1])
                },
                'inference_time': inference_time,
                'success': True
            }
            
        except Exception as e:
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'probabilities': {'NORMAL': 0.0, 'PNEUMONIA': 0.0},
                'inference_time': 0.0,
                'error': str(e),
                'success': False
            }
    
    def close(self):
        """Close the HTTP client"""
        self.httpx_client.close()


# Global client instance
client = None


def load_token(token_path: str = "/tmp/jwt") -> str:
    """Load CDP token from file with fallback"""
    try:
        # Try to read from JWT file first
        with open(token_path, 'r') as f:
            content = f.read().strip()
            
            # Check if it's HTML (error page)
            if content.startswith('<html>') or content.startswith('<!DOCTYPE'):
                print("JWT file contains HTML error, using fallback token")
                raise ValueError("JWT file contains HTML error")
                
            # Try to parse as JSON
            token_data = json.loads(content)
            print("✅ Using JWT token from file")
            return token_data["access_token"]
            
    except Exception as e:
        print(f"⚠️ JWT file error: {e}")
        print("🔄 Using hardcoded fallback token")
        
        # Your hardcoded token as fallback
        fallback_token = "eyJraWQiOiIzYzhlNzA3OTEyZmI0NTA1ODE3NzE3YzMyOTU4MmQwMTFjYjlmNTAwIiwidHlwIjoiSldUIiwiYWxnIjoiUlMyNTYifQ.eyJzdWIiOiJvemFyYXRlIiwiYXVkIjoiaHR0cHM6Ly9kZS55bGN1LWF0bWkuY2xvdWRlcmEuc2l0ZSIsImlzcyI6Imh0dHBzOi8vY29uc29sZWF1dGguY2RwLmNsb3VkZXJhLmNvbS84YTFlMTVjZC0wNGMyLTQ4YWEtOGYzNS1iNGE4YzExOTk3ZDMiLCJncm91cHMiOiJjZHBfZGVtb3Nfd29ya2Vyc193dyBjZHBfZGVtby1hd3MtcHJpbSBfY19kZl9kZXZlbG9wXzkxMTQ2M2MgX2NfbWxfYWRtaW5zXzViNTQ3ZDI2IF9jX21sX2J1c2luZXNzX3VzZXJzXzc3ODkxZjJlIF9jX2RmX3ZpZXdfNmY1OWU5ZjMgX2NfZGZfYWRtaW5pc3Rlcl85MTE0NjNjIF9jX21sX3VzZXJzXzc3ODkxZjJlIF9jX2RmX3ZpZXdfOTExNDYzYyBfY19tbF9idXNpbmVzc191c2Vyc182ZWUwZGI5MSBfY19kZl9wdWJsaXNoXzkxMTQ2M2MgX2NfZGZfdmlld185MTE0NjNjMCBfY19lbnZfYXNzaWduZWVzXzkxMTQ2M2MgX2NfcmFuZ2VyX2FkbWluc185MDZiMGJhIF9jX21sX3VzZXJzXzZmNTllOWYzIF9jX21sX3VzZXJzXzRkODNhZDdmIF9jX2Vudl9hc3NpZ25lZXNfOTA2YjBiYSBfY19kZl9kZXZlbG9wXzZmNTllOWYzIF9jX2RmX2FkbWluaXN0ZXJfNmY1OWU5ZjMgX2NfZGZfcHVibGlzaF82ZjU5ZTlmMyBfY19tbF91c2Vyc182ZWUwZGI5MSBfY19tbF9idXNpbmVzc191c2Vyc182ZjU5ZTlmMyBfY19tbF9idXNpbmVzc191c2Vyc185MTE0NjNjIF9jX21sX2FkbWluc183Nzg5MWYyZSBfY19lbnZfYXNzaWduZWVzXzZmNTllOWYzIF9jX3Jhbmdlcl9hZG1pbnNfNmY1OWU5ZjMgX2NfcmFuZ2VyX2FkbWluc185MTE0NjNjIF9jX2RlX3VzZXJzXzkxMTQ2M2MgX2NfbWxfdXNlcnNfOTExNDYzYyBfY19kZl9wcm9qZWN0X21lbWJlcl80MGRmZTU2OCBfY19kZl92aWV3XzZmNTllOWYzMCBfY19kZl9wcm9qZWN0X21lbWJlcl81NzVmODRmNyBfY19kZV91c2Vyc182ZjU5ZTlmMyIsImV4cCI6MTc1MzExNTgxNywidHlwZSI6InVzZXIiLCJnaXZlbl9uYW1lIjoiT2xpdmVyIiwiaWF0IjoxNzUzMTEyMjE3LCJmYW1pbHlfbmFtZSI6IlphcmF0ZSIsImVtYWlsIjoib3phcmF0ZUBjbG91ZGVyYS5jb20ifQ.hodrdbHGmPEdU4d-ZcZo5blNmrYdkkH43M1iwdQdNby7O0m3Bu-gDP6mFe508X_l6vk-4zCjmO1AJnVjc3qIEE38Vqp8HFsU_j831w_jK1k7IN_UFjzhYc8IADcSmWjazu3IAFL4wnEDT0M0uHCJVhIMY3g1QCiptTWOH6QQApDcSocK2FyOX4rTemIofkgo2-GxMKhsYsUMaIrCD-1hc3ZMRZQQ7dyl4LKK81pAmwC-4sVzbilaYC73jAXGCop7HskVmD6N5hCTYdUVBUq1lUoRPvjfWAoP3GfiDFvlFpaLEkrurIGCZnFdXQKEPXjhXzrqKBbt0YLI89nWtvq-WQ"
        return fallback_token


def initialize_client(base_url: str, model_name: str) -> Tuple[bool, str]:
    """Initialize the inference client"""
    global client
    
    try:
        token = load_token()
        client = PneumoniaInferenceClient(base_url, model_name, token)
        
        # Check server status
        is_ready, status_msg = client.check_server_status()
        
        if is_ready:
            return True, f"✅ Client initialized successfully!\n{status_msg}"
        else:
            return False, f"❌ Server not ready: {status_msg}"
            
    except Exception as e:
        return False, f"❌ Failed to initialize client: {str(e)}"


def predict_pneumonia(image: Image.Image, base_url: str, model_name: str) -> Tuple[str, str, Dict, str]:
    """Main prediction function for Gradio interface"""
    global client
    
    # Validate inputs
    if image is None:
        empty_chart = pd.DataFrame({"Class": [], "Probability": []})
        return "❌ No image uploaded", "Please upload a chest X-ray image", empty_chart, ""
    
    if not base_url or not model_name:
        empty_chart = pd.DataFrame({"Class": [], "Probability": []})
        return "❌ Configuration Error", "Please provide base URL and model name", empty_chart, ""
    
    try:
        # Initialize client if not already done
        if client is None:
            success, msg = initialize_client(base_url, model_name)
            if not success:
                empty_chart = pd.DataFrame({"Class": [], "Probability": []})
                return "❌ Connection Failed", msg, empty_chart, ""
        
        # Run inference
        result = client.run_inference(image)
        
        if result['success']:
            # Format results
            prediction_text = f"🔍 **Prediction: {result['prediction']}**"
            confidence_text = f"📊 **Confidence: {result['confidence']:.2%}**"
            time_text = f"⏱️ **Inference Time: {result['inference_time']:.3f}s**"
            
            status_message = f"{prediction_text}\n{confidence_text}\n{time_text}"
            
            # Create probability chart data
            prob_chart = pd.DataFrame({
                "Class": ["Normal", "Pneumonia"],
                "Probability": [result['probabilities']['NORMAL'], result['probabilities']['PNEUMONIA']]
            })
            
            # Create detailed results
            detailed_results = f"""
## Detailed Results

**Classification:** {result['prediction']}
**Confidence Score:** {result['confidence']:.4f}

**Class Probabilities:**
- Normal: {result['probabilities']['NORMAL']:.4f} ({result['probabilities']['NORMAL']*100:.2f}%)
- Pneumonia: {result['probabilities']['PNEUMONIA']:.4f} ({result['probabilities']['PNEUMONIA']*100:.2f}%)

**Performance:**
- Inference Time: {result['inference_time']:.3f} seconds

---
⚠️ **Disclaimer:** This is a demonstration model and should not be used for actual medical diagnosis. 
Always consult with qualified healthcare professionals for medical decisions.
"""
            
            return status_message, detailed_results, prob_chart, "✅ Inference completed successfully"
            
        else:
            error_msg = result.get('error', 'Unknown error occurred')
            empty_chart = pd.DataFrame({"Class": [], "Probability": []})
            return f"❌ Inference Failed", f"Error: {error_msg}", empty_chart, "❌ Inference failed"
            
    except Exception as e:
        empty_chart = pd.DataFrame({"Class": [], "Probability": []})
        return f"❌ Unexpected Error", f"An unexpected error occurred: {str(e)}", empty_chart, "❌ Unexpected error"


def create_gradio_interface():
    """Create the Gradio interface"""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    .error {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
    }
    """
    
    with gr.Blocks(
        title="Pneumonia Classification - Chest X-Ray Analysis",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as interface:
        
        gr.Markdown("""
        # 🫁 Pneumonia Classification System
        
        Upload a chest X-ray image to get an AI-powered assessment for pneumonia detection.
        
        **Instructions:**
        1. Configure your model endpoint settings below
        2. Upload a chest X-ray image (JPEG, PNG formats supported)
        3. Click "Analyze X-Ray" to get the prediction
        
        ⚠️ **Medical Disclaimer:** This tool is for demonstration purposes only and should not be used for actual medical diagnosis.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔧 Configuration")
                
                base_url_input = gr.Textbox(
                    label="Model Endpoint URL",
                    placeholder="https://<CAII_DOMAIN>/namespaces/serving-default/endpoints/<endpoint_name>",
                    value="",
                    info="Base URL for your model inference endpoint, like https://<CAII_DOMAIN>/namespaces/serving-default/endpoints/<endpoint_name>"
                )
                
                model_name_input = gr.Textbox(
                    label="Model Name",
                    placeholder="pneumonia_onnx_classifier",
                    value="pneumonia_onnx_classifier",
                    info="ID of the deployed model. It can be found in the endpoint details page."
                )
                
                test_connection_btn = gr.Button("🔗 Test Connection", variant="secondary")
                connection_status = gr.Textbox(
                    label="Connection Status",
                    interactive=False,
                    max_lines=5
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### 📸 Image Upload")
                
                image_input = gr.Image(
                    label="Upload Chest X-Ray Image",
                    type="pil",
                    height=300
                )
                
                analyze_btn = gr.Button("🔍 Analyze X-Ray", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Results")
                
                prediction_output = gr.Textbox(
                    label="Prediction Result",
                    interactive=False,
                    max_lines=3
                )
                
                probability_chart = gr.BarPlot(
                    x="Class",
                    y="Probability",
                    title="Classification Probabilities",
                    x_title="Diagnosis Class",
                    y_title="Probability Score",
                    height=300
                )
                
            with gr.Column():
                gr.Markdown("### 📋 Detailed Analysis")
                
                detailed_output = gr.Markdown(
                    value="Upload an image and click 'Analyze X-Ray' to see detailed results."
                )
        
        with gr.Row():
            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                visible=False
            )
        
        # Event handlers
        test_connection_btn.click(
            fn=lambda url, name: initialize_client(url, name)[1],
            inputs=[base_url_input, model_name_input],
            outputs=[connection_status]
        )
        
        analyze_btn.click(
            fn=predict_pneumonia,
            inputs=[image_input, base_url_input, model_name_input],
            outputs=[prediction_output, detailed_output, probability_chart, status_output]
        )
        
        # Example images section
        gr.Markdown("""
        ### 📝 Usage Tips
        
        - **Image Quality:** Use clear, high-resolution chest X-ray images for best results
        - **Format:** JPEG, PNG, and other common image formats are supported
        - **Preprocessing:** Images are automatically resized and normalized for the model
        - **Privacy:** Images are processed locally and not stored permanently
        
        ### 🔧 Configuration Help
        
        - **Endpoint URL:** Should point to your Cloudera AI Inference model endpoint. It can be found in the endpoint details page.
        - **Model Name:** Must match the exact model ID of your deployed ONNX model. It can be found in the AI Registry.
        - **Authentication:** Ensure your token file is properly configured at `/tmp/jwt`
        """)
    
    return interface


def main():
    """Main entry point"""
    # Create and launch the Gradio interface
    interface = create_gradio_interface()
    
    # Launch with configuration
    interface.launch(
        server_name="localhost",  # Allow external access
        server_port=int(os.environ.get('CDSW_APP_PORT')),
        share=False,            # Set to True to create public link
        debug=True,             # Enable debug mode
        show_error=True         # Show error messages
    )


if __name__ == "__main__":
    main() 