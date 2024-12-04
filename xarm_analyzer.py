import tensorflow_datasets as tfds
import numpy as np
import cv2
from openai import OpenAI
import os
from typing import List, Dict
import base64
from io import BytesIO
from PIL import Image

class XArmAnalyzer:
    def __init__(self, api_key: str):
        """Initialize analyzer with OpenAI API key."""
        self.client = OpenAI(api_key=api_key)
        
    def load_dataset(self):
        """Load the UTokyo XArm pick and place dataset."""
        print("Loading dataset...")
        dataset = tfds.load('utokyo_xarm_pick_and_place_converted_externally_to_rlds')
        return dataset
        
    def process_episode(self, episode):
        """Extract and analyze frames from an episode."""
        frames = []
        for step in episode['steps']:
            # Convert tensor to numpy array
            frame = step['observation']['image'].numpy()
            frames.append(frame)
            
        print(f"Extracted {len(frames)} frames")
        return self.analyze_sequence(frames)
    
    def _encode_frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64 for OpenAI API."""
        pil_image = Image.fromarray(frame)
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """Analyze a single frame with GPT-4V."""
        base64_frame = self._encode_frame_to_base64(frame)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": ("Analyze this robot pick and place frame. "
                                       "What is happening in this exact moment? "
                                       "Is this a key learning moment where the robot "
                                       "achieves something significant? "
                                       "Format response as: ACTION: <action> | LEARNING: <yes/no> | REASON: <reason>")
                            },
                            {
                                "type": "image",
                                "image_url": f"data:image/jpeg;base64,{base64_frame}"
                            }
                        ]
                    }
                ],
                max_tokens=100
            )
            return self._parse_response(response.choices[0].message.content)
        except Exception as e:
            print(f"Error analyzing frame: {e}")
            return {"action": "unknown", "is_learning": False, "reason": "error"}
    
    def _parse_response(self, response: str) -> Dict:
        """Parse GPT-4V response into structured data."""
        try:
            parts = response.split('|')
            action = parts[0].split(':')[1].strip()
            is_learning = 'yes' in parts[1].lower()
            reason = parts[2].split(':')[1].strip()
            return {
                "action": action,
                "is_learning": is_learning,
                "reason": reason
            }
        except:
            return {"action": "unknown", "is_learning": False, "reason": "parsing error"}
    
    def analyze_sequence(self, frames: List[np.ndarray]) -> List[Dict]:
        """Analyze frames to identify learning moments."""
        results = []
        # Sample every 5th frame to reduce API calls
        for i, frame in enumerate(frames[::5]):
            print(f"Analyzing frame {i*5}/{len(frames)}")
            analysis = self.analyze_frame(frame)
            analysis['frame_index'] = i*5
            results.append(analysis)
            if analysis['is_learning']:
                print(f"Learning moment detected at frame {i*5}!")
                print(f"Action: {analysis['action']}")
                print(f"Reason: {analysis['reason']}")
                print("---")
        return results

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
        
    analyzer = XArmAnalyzer(api_key)
    
    # Load dataset and get first episode
    dataset = analyzer.load_dataset()
    first_episode = next(iter(dataset['train']))
    
    # Analyze episode
    results = analyzer.process_episode(first_episode)
    
    # Print summary of learning moments
    print("\nSummary of Learning Moments:")
    learning_moments = [r for r in results if r['is_learning']]
    for moment in learning_moments:
        print(f"Frame {moment['frame_index']}:")
        print(f"Action: {moment['action']}")
        print(f"Reason: {moment['reason']}")
        print("---")

if __name__ == "__main__":
    main()