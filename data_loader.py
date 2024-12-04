import tensorflow_datasets as tfds
import numpy as np
import cv2

def load_xarm_dataset():
    """Load the UTokyo XArm pick and place dataset."""
    dataset = tfds.load('utokyo_xarm_pick_and_place_converted_externally_to_rlds')
    return dataset

def extract_episode_frames(episode):
    """Extract frames from a single episode."""
    frames = []
    for step in episode['steps']:
        # Convert tensor to numpy array
        frame = step['observation']['image'].numpy()
        frames.append(frame)
    return frames

# Example usage
def main():
    # Load dataset
    dataset = load_xarm_dataset()
    
    # Take first episode as example
    episode = next(iter(dataset['train']))
    
    # Extract frames
    frames = extract_episode_frames(episode)
    
    print(f"Extracted {len(frames)} frames from episode")
    
    return frames

if __name__ == "__main__":
    main()