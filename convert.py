from PIL import Image
import os

def gif_to_jpg(gif_path, output_folder):
    # Open the GIF image
    with Image.open(gif_path) as gif:
        # Get the base name of the file (without the extension)
        base_name = os.path.basename(gif_path)
        
        # Loop through each frame in the GIF
        for frame in range(gif.n_frames):
            gif.seek(frame)  # Move to the current frame
            
            # Convert to RGB mode (GIFs are often in 'P' mode, which doesn't support JPG directly)
            frame_image = gif.convert("RGB")
            
            # Save each frame as a JPG file
            output_path = os.path.join(output_folder, f"{base_name}_frame_{frame + 1}.jpg")
            frame_image.save(output_path, "JPEG")
            print(f"Saved: {output_path}")
            
if not os.path.exists('data_new'):
    os.makedirs('data_new')
            
##get all images in data directory
gif_files = [f for f in os.listdir('data') if f.startswith('subject')]
for gif_file in gif_files:
    gif_path = os.path.join('data', gif_file)
    gif_to_jpg(gif_path, 'data_new')
    

    

    

    
