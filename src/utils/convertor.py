import os
import argparse
from PIL import Image
from pdf2image import convert_from_path

def convert_image(input_path, output_path, output_format):
    with Image.open(input_path) as img:
        if output_format == 'JPG':
            output_format = 'JPEG'
        img.save(output_path, output_format)

def convert_pdf_to_jpg(input_path, output_dir):
    pages = convert_from_path(input_path)
    for i, page in enumerate(pages):
        output_path = os.path.join(output_dir, f"page_{i + 1}.jpg")
        page.save(output_path, 'JPEG')

def bulk_convert(input_dir, output_dir, output_format):
    os.makedirs(output_dir, exist_ok=True)
    
    for root, _, files in os.walk(input_dir):
        for filename in files:
            input_path = os.path.join(root, filename)
            relative_path = os.path.relpath(root, input_dir)
            output_subdir = os.path.join(output_dir, relative_path)
            os.makedirs(output_subdir, exist_ok=True)

            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                name, _ = os.path.splitext(filename)
                output_path = os.path.join(output_subdir, f"{name}.{output_format.lower()}")
                convert_image(input_path, output_path, output_format)
                print(f"Converted {filename} to {output_format}")
            elif filename.lower().endswith('.pdf'):
                pdf_output_dir = os.path.join(output_subdir, os.path.splitext(filename)[0])
                os.makedirs(pdf_output_dir, exist_ok=True)
                convert_pdf_to_jpg(input_path, pdf_output_dir)
                print(f"Converted {filename} to JPG")
            else:
                print(f"Skipping unsupported file: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Bulk image and PDF converter")
    parser.add_argument("input_dir", help="Input directory containing images and PDFs")
    parser.add_argument("output_dir", help="Output directory for converted files")
    parser.add_argument(
        "--format", 
        choices=['jpg', 'png', 'jpeg'], 
        default='jpg', 
        help="Output format for images (default: jpg)"
    )
    
    args = parser.parse_args()
    
    bulk_convert(args.input_dir, args.output_dir, args.format.upper())

if __name__ == "__main__":
    main()
